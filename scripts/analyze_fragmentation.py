#!/usr/bin/env python3
"""SVG Vectorization Quality Assessment Tool.

Analyzes vectorization pipeline output SVGs for fragmentation, geometric
quality, coverage/overlap, and (with --original) color fidelity metrics.

Usage:
    python scripts/analyze_fragmentation.py output.svg
    python scripts/analyze_fragmentation.py output.svg --original source.png
    python scripts/analyze_fragmentation.py --compare baseline/ experiment/
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.draw import polygon as draw_polygon
from svgpathtools import parse_path

try:
    from skimage.color import deltaE_ciede2000, rgb2lab

    _HAS_SKIMAGE_COLOR = True
except ImportError:
    _HAS_SKIMAGE_COLOR = False

# ── helpers ──────────────────────────────────────────────────────────────────


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def hex_to_rgb_float(h: str) -> tuple[float, float, float]:
    r, g, b = hex_to_rgb(h)
    return r / 255.0, g / 255.0, b / 255.0


def polygon_area_signed(verts):
    n = len(verts)
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += verts[i][0] * verts[j][1] - verts[j][0] * verts[i][1]
    return a / 2.0


def polygon_perimeter(verts):
    n = len(verts)
    if n < 2:
        return 0.0
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        dx = verts[j][0] - verts[i][0]
        dy = verts[j][1] - verts[i][1]
        s += math.hypot(dx, dy)
    return s


def gini(values):
    if len(values) < 2:
        return 0.0
    xs = sorted(values)
    n = len(xs)
    total = sum(xs)
    if total <= 0:
        return 0.0
    cum = 0.0
    g = 0.0
    for i, v in enumerate(xs):
        cum += v
        g += (2 * (i + 1) - n - 1) * v
    return g / (n * total)


def generate_tints(hex_color: str, n: int):
    r, g, b = hex_to_rgb(hex_color)
    h, l_base, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    s = max(s, 0.25)
    tints = []
    for i in range(n):
        l_new = 0.25 + 0.55 * (i / max(n - 1, 1))
        nr, ng, nb = colorsys.hls_to_rgb(h, l_new, s)
        tints.append((nr, ng, nb))
    return tints


# ── data structures ──────────────────────────────────────────────────────────


@dataclass
class ShapeInfo:
    index: int
    fill_color: str
    is_stroke: bool
    subpaths: list
    area_vb: float = 0.0
    perimeter: float = 0.0
    num_segments: int = 0
    mask: Optional[np.ndarray] = field(default=None, repr=False)


# ── SVG parsing ──────────────────────────────────────────────────────────────

_SVG_NS = "{http://www.w3.org/2000/svg}"


def _split_subpaths(path, samples_per_seg: int):
    if not path:
        return []
    groups: list[list] = []
    cur: list = [path[0]]
    for i in range(1, len(path)):
        if abs(path[i].start - path[i - 1].end) > 0.5:
            groups.append(cur)
            cur = []
        cur.append(path[i])
    if cur:
        groups.append(cur)

    subpaths = []
    for grp in groups:
        pts = []
        for seg in grp:
            for t in np.linspace(0, 1, samples_per_seg, endpoint=False):
                pt = seg.point(t)
                pts.append((pt.real, pt.imag))
        if grp:
            pt = grp[-1].end
            pts.append((pt.real, pt.imag))
        if len(pts) >= 3:
            subpaths.append(pts)
    return subpaths


def parse_svg(svg_path: str, samples_per_seg: int = 10):
    tree = ET.parse(svg_path)
    root = tree.getroot()

    vb = root.get("viewBox")
    if vb:
        parts = vb.split()
        vb_w, vb_h = float(parts[2]), float(parts[3])
    else:
        vb_w = float(root.get("width", 100))
        vb_h = float(root.get("height", 100))

    paths = list(root.iter(f"{_SVG_NS}path"))
    if not paths:
        paths = list(root.iter("path"))

    shapes: list[ShapeInfo] = []
    for i, elem in enumerate(paths):
        fill = elem.get("fill", "none")
        stroke = elem.get("stroke", "none")
        d = elem.get("d", "")
        if not d:
            continue

        is_stroke = fill == "none" and stroke != "none"
        color = stroke if is_stroke else fill
        if color == "none":
            continue

        try:
            path_obj = parse_path(d)
        except Exception:
            continue

        subpaths = _split_subpaths(path_obj, samples_per_seg)
        if not subpaths:
            continue

        perim = sum(polygon_perimeter(sp) for sp in subpaths)

        shapes.append(
            ShapeInfo(
                index=i,
                fill_color=color.upper(),
                is_stroke=is_stroke,
                subpaths=subpaths,
                perimeter=perim,
                num_segments=len(path_obj),
            )
        )
    return shapes, vb_w, vb_h


# ── rasterization ────────────────────────────────────────────────────────────


def rasterize_shape(subpaths, rw: int, rh: int, vb_w: float, vb_h: float) -> np.ndarray:
    sx, sy = rw / vb_w, rh / vb_h
    mask = np.zeros((rh, rw), dtype=bool)
    for sp in subpaths:
        xs = np.clip([p[0] * sx for p in sp], 0, rw - 1)
        ys = np.clip([p[1] * sy for p in sp], 0, rh - 1)
        rr, cc = draw_polygon(ys, xs, (rh, rw))
        sub = np.zeros_like(mask)
        sub[rr, cc] = True
        mask ^= sub
    return mask


def rasterize_all(shapes, rw, rh, vb_w, vb_h):
    pixel_scale = (vb_w * vb_h) / (rw * rh)
    for s in shapes:
        s.mask = rasterize_shape(s.subpaths, rw, rh, vb_w, vb_h)
        s.area_vb = int(np.count_nonzero(s.mask)) * pixel_scale


def render_shapes_rgb(shapes, rw, rh):
    img = np.ones((rh, rw, 3), dtype=np.float32)
    for s in shapes:
        if s.mask is not None and s.fill_color:
            rgb = hex_to_rgb_float(s.fill_color)
            img[s.mask] = rgb
    return img


# ── Union-Find ───────────────────────────────────────────────────────────────


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.r[a] < self.r[b]:
            a, b = b, a
        self.p[b] = a
        if self.r[a] == self.r[b]:
            self.r[a] += 1

    def components(self, n):
        return len({self.find(i) for i in range(n)})


# ── fragmentation metrics ───────────────────────────────────────────────────


def compute_fragmentation(shapes, color_groups):
    fill_shapes = [s for s in shapes if not s.is_stroke]
    total_shapes = len(fill_shapes)
    total_components = 0

    kern = np.ones((3, 3), dtype=bool)

    per_color_frags: dict[str, dict] = {}

    for color, indices in color_groups.items():
        n = len(indices)
        if n <= 1:
            total_components += n
            per_color_frags[color] = {"fragments": n, "components": n}
            continue

        uf = UnionFind(n)
        dilated = [binary_dilation(shapes[idx].mask, kern) for idx in indices]
        for i in range(n):
            for j in range(i + 1, n):
                if np.any(dilated[i] & shapes[indices[j]].mask):
                    uf.union(i, j)
        comps = uf.components(n)
        total_components += comps
        per_color_frags[color] = {"fragments": n, "components": comps}

    mergeable_ratio = total_shapes / max(total_components, 1)

    vb_area = sum(s.area_vb for s in fill_shapes)
    tiny_thresh = vb_area * 0.0001 if vb_area > 0 else 1.0
    tiny_count = sum(1 for s in fill_shapes if s.area_vb < tiny_thresh)
    tiny_rate = tiny_count / max(total_shapes, 1)

    all_areas_by_color = defaultdict(list)
    for s in fill_shapes:
        all_areas_by_color[s.fill_color].append(s.area_vb)
    gini_values = [gini(areas) for areas in all_areas_by_color.values() if len(areas) > 1]
    gini_avg = float(np.mean(gini_values)) if gini_values else 0.0

    return {
        "total_shapes": total_shapes,
        "unique_colors": len(color_groups),
        "total_components": total_components,
        "mergeable_ratio": round(mergeable_ratio, 3),
        "tiny_fragment_count": tiny_count,
        "tiny_fragment_rate": round(tiny_rate, 4),
        "gini_coefficient": round(gini_avg, 4),
        "per_color": per_color_frags,
    }


# ── geometry metrics ─────────────────────────────────────────────────────────


def compute_geometry(shapes):
    fill_shapes = [s for s in shapes if not s.is_stroke]
    circularities = []
    complexities = []

    for s in fill_shapes:
        if s.area_vb > 0 and s.perimeter > 0:
            circ = (s.perimeter**2) / (4 * math.pi * s.area_vb)
            circularities.append(circ)
        complexities.append(s.num_segments)

    circularities.sort()
    complexities.sort()

    def percentile(arr, p):
        if not arr:
            return 0.0
        idx = min(int(p / 100.0 * len(arr)), len(arr) - 1)
        return arr[idx]

    sliver_thresh = 20.0
    sliver_count = sum(1 for c in circularities if c > sliver_thresh)

    return {
        "circularity_median": round(percentile(circularities, 50), 2),
        "circularity_p95": round(percentile(circularities, 95), 2),
        "circularity_max": round(circularities[-1], 2) if circularities else 0.0,
        "sliver_count": sliver_count,
        "path_complexity_median": int(percentile(complexities, 50)),
        "path_complexity_p95": int(percentile(complexities, 95)),
        "path_complexity_max": max(complexities) if complexities else 0,
    }


def detect_islands(shapes, rw, rh):
    kern = np.ones((5, 5), dtype=bool)
    fill_shapes = [s for s in shapes if not s.is_stroke and s.mask is not None]
    islands = []
    for i, si in enumerate(fill_shapes):
        border = binary_dilation(si.mask, kern) & ~si.mask
        if not np.any(border):
            continue
        touching = set()
        for j, sj in enumerate(fill_shapes):
            if i == j:
                continue
            if np.any(border & sj.mask):
                touching.add(j)
        if len(touching) == 1:
            islands.append(si.index)
    return islands


# ── coverage / overlap ───────────────────────────────────────────────────────


def compute_coverage_overlap(shapes, rw, rh):
    cov_map = np.zeros((rh, rw), dtype=np.int32)
    for s in shapes:
        if s.mask is not None:
            cov_map += s.mask.astype(np.int32)

    total = rw * rh
    covered = int(np.count_nonzero(cov_map > 0))
    overlapping = int(np.count_nonzero(cov_map > 1))

    return {
        "coverage": round(covered / total, 5),
        "overlap": round(overlapping / total, 5),
        "gap_pixels": total - covered,
        "overlap_pixels": overlapping,
    }


def compute_same_color_gaps(shapes, color_groups, rw, rh):
    kern = np.ones((3, 3), dtype=bool)
    gap_pixels = 0
    for _color, indices in color_groups.items():
        if len(indices) < 2:
            continue
        union_mask = np.zeros((rh, rw), dtype=bool)
        for idx in indices:
            union_mask |= shapes[idx].mask
        dilated_union = binary_dilation(union_mask, kern)
        gap_region = dilated_union & ~union_mask
        for idx in indices:
            gap_region &= ~shapes[idx].mask
        gap_pixels += int(np.count_nonzero(gap_region))
    return gap_pixels


# ── color fidelity ───────────────────────────────────────────────────────────


def compute_color_fidelity(svg_img_f32, original_path, rw, rh):
    if not _HAS_SKIMAGE_COLOR:
        return None, None

    orig = Image.open(original_path).convert("RGB").resize((rw, rh), Image.LANCZOS)
    orig_arr = np.array(orig, dtype=np.float64) / 255.0
    svg_arr = svg_img_f32[:, :, :3].astype(np.float64)

    lab_orig = rgb2lab(orig_arr)
    lab_svg = rgb2lab(svg_arr)
    de = deltaE_ciede2000(lab_orig, lab_svg)

    sorted_de = np.sort(de.ravel())
    n = len(sorted_de)

    orig_colors = len(np.unique(np.array(orig).reshape(-1, 3), axis=0))
    svg_rgb_u8 = (np.clip(svg_arr, 0, 1) * 255).astype(np.uint8)
    svg_colors = len(np.unique(svg_rgb_u8.reshape(-1, 3), axis=0))

    metrics = {
        "delta_e_mean": round(float(np.mean(de)), 3),
        "delta_e_p95": round(float(sorted_de[min(int(0.95 * n), n - 1)]), 3),
        "delta_e_max": round(float(sorted_de[-1]), 3),
        "svg_unique_colors": svg_colors,
        "original_unique_colors": orig_colors,
        "color_compression": round(svg_colors / max(orig_colors, 1), 6),
    }
    return metrics, de


def compute_border_delta_e(shapes, rw, rh):
    if not _HAS_SKIMAGE_COLOR:
        return 0.0

    shape_map = np.full((rh, rw), -1, dtype=np.int32)
    for i, s in enumerate(shapes):
        if s.mask is not None:
            shape_map[s.mask] = i

    adj_pairs: set[tuple[int, int]] = set()
    diff_h = shape_map[:, 1:] != shape_map[:, :-1]
    for r, c in zip(*np.where(diff_h)):
        a, b = int(shape_map[r, c]), int(shape_map[r, c + 1])
        if a >= 0 and b >= 0 and a != b:
            adj_pairs.add((min(a, b), max(a, b)))
    diff_v = shape_map[1:, :] != shape_map[:-1, :]
    for r, c in zip(*np.where(diff_v)):
        a, b = int(shape_map[r, c]), int(shape_map[r + 1, c])
        if a >= 0 and b >= 0 and a != b:
            adj_pairs.add((min(a, b), max(a, b)))

    if not adj_pairs:
        return 0.0

    des = []
    for i, j in adj_pairs:
        if shapes[i].fill_color == shapes[j].fill_color:
            continue
        c1 = np.array([[hex_to_rgb(shapes[i].fill_color)]], dtype=np.float64) / 255.0
        c2 = np.array([[hex_to_rgb(shapes[j].fill_color)]], dtype=np.float64) / 255.0
        lab1 = rgb2lab(c1)
        lab2 = rgb2lab(c2)
        des.append(float(deltaE_ciede2000(lab1, lab2)[0, 0]))

    return round(float(np.mean(des)), 3) if des else 0.0


# ── visualization ────────────────────────────────────────────────────────────


def build_frag_image(shapes, color_groups, rw, rh):
    img = np.ones((rh, rw, 3), dtype=np.float32)
    border_mask = np.zeros((rh, rw), dtype=bool)
    kern = np.ones((3, 3), dtype=bool)

    for color, indices in color_groups.items():
        if len(indices) == 1:
            rgb = hex_to_rgb_float(color)
            img[shapes[indices[0]].mask] = rgb
        else:
            tints = generate_tints(color, len(indices))
            for k, idx in enumerate(indices):
                m = shapes[idx].mask
                img[m] = tints[k]
                edge = binary_dilation(m, kern) & ~m
                border_mask |= edge

    img[border_mask] = (0.0, 0.0, 0.0)
    return img


def visualize(
    shapes,
    color_groups,
    metrics,
    svg_img,
    frag_img,
    de_map,
    original_path,
    rw,
    rh,
    save_path=None,
):
    has_orig = original_path is not None and de_map is not None

    fig = plt.figure(figsize=(18, 14 if has_orig else 10))

    if has_orig:
        gs = GridSpec(3, 2, figure=fig, height_ratios=[4, 4, 3], hspace=0.25, wspace=0.15)
        ax_orig = fig.add_subplot(gs[0, 0])
        ax_svg = fig.add_subplot(gs[0, 1])
        ax_frag = fig.add_subplot(gs[1, 0])
        ax_de = fig.add_subplot(gs[1, 1])
        ax_stats = fig.add_subplot(gs[2, :])
    else:
        gs = GridSpec(2, 2, figure=fig, height_ratios=[4, 3], hspace=0.25, wspace=0.15)
        ax_svg = fig.add_subplot(gs[0, 0])
        ax_frag = fig.add_subplot(gs[0, 1])
        ax_stats = fig.add_subplot(gs[1, :])
        ax_orig = None
        ax_de = None

    if ax_orig and original_path:
        orig = Image.open(original_path).convert("RGB").resize((rw, rh), Image.LANCZOS)
        ax_orig.imshow(np.array(orig))
        ax_orig.set_title("Original Image", fontsize=11)
        ax_orig.axis("off")

    ax_svg.imshow(svg_img)
    ax_svg.set_title("SVG Rendering", fontsize=11)
    ax_svg.axis("off")

    ax_frag.imshow(frag_img)
    ax_frag.set_title("Fragmentation Map", fontsize=11)
    ax_frag.axis("off")

    if ax_de is not None and de_map is not None:
        im = ax_de.imshow(de_map, cmap="hot", vmin=0, vmax=max(np.percentile(de_map, 99), 5))
        ax_de.set_title("Delta-E Heatmap (CIEDE2000)", fontsize=11)
        ax_de.axis("off")
        plt.colorbar(im, ax=ax_de, fraction=0.046, pad=0.04, label="ΔE")

    _draw_stats_panel(ax_stats, metrics, color_groups, has_orig)

    fig.suptitle("SVG Vectorization Quality Report", fontsize=14, fontweight="bold", y=0.98)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Report saved to: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _draw_stats_panel(ax, metrics, color_groups, has_orig):
    ax.axis("off")

    frag = metrics["fragmentation"]
    geom = metrics["geometry"]
    cov = metrics["coverage"]
    fid = metrics.get("fidelity")

    lines = [
        f"Shapes: {frag['total_shapes']}   Colors: {frag['unique_colors']}",
        f"Mergeable ratio: {frag['mergeable_ratio']:.2f}  "
        f"({frag['total_shapes']} shapes / {frag['total_components']} components)",
        f"Tiny fragments: {frag['tiny_fragment_rate']*100:.1f}%  "
        f"({frag['tiny_fragment_count']} < 0.01% area)",
        f"Gini coefficient: {frag['gini_coefficient']:.3f}",
        f"Coverage: {cov['coverage']*100:.2f}%   "
        f"Overlap: {cov['overlap']*100:.2f}%   "
        f"Same-color gaps: {metrics.get('same_color_gap_pixels', 0)} px",
        f"Circularity P95: {geom['circularity_p95']:.1f}   "
        f"Slivers (>20): {geom['sliver_count']}   "
        f"Islands: {metrics.get('island_count', 0)}",
        f"Path complexity  median: {geom['path_complexity_median']}   "
        f"P95: {geom['path_complexity_p95']}   "
        f"max: {geom['path_complexity_max']}",
    ]
    if fid:
        lines.append("")
        lines.append(
            f"Delta-E  mean: {fid['delta_e_mean']:.2f}   "
            f"P95: {fid['delta_e_p95']:.2f}   "
            f"max: {fid['delta_e_max']:.2f}"
        )
        lines.append(
            f"Border Delta-E mean: {metrics.get('border_delta_e_mean', 0):.2f}   "
            f"Color compression: {fid['svg_unique_colors']} / {fid['original_unique_colors']}"
        )

    text = "\n".join(lines)
    ax.text(
        0.02,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=9,
        fontfamily="monospace",
        verticalalignment="top",
    )

    sorted_colors = sorted(
        color_groups.items(), key=lambda kv: len(kv[1]), reverse=True
    )[:15]
    if sorted_colors:
        labels = [c[:7] for c, _ in sorted_colors]
        counts = [len(idxs) for _, idxs in sorted_colors]
        bar_colors = [hex_to_rgb_float(c) for c, _ in sorted_colors]

        bar_ax = ax.inset_axes([0.55, 0.05, 0.42, 0.88])
        bars = bar_ax.barh(range(len(labels)), counts, color=bar_colors, edgecolor="gray", linewidth=0.5)
        bar_ax.set_yticks(range(len(labels)))
        bar_ax.set_yticklabels(labels, fontsize=7, fontfamily="monospace")
        bar_ax.invert_yaxis()
        bar_ax.set_xlabel("Fragments", fontsize=8)
        bar_ax.set_title("Top colors by fragment count", fontsize=9)
        for bar, cnt in zip(bars, counts):
            bar_ax.text(
                bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(cnt), va="center", fontsize=7,
            )


# ── JSON export ──────────────────────────────────────────────────────────────


def export_json(metrics, svg_path, original_path, output_path):
    data = {
        "svg_file": str(svg_path),
        "original_file": str(original_path) if original_path else None,
        "total_shapes": metrics["fragmentation"]["total_shapes"],
        "unique_colors": metrics["fragmentation"]["unique_colors"],
        "mergeable_ratio": metrics["fragmentation"]["mergeable_ratio"],
        "tiny_fragment_rate": metrics["fragmentation"]["tiny_fragment_rate"],
        "gini_coefficient": metrics["fragmentation"]["gini_coefficient"],
        "coverage": metrics["coverage"]["coverage"],
        "overlap": metrics["coverage"]["overlap"],
        "same_color_gap_pixels": metrics.get("same_color_gap_pixels", 0),
        "island_count": metrics.get("island_count", 0),
        "circularity_p95": metrics["geometry"]["circularity_p95"],
        "sliver_count": metrics["geometry"]["sliver_count"],
        "path_complexity_median": metrics["geometry"]["path_complexity_median"],
        "path_complexity_p95": metrics["geometry"]["path_complexity_p95"],
    }
    fid = metrics.get("fidelity")
    if fid:
        data.update(
            {
                "delta_e_mean": fid["delta_e_mean"],
                "delta_e_p95": fid["delta_e_p95"],
                "delta_e_max": fid["delta_e_max"],
                "border_delta_e_mean": metrics.get("border_delta_e_mean", 0),
                "color_compression": fid["color_compression"],
            }
        )
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"JSON exported to: {output_path}")


# ── compare mode ─────────────────────────────────────────────────────────────

_COMPARE_KEYS = [
    ("mergeable_ratio", "MergeRatio", "{:.2f}", True),
    ("tiny_fragment_rate", "TinyFrag%", "{:.1%}", True),
    ("coverage", "Coverage%", "{:.2%}", False),
    ("delta_e_mean", "ΔE_mean", "{:.1f}", True),
]

_REGRESSION_RULES = [
    ("delta_e_mean", lambda b, e: e > b + 1.0),
    ("coverage", lambda b, e: e < b - 0.005),
    ("mergeable_ratio", lambda b, e: e > b * 1.2 and e > 1.05),
]


def run_compare(baseline_dir: str, experiment_dir: str):
    bd = Path(baseline_dir)
    ed = Path(experiment_dir)
    bf = {f.stem: f for f in bd.glob("*.json")}
    ef = {f.stem: f for f in ed.glob("*.json")}
    common = sorted(set(bf) & set(ef))

    if not common:
        print("No matching JSON files found.")
        return

    header = f"{'Image':<22}"
    for _, label, _, _ in _COMPARE_KEYS:
        header += f" {label:<16}"
    header += " Verdict"
    print(f"=== Comparison: {bd}/ vs {ed}/ ===")
    print(header)
    print("-" * len(header))

    for name in common:
        with open(bf[name]) as f:
            bl = json.load(f)
        with open(ef[name]) as f:
            ex = json.load(f)

        verdict = "OK"
        for key, rule in _REGRESSION_RULES:
            bv = bl.get(key)
            ev = ex.get(key)
            if bv is not None and ev is not None and rule(bv, ev):
                verdict = "REGRESSED ⚠"
                break

        if verdict == "OK":
            improved = False
            for key, _, _, lower_better in _COMPARE_KEYS:
                bv = bl.get(key)
                ev = ex.get(key)
                if bv is not None and ev is not None:
                    if lower_better and ev < bv - 0.01:
                        improved = True
                    elif not lower_better and ev > bv + 0.001:
                        improved = True
            verdict = "IMPROVED" if improved else "UNCHANGED"

        row = f"{name:<22}"
        for key, _, fmt, _ in _COMPARE_KEYS:
            bv = bl.get(key)
            ev = ex.get(key)
            if bv is not None and ev is not None:
                cell = f"{fmt.format(bv)}→{fmt.format(ev)}"
            else:
                cell = "N/A"
            row += f" {cell:<16}"
        row += f" {verdict}"
        print(row)


# ── terminal report ──────────────────────────────────────────────────────────


def print_report(metrics):
    frag = metrics["fragmentation"]
    geom = metrics["geometry"]
    cov = metrics["coverage"]
    fid = metrics.get("fidelity")

    print("\n=== SVG Vectorization Quality Report ===")
    print(f"Total shapes:           {frag['total_shapes']}")
    print(f"Unique colors:          {frag['unique_colors']}")
    print(
        f"Mergeable ratio:        {frag['mergeable_ratio']:.2f}  "
        f"({frag['total_shapes']} shapes / {frag['total_components']} components)"
    )
    print(
        f"Tiny fragment rate:     {frag['tiny_fragment_rate']*100:.1f}%  "
        f"({frag['tiny_fragment_count']} shapes < 0.01% area)"
    )
    print(f"Gini coefficient:       {frag['gini_coefficient']:.3f}")
    print(f"Coverage:               {cov['coverage']*100:.2f}%")
    print(f"Overlap:                {cov['overlap']*100:.2f}%")
    print(f"Same-color gap pixels:  {metrics.get('same_color_gap_pixels', 0)}")
    print(
        f"Circularity P95:        {geom['circularity_p95']:.1f}  "
        f"(slivers >20: {geom['sliver_count']})"
    )
    print(
        f"Path complexity:        median={geom['path_complexity_median']}  "
        f"P95={geom['path_complexity_p95']}  max={geom['path_complexity_max']}"
    )
    print(f"Islands:                {metrics.get('island_count', 0)}")

    if fid:
        print("\n--- Color Fidelity (with original) ---")
        print(f"Delta-E mean:           {fid['delta_e_mean']:.2f}")
        print(f"Delta-E P95:            {fid['delta_e_p95']:.2f}")
        print(f"Delta-E max:            {fid['delta_e_max']:.2f}")
        print(f"Border Delta-E mean:    {metrics.get('border_delta_e_mean', 0):.2f}")
        print(
            f"Color compression:      {fid['svg_unique_colors']} / "
            f"{fid['original_unique_colors']} "
            f"({fid['color_compression']:.4%})"
        )
    print()


# ── main ─────────────────────────────────────────────────────────────────────


def analyze(svg_path, original_path=None, resolution=1000, save_path=None, json_path=None):
    print(f"Parsing SVG: {svg_path}")
    shapes, vb_w, vb_h = parse_svg(svg_path)
    if not shapes:
        print("No shapes found in SVG.")
        return

    aspect = vb_w / max(vb_h, 1)
    if aspect >= 1:
        rw = resolution
        rh = max(1, int(resolution / aspect))
    else:
        rh = resolution
        rw = max(1, int(resolution * aspect))

    print(f"Rasterizing {len(shapes)} shapes at {rw}x{rh} ...")
    rasterize_all(shapes, rw, rh, vb_w, vb_h)

    fill_shapes = [s for s in shapes if not s.is_stroke]
    color_groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(fill_shapes):
        color_groups[s.fill_color].append(i)

    print("Computing fragmentation metrics ...")
    frag_metrics = compute_fragmentation(fill_shapes, color_groups)

    print("Computing geometry metrics ...")
    geom_metrics = compute_geometry(fill_shapes)
    islands = detect_islands(fill_shapes, rw, rh)

    print("Computing coverage/overlap ...")
    cov_metrics = compute_coverage_overlap(fill_shapes, rw, rh)
    gap_px = compute_same_color_gaps(fill_shapes, color_groups, rw, rh)

    metrics: dict = {
        "fragmentation": frag_metrics,
        "geometry": geom_metrics,
        "coverage": cov_metrics,
        "island_count": len(islands),
        "same_color_gap_pixels": gap_px,
    }

    de_map = None
    if original_path:
        print("Computing color fidelity ...")
        svg_img = render_shapes_rgb(fill_shapes, rw, rh)
        fid_metrics, de_map = compute_color_fidelity(svg_img, original_path, rw, rh)
        if fid_metrics:
            metrics["fidelity"] = fid_metrics
            bde = compute_border_delta_e(fill_shapes, rw, rh)
            metrics["border_delta_e_mean"] = bde

    print_report(metrics)

    if json_path:
        export_json(metrics, svg_path, original_path, json_path)

    if save_path or not json_path:
        svg_img = render_shapes_rgb(fill_shapes, rw, rh)
        frag_img = build_frag_image(fill_shapes, color_groups, rw, rh)
        visualize(
            fill_shapes, color_groups, metrics,
            svg_img, frag_img, de_map,
            original_path, rw, rh, save_path,
        )


def main():
    parser = argparse.ArgumentParser(
        description="SVG Vectorization Quality Assessment Tool",
        usage=(
            "%(prog)s <svg> [--original IMG] [--save OUT] [--json OUT] [--resolution N]\n"
            "       %(prog)s --compare BASELINE_DIR EXPERIMENT_DIR"
        ),
    )
    parser.add_argument("svg", nargs="?", help="Path to SVG file to analyze")
    parser.add_argument("--original", help="Path to original raster image (enables color fidelity)")
    parser.add_argument("--save", help="Save report image to file")
    parser.add_argument("--json", help="Export metrics to JSON file")
    parser.add_argument(
        "--resolution", type=int, default=1000, help="Rasterization resolution (default 1000)"
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("BASELINE", "EXPERIMENT"),
        help="Compare two directories of JSON result files",
    )
    args = parser.parse_args()

    if args.compare:
        run_compare(args.compare[0], args.compare[1])
    elif args.svg:
        analyze(args.svg, args.original, args.resolution, args.save, args.json)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
