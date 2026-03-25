#!/usr/bin/env python3
"""
Analyze color gamut coverage and DeltaE distribution for ColorDB / ModelPack.

Supports three progressive analysis modes:
  A) Individual ColorDB analysis
  B) Merged multi-ColorDB analysis (--merge)
  C) ColorDB + ModelPack joint analysis (--model-pack)

Examples:
  # Single DB
  python -m modeling.eval.analyze_color_gamut \\
    --db data/dbs/PLA/BambooLab/RYBW_008_5L.json

  # Multiple DBs, each analyzed individually + merged
  python -m modeling.eval.analyze_color_gamut \\
    --db data/dbs/PLA/BambooLab/RYBW_008_5L.json \\
    --db data/dbs/PLA/BambooLab/CMYW_008_5L.json \\
    --db data/dbs/PLA/BambooLab/RYBWCMKG_008_5L_001.json \\
    --merge

  # DB + ModelPack joint
  python -m modeling.eval.analyze_color_gamut \\
    --db data/dbs/PLA/BambooLab/RYBWCMKG_008_5L_001.json \\
    --model-pack data/model_pack/model_package.json --merge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import pdist

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_MAX_PAIRWISE = 8000
_PAIR_COUNT_EXACT_LIMIT = 12000
_SRGB_STEPS_DEFAULT = 21
_CHROMA_GRID_SIZE = 5.0
_DEDUP_THRESHOLD = 1.0


# ── Data Loading ─────────────────────────────────────────────────────────

def load_colordb(path: Path) -> Tuple[str, np.ndarray]:
    with open(path) as f:
        data = json.load(f)
    name = data.get("name", path.stem)
    entries = data.get("entries", [])
    if not entries:
        raise ValueError(f"Empty ColorDB: {path}")
    lab = np.array([e["lab"] for e in entries], dtype=np.float64)
    palette = [ch.get("color", "?") for ch in data.get("palette", [])]
    lh = data.get("layer_height_mm", "?")
    mcl = data.get("max_color_layers", "?")
    desc = (
        f"{name}  "
        f"[{len(palette)} colors: {', '.join(palette)}]  "
        f"lh={lh}  layers={mcl}"
    )
    return desc, lab


def load_model_pack(
    path: Path, modes: Optional[List[str]] = None
) -> List[Tuple[str, np.ndarray]]:
    with open(path) as f:
        data = json.load(f)
    channel_keys = data.get("channel_keys", [])
    results: List[Tuple[str, np.ndarray]] = []
    for mode_name, mode_data in data.get("modes", {}).items():
        if modes and mode_name not in modes:
            continue
        pred_lab = mode_data.get("pred_lab", [])
        if not pred_lab:
            continue
        lab = np.array(pred_lab, dtype=np.float64)
        desc = (
            f"ModelPack [{mode_name}]  "
            f"[{len(channel_keys)} ch: {', '.join(channel_keys[:4])}...]  "
            f"{lab.shape[0]} candidates"
        )
        results.append((desc, lab))
    return results


# ── sRGB Reference Gamut ─────────────────────────────────────────────────

def _srgb_reference_lab(steps: int) -> np.ndarray:
    vals = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    r, g, b = np.meshgrid(vals, vals, vals, indexing="ij")
    rgb = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=1)
    bgr = rgb[:, ::-1].reshape(-1, 1, 3)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).reshape(-1, 3)
    return lab.astype(np.float64)


# ── Core Analysis ────────────────────────────────────────────────────────

def analyze(name: str, lab: np.ndarray, srgb_lab: np.ndarray) -> Dict:
    n = lab.shape[0]
    L, a, b_ch = lab[:, 0], lab[:, 1], lab[:, 2]

    stats: Dict = {"name": name, "n_total": n}

    for ch_tag, ch_data in [("L", L), ("a", a), ("b", b_ch)]:
        stats[f"{ch_tag}_min"] = float(ch_data.min())
        stats[f"{ch_tag}_max"] = float(ch_data.max())
        stats[f"{ch_tag}_mean"] = float(ch_data.mean())
        stats[f"{ch_tag}_std"] = float(ch_data.std())

    tree = cKDTree(lab)

    # nearest neighbour (exclude self)
    k = min(2, n)
    dd, _ = tree.query(lab, k=k)
    nn = dd[:, 1] if n >= 2 else np.array([0.0])
    stats.update(
        nn_de_min=float(nn.min()),
        nn_de_mean=float(nn.mean()),
        nn_de_median=float(np.median(nn)),
        nn_de_p90=float(np.percentile(nn, 90)),
        nn_de_max=float(nn.max()),
        n_unique_de1=int(np.sum(nn >= _DEDUP_THRESHOLD)),
    )
    stats["_nn_dists"] = nn

    # pairwise ΔE76 (subsample for large datasets)
    if n > _MAX_PAIRWISE:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, _MAX_PAIRWISE, replace=False)
        sample = lab[idx]
        stats["pairwise_sampled"] = True
        stats["pairwise_n"] = _MAX_PAIRWISE
    else:
        sample = lab
        stats["pairwise_sampled"] = False
        stats["pairwise_n"] = n

    pw = pdist(sample)
    stats.update(
        de_mean=float(pw.mean()),
        de_median=float(np.median(pw)),
        de_p10=float(np.percentile(pw, 10)),
        de_p90=float(np.percentile(pw, 90)),
        de_max=float(pw.max()),
        de_min=float(pw.min()),
    )

    # near-duplicate pair counts
    total_pairs = n * (n - 1) // 2
    stats["pairs_total"] = total_pairs
    if n <= _PAIR_COUNT_EXACT_LIMIT:
        stats["pairs_de_lt1"] = len(tree.query_pairs(r=1.0))
        stats["pairs_de_lt2"] = len(tree.query_pairs(r=2.0))
        stats["pairs_de_lt5"] = len(tree.query_pairs(r=5.0))
        stats["pairs_estimated"] = False
    else:
        sampled_total = len(pw)
        scale = total_pairs / sampled_total
        stats["pairs_de_lt1"] = int(np.sum(pw < 1.0) * scale)
        stats["pairs_de_lt2"] = int(np.sum(pw < 2.0) * scale)
        stats["pairs_de_lt5"] = int(np.sum(pw < 5.0) * scale)
        stats["pairs_estimated"] = True

    # convex hull volume
    try:
        hull = ConvexHull(lab)
        stats["hull_volume"] = float(hull.volume)
    except Exception:
        stats["hull_volume"] = 0.0

    # sRGB coverage
    d_to_db, _ = tree.query(srgb_lab)
    stats["srgb_coverage_de3"] = float(np.mean(d_to_db < 3.0)) * 100
    stats["srgb_coverage_de5"] = float(np.mean(d_to_db < 5.0)) * 100
    stats["srgb_coverage_de10"] = float(np.mean(d_to_db < 10.0)) * 100

    # lightness bins
    bins_edges = np.linspace(0, 100, 11)
    hist, _ = np.histogram(L, bins=bins_edges)
    stats["lightness_bins"] = hist.tolist()

    # chroma a*-b* grid coverage vs sRGB footprint
    srgb_ab = srgb_lab[:, 1:]
    data_ab = lab[:, 1:]
    a_lo, a_hi = float(srgb_ab[:, 0].min()), float(srgb_ab[:, 0].max())
    b_lo, b_hi = float(srgb_ab[:, 1].min()), float(srgb_ab[:, 1].max())
    a_edges = np.arange(a_lo, a_hi + _CHROMA_GRID_SIZE, _CHROMA_GRID_SIZE)
    b_edges = np.arange(b_lo, b_hi + _CHROMA_GRID_SIZE, _CHROMA_GRID_SIZE)
    srgb_grid, _, _ = np.histogram2d(
        srgb_ab[:, 0], srgb_ab[:, 1], bins=[a_edges, b_edges]
    )
    data_grid, _, _ = np.histogram2d(
        data_ab[:, 0], data_ab[:, 1], bins=[a_edges, b_edges]
    )
    srgb_cells = int(np.count_nonzero(srgb_grid > 0))
    data_in_srgb = int(np.count_nonzero((data_grid > 0) & (srgb_grid > 0)))
    stats["chroma_srgb_cells"] = srgb_cells
    stats["chroma_data_cells"] = data_in_srgb
    stats["chroma_coverage"] = float(data_in_srgb / max(srgb_cells, 1)) * 100

    stats["_lab"] = lab
    stats["_data_grid"] = data_grid
    stats["_srgb_grid"] = srgb_grid
    stats["_a_edges"] = a_edges
    stats["_b_edges"] = b_edges

    return stats


# ── Text Report ──────────────────────────────────────────────────────────

_SEP = "=" * 72


def _short(name: str) -> str:
    return name.split("  ")[0]


def print_stats(s: Dict) -> None:
    print(f"\n{_SEP}\n  {s['name']}\n{_SEP}")

    print(f"\n  条目总数:            {s['n_total']}")
    print(f"  感知独立条目 (NN≥1): {s['n_unique_de1']}")

    print("\n  ── Lab 范围 ──")
    for ch in ("L", "a", "b"):
        print(
            f"  {ch}*: [{s[f'{ch}_min']:>8.2f}, {s[f'{ch}_max']:>8.2f}]  "
            f"均值={s[f'{ch}_mean']:>7.2f}  σ={s[f'{ch}_std']:>6.2f}"
        )

    tag = f" (采样 {s['pairwise_n']})" if s["pairwise_sampled"] else ""
    print(f"\n  ── Pairwise ΔE76{tag} ──")
    print(
        f"  最小={s['de_min']:.2f}  P10={s['de_p10']:.2f}  中位={s['de_median']:.2f}  "
        f"均值={s['de_mean']:.2f}  P90={s['de_p90']:.2f}  最大={s['de_max']:.2f}"
    )

    est = "≈" if s.get("pairs_estimated") else ""
    pt = max(s["pairs_total"], 1)
    print("\n  ── 近似重复对 ──")
    for thr, key in [(1, "pairs_de_lt1"), (2, "pairs_de_lt2"), (5, "pairs_de_lt5")]:
        v = s[key]
        print(f"  ΔE<{thr}: {est}{v:>10,}   ({v / pt * 100:.3f}%)")
    print(f"  总对数:  {s['pairs_total']:>10,}")

    print("\n  ── 最近邻 ΔE76 ──")
    print(
        f"  最小={s['nn_de_min']:.3f}  中位={s['nn_de_median']:.2f}  "
        f"均值={s['nn_de_mean']:.2f}  P90={s['nn_de_p90']:.2f}  最大={s['nn_de_max']:.2f}"
    )

    print("\n  ── 色域覆盖 ──")
    print(f"  Lab 凸包体积:         {s['hull_volume']:>12,.0f}")
    print(f"  sRGB 覆盖 (ΔE<3):    {s['srgb_coverage_de3']:>8.1f} %")
    print(f"  sRGB 覆盖 (ΔE<5):    {s['srgb_coverage_de5']:>8.1f} %")
    print(f"  sRGB 覆盖 (ΔE<10):   {s['srgb_coverage_de10']:>8.1f} %")
    print(
        f"  色度网格覆盖:         {s['chroma_coverage']:>8.1f} %  "
        f"({s['chroma_data_cells']}/{s['chroma_srgb_cells']} 格)"
    )

    print("\n  ── 明度分布 (L* 0–100, 每段 10) ──")
    counts = s["lightness_bins"]
    peak = max(max(counts), 1)
    for i, cnt in enumerate(counts):
        bar = "█" * int(cnt * 40 / peak)
        print(f"  {i * 10:>3}–{(i + 1) * 10:<3}: {cnt:>6} {bar}")


# ── Merge & Incremental ─────────────────────────────────────────────────

def merge_datasets(
    datasets: List[Tuple[str, np.ndarray]],
) -> Tuple[str, np.ndarray]:
    all_lab = np.vstack([lab for _, lab in datasets])
    names = [_short(n) for n, _ in datasets]
    return "Merged: " + " + ".join(names), all_lab


def print_incremental(
    datasets: List[Tuple[str, np.ndarray]], srgb_lab: np.ndarray
) -> None:
    print(f"\n{'-' * 72}\n  增量贡献分析\n{'-' * 72}")
    cumulative: Optional[np.ndarray] = None
    prev_vol = 0.0
    prev_cov = 0.0
    rows: List[Tuple[str, int, float, float, float, float]] = []

    for name, lab in datasets:
        cumulative = lab.copy() if cumulative is None else np.vstack([cumulative, lab])
        try:
            vol = ConvexHull(cumulative).volume
        except Exception:
            vol = 0.0
        d, _ = cKDTree(cumulative).query(srgb_lab)
        cov = float(np.mean(d < 5.0)) * 100
        rows.append((_short(name), cumulative.shape[0], vol, vol - prev_vol, cov, cov - prev_cov))
        prev_vol, prev_cov = vol, cov

    print(f"  {'数据源':<35} {'累计点':>8} {'凸包体积':>12} {'Δ体积':>10} {'覆盖%':>7} {'Δ覆盖':>7}")
    for short, n_cum, vol, dv, cov, dc in rows:
        print(
            f"  + {short:<33} {n_cum:>8,} {vol:>12,.0f} {dv:>+10,.0f} {cov:>7.1f} {dc:>+7.1f}"
        )


# ── Visualization ────────────────────────────────────────────────────────

def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    lab32 = lab.astype(np.float32).reshape(-1, 1, 3)
    bgr = cv2.cvtColor(lab32, cv2.COLOR_Lab2BGR)
    return np.clip(bgr[:, 0, ::-1], 0, 1)


def plot_all(
    all_stats: List[Dict],
    output_dir: Path,
    merged_stats: Optional[Dict] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9})

    _plot_ab_scatter(all_stats, merged_stats, output_dir)
    _plot_lightness_hist(all_stats, output_dir)
    _plot_nn_boxplot(all_stats, merged_stats, output_dir)
    _plot_chroma_heatmap(all_stats, merged_stats, output_dir)
    if len(all_stats) > 1 or merged_stats:
        _plot_comparison_bars(all_stats, merged_stats, output_dir)

    print(f"\n  图表已保存到 {output_dir}/")


def _plot_ab_scatter(
    all_stats: List[Dict], merged: Optional[Dict], out: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    for s in all_stats:
        lab = s["_lab"]
        rgb = _lab_to_rgb(lab)
        ax.scatter(
            lab[:, 1], lab[:, 2], c=rgb, s=3, alpha=0.6,
            label=f"{_short(s['name'])} ({s['n_total']})",
        )
    ax.set_xlabel("a*")
    ax.set_ylabel("b*")
    ax.set_title("Color Gamut  —  a*b* Plane")
    ax.set_aspect("equal")
    ax.axhline(0, color="grey", lw=0.4)
    ax.axvline(0, color="grey", lw=0.4)
    ax.legend(fontsize=7, markerscale=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "gamut_ab_scatter.png", dpi=150)
    plt.close(fig)


def _plot_lightness_hist(all_stats: List[Dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 100, 51)
    for s in all_stats:
        ax.hist(
            s["_lab"][:, 0], bins=bins, alpha=0.35,
            label=_short(s["name"]),
        )
    ax.set_xlabel("L*")
    ax.set_ylabel("Count")
    ax.set_title("Lightness (L*) Distribution")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "gamut_L_histogram.png", dpi=150)
    plt.close(fig)


def _plot_nn_boxplot(
    all_stats: List[Dict], merged: Optional[Dict], out: Path
) -> None:
    entries = list(all_stats)
    if merged:
        entries.append(merged)
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(entries)), 5))
    data = [s["_nn_dists"] for s in entries]
    labels = [_short(s["name"]) for s in entries]
    ax.boxplot(data, tick_labels=labels, vert=True, showfliers=False)
    ax.set_ylabel("Nearest Neighbor ΔE76")
    ax.set_title("NN Distance Distribution (fliers hidden)")
    plt.xticks(rotation=25, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "gamut_nn_de_boxplot.png", dpi=150)
    plt.close(fig)


def _plot_chroma_heatmap(
    all_stats: List[Dict], merged: Optional[Dict], out: Path
) -> None:
    target = merged if merged else all_stats[-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    dg = target["_data_grid"]
    sg = target["_srgb_grid"]
    ae = target["_a_edges"]
    be = target["_b_edges"]
    ext = [be[0], be[-1], ae[0], ae[-1]]

    im1 = axes[0].imshow(
        np.log1p(dg), origin="lower", aspect="auto", extent=ext, cmap="YlOrRd",
    )
    axes[0].set_xlabel("b*")
    axes[0].set_ylabel("a*")
    axes[0].set_title(f"Chroma Density — {_short(target['name'])}")
    fig.colorbar(im1, ax=axes[0], label="log(1 + count)")

    cov_map = np.where(sg > 0, np.where(dg > 0, 1.0, 0.0), np.nan)
    im2 = axes[1].imshow(
        cov_map, origin="lower", aspect="auto", extent=ext,
        cmap="RdYlGn", vmin=0, vmax=1,
    )
    axes[1].set_xlabel("b*")
    axes[1].set_ylabel("a*")
    axes[1].set_title(
        f"Grid Coverage vs sRGB  ({target['chroma_coverage']:.1f}%)"
    )
    fig.colorbar(im2, ax=axes[1], label="Covered / Missing")

    fig.tight_layout()
    fig.savefig(out / "gamut_chroma_heatmap.png", dpi=150)
    plt.close(fig)


def _plot_comparison_bars(
    all_stats: List[Dict], merged: Optional[Dict], out: Path
) -> None:
    entries = list(all_stats)
    if merged:
        entries.append(merged)

    labels = [_short(s["name"]) for s in entries]
    y = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(20, max(5, len(labels) * 0.9 + 1.5)))

    vols = [s["hull_volume"] for s in entries]
    axes[0].barh(y, vols, color="steelblue")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].set_xlabel("Convex Hull Volume (Lab³)")
    axes[0].set_title("Gamut Volume")
    for i, v in enumerate(vols):
        axes[0].text(v, i, f" {v:,.0f}", va="center", fontsize=7)

    covs = [s["srgb_coverage_de5"] for s in entries]
    axes[1].barh(y, covs, color="seagreen")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, fontsize=7)
    axes[1].set_xlabel("sRGB Coverage ΔE<5 (%)")
    axes[1].set_title("sRGB Coverage")
    for i, v in enumerate(covs):
        axes[1].text(v, i, f" {v:.1f}%", va="center", fontsize=7)

    nns = [s["nn_de_mean"] for s in entries]
    axes[2].barh(y, nns, color="coral")
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(labels, fontsize=7)
    axes[2].set_xlabel("Mean NN ΔE76")
    axes[2].set_title("Avg Nearest-Neighbor Distance")
    for i, v in enumerate(nns):
        axes[2].text(v, i, f" {v:.2f}", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out / "gamut_comparison.png", dpi=150)
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze color gamut coverage and ΔE distribution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--db", action="append", type=Path, default=[],
        metavar="PATH", help="ColorDB JSON path (repeatable)",
    )
    p.add_argument(
        "--model-pack", type=Path, default=None,
        metavar="PATH", help="ModelPack JSON path",
    )
    p.add_argument(
        "--mp-mode", action="append", default=None,
        metavar="MODE", help="ModelPack mode to include (repeatable; default: all)",
    )
    p.add_argument(
        "--merge", action="store_true",
        help="Merge all sources and run combined analysis",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("modeling/output/reports/gamut_analysis"),
        metavar="DIR", help="Output directory for plots",
    )
    p.add_argument(
        "--srgb-steps", type=int, default=_SRGB_STEPS_DEFAULT,
        help="sRGB sampling grid steps per axis (default: 21 → 9261 pts)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.db and args.model_pack is None:
        print("错误: 至少指定 --db 或 --model-pack", file=sys.stderr)
        return 1

    datasets: List[Tuple[str, np.ndarray]] = []
    for p in args.db:
        datasets.append(load_colordb(p))
    if args.model_pack is not None:
        datasets.extend(load_model_pack(args.model_pack, args.mp_mode))

    if not datasets:
        print("错误: 未加载到任何数据", file=sys.stderr)
        return 1

    srgb_lab = _srgb_reference_lab(args.srgb_steps)
    print(f"sRGB 参考采样: {srgb_lab.shape[0]} 点 (grid {args.srgb_steps}³)")

    all_stats: List[Dict] = []
    for name, lab in datasets:
        s = analyze(name, lab, srgb_lab)
        all_stats.append(s)
        print_stats(s)

    merged_stats: Optional[Dict] = None
    if args.merge and len(datasets) > 1:
        m_name, m_lab = merge_datasets(datasets)
        merged_stats = analyze(m_name, m_lab, srgb_lab)
        print_stats(merged_stats)
        print_incremental(datasets, srgb_lab)

    plot_all(all_stats, args.output_dir, merged_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
