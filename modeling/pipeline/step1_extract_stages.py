#!/usr/bin/env python3
"""
Extract single-color stage measurements from labeled images.

Example:
  python -m modeling.pipeline.step1_extract_stages \
    --input-root modeling/data/1_single \
    --output modeling/output/params/1_single_stage.json
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from modeling.core.colorchecker_tool import ColorCheckerTool
from modeling.core.color_calibration import (
    calibrate_image_with_colorchecker,
    color_metrics_from_linear_rgb,
)
from modeling.core.color_space import linear_rgb_to_lab_d65, srgb_to_linear
from modeling.core.io_utils import normalize_label


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "modeling" / "data" / "1_single"
DEFAULT_OUTPUT = REPO_ROOT / "modeling" / "output" / "params" / "1_single_stage.json"


@dataclass(frozen=True)
class StageConfig:
    grid_cols: int = 6
    grid_rows: int = 3
    block_mm: float = 10.0
    gap_mm: float = 1.0
    margin_mm: float = 1.0
    layer_height_mm: float = 0.04
    step_layers: Tuple[int, ...] = tuple(list(range(17)) + [25])

    @property
    def width_mm(self) -> float:
        return (
            self.grid_cols * self.block_mm
            + (self.grid_cols - 1) * self.gap_mm
            + 2 * self.margin_mm
        )

    @property
    def height_mm(self) -> float:
        return (
            self.grid_rows * self.block_mm
            + (self.grid_rows - 1) * self.gap_mm
            + 2 * self.margin_mm
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract single-color stage data from labeled images."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT,
        help="Root folder containing base/color subfolders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--ref-json",
        type=Path,
        default=REPO_ROOT
        / "modeling"
        / "data"
        / "ColorChecker"
        / "ColorCheckerjson.json",
        help="Reference ColorChecker LabelMe JSON.",
    )
    parser.add_argument(
        "--ref-desc",
        type=Path,
        default=None,
        help="Optional ColorChecker desc.txt path.",
    )
    parser.add_argument(
        "--method",
        choices=("mean", "median"),
        default="mean",
        help="Sampling method within each patch.",
    )
    parser.add_argument(
        "--shrink",
        type=float,
        default=0.85,
        help="Shrink ratio for patch sampling polygon.",
    )
    parser.add_argument(
        "--agg-method",
        choices=("mean", "median"),
        default="mean",
        help="Aggregation method across multiple images.",
    )
    parser.add_argument(
        "--agg-space",
        choices=("lab", "linear_rgb"),
        default="lab",
        help="Primary space for aggregated value selection.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".jpg,.jpeg,.png,.tif,.tiff",
        help="Comma separated image extensions to match.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Optional directory to save ColorChecker debug images.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker threads budget for dataset/image processing (0=auto).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def points_to_quad(points: Sequence[Sequence[float]]) -> Optional[np.ndarray]:
    if len(points) == 2:
        (x0, y0), (x1, y1) = points
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    if len(points) >= 4:
        return np.array(points[:4], dtype=np.float32)
    return None


def quad_signed_area(pts: np.ndarray) -> float:
    if pts.shape != (4, 2):
        pts = pts.reshape(4, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def order_quad(pts: np.ndarray) -> np.ndarray:
    if pts.shape != (4, 2):
        pts = pts.reshape(4, 2)
    pts = np.asarray(pts, dtype=np.float32)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    if quad_signed_area(ordered) < 0:
        ordered = ordered[::-1]
    return ordered


def order_stage_quad(pts: np.ndarray, cfg: StageConfig) -> np.ndarray:
    ordered = order_quad(pts)
    expected_ratio = cfg.width_mm / cfg.height_mm if cfg.height_mm else 0.0
    if expected_ratio <= 0.0:
        return ordered
    best_score: Optional[float] = None
    best_quad: Optional[np.ndarray] = None
    for shift in range(4):
        candidate = np.roll(ordered, -shift, axis=0)
        edge_len = np.linalg.norm(candidate - np.roll(candidate, -1, axis=0), axis=1)
        width = 0.5 * (edge_len[0] + edge_len[2])
        height = 0.5 * (edge_len[1] + edge_len[3])
        if width <= 1e-6 or height <= 1e-6:
            continue
        ratio = width / height
        score = abs(float(np.log(ratio / expected_ratio)))
        if best_score is None or score < best_score:
            best_score = score
            best_quad = candidate
    return best_quad if best_quad is not None else ordered


# ---------------------------------------------------------------------------
# Label / quad finders
# ---------------------------------------------------------------------------

def find_quad_by_label(
    shapes: Iterable[Dict[str, object]], label: str
) -> Optional[np.ndarray]:
    target = normalize_label(label)
    for shape in shapes:
        shape_label = normalize_label(str(shape.get("label", "")))
        if shape_label != target:
            continue
        points = shape.get("points") or []
        quad = points_to_quad(points)
        if quad is not None:
            return quad
    return None


def find_stage_quad(
    shapes: Iterable[Dict[str, object]], color_name: str
) -> Tuple[Optional[np.ndarray], Optional[str], bool]:
    color_label = normalize_label(color_name)
    for shape in shapes:
        label = str(shape.get("label", ""))
        if normalize_label(label) == color_label:
            quad = points_to_quad(shape.get("points") or [])
            if quad is not None:
                return quad, label, False
    keywords = ("stage", "step", "ladder", "stair", "colorstage", "colorstep")
    keyword_set = {normalize_label(k) for k in keywords}
    for shape in shapes:
        label = str(shape.get("label", ""))
        if normalize_label(label) in keyword_set:
            quad = points_to_quad(shape.get("points") or [])
            if quad is not None:
                return quad, label, False
    candidates = []
    for shape in shapes:
        label = str(shape.get("label", ""))
        if normalize_label(label) == "colorchecker":
            continue
        quad = points_to_quad(shape.get("points") or [])
        if quad is not None:
            candidates.append((quad, label))
    if len(candidates) == 1:
        return candidates[0][0], candidates[0][1], False
    if candidates:
        return candidates[0][0], candidates[0][1], True
    return None, None, False


# ---------------------------------------------------------------------------
# Grid & orientation
# ---------------------------------------------------------------------------

def build_stage_grid_positions(cfg: StageConfig) -> List[Dict[str, object]]:
    if len(cfg.step_layers) != cfg.grid_cols * cfg.grid_rows:
        raise ValueError("step_layers size does not match grid size.")
    positions: List[Dict[str, object]] = []
    idx = 0
    for row in range(cfg.grid_rows):
        for col in range(cfg.grid_cols):
            x0 = cfg.margin_mm + col * (cfg.block_mm + cfg.gap_mm)
            y0 = cfg.margin_mm + row * (cfg.block_mm + cfg.gap_mm)
            x1 = x0 + cfg.block_mm
            y1 = y0 + cfg.block_mm
            positions.append(
                {
                    "pos_index": idx,
                    "row": row,
                    "col": col,
                    "quad_stage": [
                        [x0, y0],
                        [x1, y0],
                        [x1, y1],
                        [x0, y1],
                    ],
                }
            )
            idx += 1
    return positions


def orientation_mapping(cfg: StageConfig, flip_x: bool, flip_y: bool) -> List[int]:
    mapping: List[int] = []
    for step_idx in range(len(cfg.step_layers)):
        row_s = step_idx // cfg.grid_cols
        col_s = step_idx % cfg.grid_cols
        row_p = row_s if flip_y else cfg.grid_rows - 1 - row_s
        col_p = cfg.grid_cols - 1 - col_s if flip_x else col_s
        mapping.append(row_p * cfg.grid_cols + col_p)
    return mapping


def expected_base_linear_rgb(substrate_id: str) -> Optional[np.ndarray]:
    key = normalize_label(substrate_id)
    if "white" in key:
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)
    if "black" in key:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return None


def select_stage_orientation(
    position_patches: Sequence[Dict[str, object]],
    cfg: StageConfig,
    substrate_id: str,
    warnings: List[str],
) -> List[int]:
    expected_linear = expected_base_linear_rgb(substrate_id)
    if expected_linear is None:
        warnings.append(
            f"Unknown substrate '{substrate_id}', using default orientation."
        )
        return orientation_mapping(cfg, False, False)

    expected_lab = np.array(linear_rgb_to_lab_d65(expected_linear), dtype=np.float32)
    idx_bottom_left = (cfg.grid_rows - 1) * cfg.grid_cols
    idx_top_right = cfg.grid_cols - 1

    def patch_lab(index: int) -> Optional[np.ndarray]:
        if index < 0 or index >= len(position_patches):
            return None
        lab = position_patches[index].get("measured_lab_d65")
        if not lab:
            return None
        return np.asarray(lab, dtype=np.float32)

    lab_bl = patch_lab(idx_bottom_left)
    lab_tr = patch_lab(idx_top_right)

    if lab_bl is None and lab_tr is None:
        warnings.append(f"Stage corner colors missing for '{substrate_id}'.")
        return orientation_mapping(cfg, False, False)
    if lab_bl is None:
        warnings.append(f"Bottom-left stage patch missing for '{substrate_id}'.")
        return orientation_mapping(cfg, True, True)
    if lab_tr is None:
        warnings.append(f"Top-right stage patch missing for '{substrate_id}'.")
        return orientation_mapping(cfg, False, False)

    dist_bl = float(np.linalg.norm(lab_bl - expected_lab))
    dist_tr = float(np.linalg.norm(lab_tr - expected_lab))
    if dist_tr < dist_bl:
        return orientation_mapping(cfg, True, True)
    return orientation_mapping(cfg, False, False)


def build_stage_patches_from_positions(
    position_patches: Sequence[Dict[str, object]],
    cfg: StageConfig,
    mapping: Sequence[int],
) -> List[Dict[str, object]]:
    stage_patches: List[Dict[str, object]] = []
    for step_idx, step_layers in enumerate(cfg.step_layers):
        pos_idx = mapping[step_idx]
        pos_patch = position_patches[pos_idx]
        entry = {
            "index": step_idx,
            "step_layers": int(step_layers),
            "thickness_mm": round(step_layers * cfg.layer_height_mm, 6),
            "valid": bool(pos_patch.get("valid")),
        }
        if pos_patch.get("valid"):
            entry["measured_srgb"] = pos_patch.get("measured_srgb")
            entry["measured_lab_d65"] = pos_patch.get("measured_lab_d65")
            entry["measured_linear_rgb"] = pos_patch.get("measured_linear_rgb")
        stage_patches.append(entry)
    return stage_patches


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def shrink_polygon(pts: np.ndarray, ratio: float) -> np.ndarray:
    if ratio >= 1.0:
        return pts
    center = pts.mean(axis=0, keepdims=True)
    return center + (pts - center) * ratio


def sample_polygon_color(
    image: np.ndarray,
    polygon: np.ndarray,
    shrink_ratio: float,
    method: str,
) -> Optional[np.ndarray]:
    if polygon.size == 0:
        return None
    poly = shrink_polygon(polygon, shrink_ratio)
    poly_int = np.round(poly).astype(np.int32)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly_int], 255)
    ys, xs = np.where(mask == 255)
    if xs.size == 0:
        return None
    pixels = image[ys, xs]
    if method == "median":
        return np.median(pixels, axis=0)
    return pixels.mean(axis=0)


def save_stage_debug_image(
    image: np.ndarray,
    image_path: Path,
    stage_corners: np.ndarray,
    cfg: StageConfig,
    positions: Sequence[Dict[str, object]],
    mapping: Sequence[int],
    output_dir: Path,
) -> Path:
    src_quad = np.array(
        [[0, 0], [cfg.width_mm, 0], [cfg.width_mm, cfg.height_mm], [0, cfg.height_mm]],
        dtype=np.float32,
    )
    dst_quad = order_stage_quad(stage_corners, cfg)
    h = cv2.getPerspectiveTransform(src_quad, dst_quad)

    overlay = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_map: Dict[int, str] = {}
    for step_idx, step_layers in enumerate(cfg.step_layers):
        pos_idx = mapping[step_idx]
        label_map[pos_idx] = f"{step_idx}:{int(step_layers)}"

    for cell in positions:
        pts_stage = np.array(cell["quad_stage"], dtype=np.float32).reshape(1, -1, 2)
        pts_img = cv2.perspectiveTransform(pts_stage, h)[0]
        pts = pts_img.astype(np.int32)
        cv2.polylines(overlay, [pts], True, (0, 200, 255), 1, cv2.LINE_AA)
        center = pts_img.mean(axis=0)
        label = label_map.get(cell["pos_index"], "")
        cv2.putText(
            overlay, label,
            (int(center[0]), int(center[1])),
            font, 0.45, (0, 200, 255), 1, cv2.LINE_AA,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{image_path.stem}_stage_debug.png"
    cv2.imwrite(str(out_path), overlay)
    return out_path


def resolve_image_path(
    label_path: Path, label_data: Dict[str, object], extensions: Sequence[str]
) -> Optional[Path]:
    image_path = label_data.get("imagePath")
    if isinstance(image_path, str) and image_path:
        candidate = (label_path.parent / image_path).resolve()
        if candidate.exists():
            return candidate
    for ext in extensions:
        candidate = label_path.with_suffix(ext)
        if candidate.exists():
            return candidate.resolve()
    return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_vectors(
    vectors: List[List[float]],
    method: str,
    ndigits: int,
    as_int: bool = False,
) -> Optional[Dict[str, object]]:
    if not vectors:
        return None
    arr = np.array(vectors, dtype=np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    median = np.median(arr, axis=0)
    value = median if method == "median" else mean

    def finalize(values: np.ndarray) -> List[float]:
        rounded = [round(float(v), ndigits) for v in values.tolist()]
        if as_int:
            return [int(round(v)) for v in rounded]
        return rounded

    return {
        "count": int(arr.shape[0]),
        "value": finalize(value),
        "mean": finalize(mean),
        "std": finalize(std),
        "median": finalize(median),
    }


def sample_stage_positions(
    image: np.ndarray,
    stage_corners: np.ndarray,
    cfg: StageConfig,
    positions: Sequence[Dict[str, object]],
    shrink_ratio: float,
    method: str,
    ccm: np.ndarray,
) -> List[Dict[str, object]]:
    src_quad = np.array(
        [[0, 0], [cfg.width_mm, 0], [cfg.width_mm, cfg.height_mm], [0, cfg.height_mm]],
        dtype=np.float32,
    )
    dst_quad = order_stage_quad(stage_corners, cfg)
    h = cv2.getPerspectiveTransform(src_quad, dst_quad)

    patches: List[Dict[str, object]] = []
    for cell in positions:
        pts_stage = np.array(cell["quad_stage"], dtype=np.float32).reshape(1, -1, 2)
        pts_img = cv2.perspectiveTransform(pts_stage, h)[0]
        color_bgr = sample_polygon_color(image, pts_img, shrink_ratio, method)
        entry = {
            "pos_index": cell["pos_index"],
            "valid": color_bgr is not None,
        }
        if color_bgr is not None:
            rgb = np.clip(color_bgr[::-1], 0, 255) / 255.0
            linear_rgb = srgb_to_linear(rgb)
            corrected = np.clip(linear_rgb @ ccm, 0.0, 1.0)
            entry.update(color_metrics_from_linear_rgb(corrected))
        patches.append(entry)

    return patches


def collect_label_files(color_dir: Path) -> List[Path]:
    return sorted(p for p in color_dir.glob("*.json") if p.is_file())


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image(
    label_path: Path,
    image_path: Path,
    color_name: str,
    label_data: Dict[str, object],
    path_root: Path,
    cc_tool: ColorCheckerTool,
    cfg: StageConfig,
    positions: Sequence[Dict[str, object]],
    substrate_id: str,
    shrink_ratio: float,
    method: str,
    warnings: List[str],
    debug_dir: Optional[Path],
) -> Optional[Dict[str, object]]:
    shapes = label_data.get("shapes", [])
    cc_quad = find_quad_by_label(shapes, "ColorChecker")
    if cc_quad is None:
        warnings.append(f"Missing ColorChecker in {label_path}")
        return None

    stage_quad, stage_label, ambiguous = find_stage_quad(shapes, color_name)
    if stage_quad is None:
        warnings.append(f"Missing stage quad in {label_path}")
        return None
    if ambiguous:
        warnings.append(f"Ambiguous stage label in {label_path}, using {stage_label}")
    if stage_label and normalize_label(stage_label) != normalize_label(color_name):
        warnings.append(
            f"Stage label '{stage_label}' does not match folder '{color_name}' "
            f"in {label_path}"
        )

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        warnings.append(f"Failed to read image: {image_path}")
        return None

    try:
        ccm, calib_stats, _ = calibrate_image_with_colorchecker(
            image=image,
            cc_quad=cc_quad,
            cc_tool=cc_tool,
            shrink_ratio=shrink_ratio,
            method=method,
            debug_dir=debug_dir,
            image_path=image_path,
        )
    except ValueError as exc:
        warnings.append(f"Calibration failed for {label_path}: {exc}")
        return None

    position_patches = sample_stage_positions(
        image, stage_quad, cfg, positions, shrink_ratio, method, ccm
    )
    mapping = select_stage_orientation(position_patches, cfg, substrate_id, warnings)
    stage_patches = build_stage_patches_from_positions(position_patches, cfg, mapping)

    if debug_dir is not None:
        save_stage_debug_image(
            image, image_path, stage_quad, cfg, positions, mapping, debug_dir,
        )

    def relpath(path: Path) -> str:
        try:
            return str(path.relative_to(path_root))
        except ValueError:
            return str(path)

    return {
        "image_path": relpath(image_path),
        "calibration": calib_stats,
        "stage_patches": stage_patches,
    }


def aggregate_dataset(
    images: Sequence[Dict[str, object]],
    cfg: StageConfig,
    agg_method: str,
    agg_space: str,
) -> List[Dict[str, object]]:
    aggregated: List[Dict[str, object]] = []
    for idx, step_layers in enumerate(cfg.step_layers):
        srgb_values: List[List[float]] = []
        lab_d65_values: List[List[float]] = []
        linear_values: List[List[float]] = []

        for image in images:
            patches = image.get("stage_patches") or []
            patch = patches[idx] if idx < len(patches) else None
            if not patch or not patch.get("valid"):
                continue
            srgb = patch.get("measured_srgb")
            lab_d65 = patch.get("measured_lab_d65")
            linear_rgb = patch.get("measured_linear_rgb")
            if srgb:
                srgb_values.append(srgb)
            if lab_d65:
                lab_d65_values.append(lab_d65)
            if linear_rgb:
                linear_values.append(linear_rgb)

        srgb_stats = aggregate_vectors(srgb_values, agg_method, 0, as_int=True)
        lab_d65_stats = aggregate_vectors(lab_d65_values, agg_method, 2)
        linear_stats = aggregate_vectors(linear_values, agg_method, 6)

        entry = {
            "index": idx,
            "step_layers": int(step_layers),
            "thickness_mm": round(step_layers * cfg.layer_height_mm, 6),
            "count": int(srgb_stats["count"]) if srgb_stats else 0,
            "measured_srgb": srgb_stats["value"] if srgb_stats else None,
            "measured_lab_d65": lab_d65_stats["value"] if lab_d65_stats else None,
            "measured_linear_rgb": linear_stats["value"] if linear_stats else None,
            "std_lab_d65": lab_d65_stats["std"] if lab_d65_stats else None,
        }
        aggregated.append(entry)

    return aggregated


def summarize_calibration(
    images: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    mean_errors: List[float] = []
    max_errors: List[float] = []
    patch_counts: List[int] = []
    image_summaries: List[Dict[str, object]] = []

    for image in images:
        calib = image.get("calibration") or {}
        mean_delta = calib.get("mean_delta_e")
        max_delta = calib.get("max_delta_e")
        patch_count = calib.get("patch_count")
        if isinstance(mean_delta, (int, float)):
            mean_errors.append(float(mean_delta))
        if isinstance(max_delta, (int, float)):
            max_errors.append(float(max_delta))
        if isinstance(patch_count, int):
            patch_counts.append(patch_count)

        image_summaries.append(
            {
                "image_path": image.get("image_path"),
                "mean_delta_e": mean_delta,
                "max_delta_e": max_delta,
                "patch_count": patch_count,
            }
        )

    summary = {
        "method": "ccm_3x3",
        "image_count": int(len(images)),
        "mean_delta_e": round(float(np.mean(mean_errors)), 4) if mean_errors else None,
        "max_delta_e": round(float(np.max(max_errors)), 4) if max_errors else None,
        "mean_patch_count": (
            round(float(np.mean(patch_counts)), 2) if patch_counts else None
        ),
        "image_summaries": image_summaries,
    }
    return summary


def process_dataset(
    base_dir: Path,
    color_dir: Path,
    cc_tool: ColorCheckerTool,
    cfg: StageConfig,
    positions: Sequence[Dict[str, object]],
    path_root: Path,
    extensions: Sequence[str],
    shrink_ratio: float,
    method: str,
    agg_method: str,
    agg_space: str,
    image_workers: int,
    warnings: List[str],
    debug_root: Optional[Path],
) -> Optional[Dict[str, object]]:
    label_files = collect_label_files(color_dir)
    images: List[Dict[str, object]] = []
    tasks: List[Tuple[Path, Path, Dict[str, object], Optional[Path]]] = []

    for label_path in label_files:
        label_data = json.loads(label_path.read_text(encoding="utf-8"))
        image_path = resolve_image_path(label_path, label_data, extensions)
        if image_path is None:
            warnings.append(f"Missing image for label: {label_path}")
            continue
        debug_dir = None
        if debug_root is not None:
            debug_dir = debug_root / base_dir.name / color_dir.name
        tasks.append((label_path, image_path, label_data, debug_dir))

    if not tasks:
        return None

    if image_workers <= 1 or len(tasks) == 1:
        for label_path, image_path, label_data, debug_dir in tasks:
            entry = process_image(
                label_path, image_path, color_dir.name, label_data, path_root,
                cc_tool, cfg, positions, base_dir.name, shrink_ratio, method,
                warnings, debug_dir,
            )
            if entry is not None:
                images.append(entry)
    else:
        max_workers = min(image_workers, len(tasks))

        def run_task(
            item: Tuple[Path, Path, Dict[str, object], Optional[Path]]
        ) -> Tuple[Optional[Dict[str, object]], List[str]]:
            label_path, image_path, label_data, debug_dir = item
            local_warnings: List[str] = []
            entry = process_image(
                label_path, image_path, color_dir.name, label_data, path_root,
                cc_tool, cfg, positions, base_dir.name, shrink_ratio, method,
                local_warnings, debug_dir,
            )
            return entry, local_warnings

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for entry, local_warnings in executor.map(run_task, tasks):
                if local_warnings:
                    warnings.extend(local_warnings)
                if entry is not None:
                    images.append(entry)

    if not images:
        return None

    aggregated = aggregate_dataset(images, cfg, agg_method, agg_space)
    calibration_summary = summarize_calibration(images)
    return {
        "substrate_id": base_dir.name,
        "color_name": color_dir.name,
        "image_count": len(images),
        "calibration": calibration_summary,
        "aggregated": aggregated,
    }


def build_output(
    args: argparse.Namespace,
    cfg: StageConfig,
    datasets: Sequence[Dict[str, object]],
    warnings: List[str],
) -> Dict[str, object]:
    return {
        "meta": {
            "schema_version": "2.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_root": str(args.input_root),
            "calibration_method": "ccm_3x3",
            "reference_json": str(args.ref_json),
            "sampling_method": args.method,
            "sampling_shrink_ratio": args.shrink,
            "aggregation_method": args.agg_method,
            "aggregation_space": args.agg_space,
            "warnings": warnings,
        },
        "stage_config": {
            "grid_cols": cfg.grid_cols,
            "grid_rows": cfg.grid_rows,
            "block_mm": cfg.block_mm,
            "gap_mm": cfg.gap_mm,
            "margin_mm": cfg.margin_mm,
            "layer_height_mm": cfg.layer_height_mm,
            "step_layers": list(cfg.step_layers),
            "stage_width_mm": cfg.width_mm,
            "stage_height_mm": cfg.height_mm,
            "grid_order": "row_major_bottom_left_auto",
        },
        "datasets": list(datasets),
    }


def main() -> int:
    args = parse_args()
    input_root = args.input_root
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if not args.ref_json.exists():
        raise FileNotFoundError(f"Reference JSON not found: {args.ref_json}")

    ref_desc = args.ref_desc
    if ref_desc is None:
        candidate = args.ref_json.parent / "desc.txt"
        if candidate.exists():
            ref_desc = candidate

    cc_tool = ColorCheckerTool(
        args.ref_json, reference_desc_path=ref_desc if ref_desc else None
    )
    if not cc_tool.reference_colors:
        raise ValueError("Reference ColorChecker colors not found for calibration.")
    cfg = StageConfig()
    positions = build_stage_grid_positions(cfg)
    extensions = [
        ext.strip().lower() for ext in args.extensions.split(",") if ext.strip()
    ]
    cpu_count = os.cpu_count() or 1
    workers = args.workers if args.workers > 0 else min(32, cpu_count + 4)

    warnings: List[str] = []
    datasets: List[Dict[str, object]] = []

    dataset_dirs: List[Tuple[Path, Path]] = []
    for base_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        for color_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
            dataset_dirs.append((base_dir, color_dir))

    dataset_workers = min(len(dataset_dirs), max(1, workers)) if dataset_dirs else 1
    image_workers = max(1, workers // dataset_workers) if dataset_workers else workers

    if dataset_workers <= 1 or len(dataset_dirs) <= 1:
        for base_dir, color_dir in tqdm(dataset_dirs, desc="Datasets", unit="set"):
            dataset = process_dataset(
                base_dir, color_dir, cc_tool, cfg, positions, input_root,
                extensions, args.shrink, args.method, args.agg_method,
                args.agg_space, image_workers, warnings, args.debug_dir,
            )
            if dataset is not None:
                datasets.append(dataset)
    else:
        def run_dataset(
            item: Tuple[Path, Path]
        ) -> Tuple[Optional[Dict[str, object]], List[str]]:
            base_dir, color_dir = item
            local_warnings: List[str] = []
            dataset = process_dataset(
                base_dir, color_dir, cc_tool, cfg, positions, input_root,
                extensions, args.shrink, args.method, args.agg_method,
                args.agg_space, image_workers, local_warnings, args.debug_dir,
            )
            return dataset, local_warnings

        with ThreadPoolExecutor(max_workers=dataset_workers) as executor:
            for dataset, local_warnings in tqdm(
                executor.map(run_dataset, dataset_dirs),
                total=len(dataset_dirs),
                desc="Datasets",
                unit="set",
            ):
                if local_warnings:
                    warnings.extend(local_warnings)
                if dataset is not None:
                    datasets.append(dataset)

    output = build_output(args, cfg, datasets, warnings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
