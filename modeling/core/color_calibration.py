#!/usr/bin/env python3
"""
Reusable color calibration utilities based on ColorChecker.

Example:
  python -m modeling.core.color_calibration \
    --image /path/to/photo.jpg \
    --label /path/to/photo.json \
    --output-image /path/to/photo_calibrated.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from modeling.core.colorchecker_tool import ColorCheckerTool
from modeling.core.color_space import (
    linear_rgb_to_lab_d65,
    linear_to_srgb,
    srgb_to_linear,
)
from modeling.core.math_utils import round_list as _round_list

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REF_JSON = (
    REPO_ROOT / "modeling" / "data" / "ColorChecker" / "ColorCheckerjson.json"
)


def color_metrics_from_linear_rgb(linear_rgb: np.ndarray) -> Dict[str, object]:
    linear = np.clip(np.asarray(linear_rgb, dtype=np.float32), 0.0, 1.0)
    srgb = linear_to_srgb(linear)
    srgb_8bit = [int(round(float(v) * 255.0)) for v in srgb.tolist()]
    lab_d65 = linear_rgb_to_lab_d65(linear)
    return {
        "measured_srgb": srgb_8bit,
        "measured_lab_d65": lab_d65,
        "measured_linear_rgb": _round_list(linear, 6),
    }


def compute_ccm_from_colorchecker(
    patch_colors: Dict[str, Dict[str, object]],
    reference_colors: Dict[str, Dict[str, List[float]]],
) -> Tuple[np.ndarray, Dict[str, object]]:
    pairs: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
    for label, measured in patch_colors.items():
        ref = reference_colors.get(label)
        if not ref:
            continue
        ref_srgb = ref.get("sRGB")
        if not ref_srgb:
            continue
        meas_srgb = measured.get("sRGB")
        if not meas_srgb:
            continue
        meas_linear = srgb_to_linear(np.array(meas_srgb, dtype=np.float32) / 255.0)
        ref_linear = srgb_to_linear(np.array(ref_srgb, dtype=np.float32) / 255.0)
        ref_lab = ref.get("Lab_d65")
        ref_lab_arr = (
            np.array(ref_lab, dtype=np.float32) if ref_lab is not None else None
        )
        pairs.append((meas_linear, ref_linear, ref_lab_arr))

    if len(pairs) < 3:
        raise ValueError("Not enough ColorChecker patches for calibration.")

    measured_arr = np.stack([p[0] for p in pairs], axis=0)
    reference_arr = np.stack([p[1] for p in pairs], axis=0)
    ccm = np.linalg.lstsq(measured_arr, reference_arr, rcond=None)[0]

    errors: List[float] = []
    for meas_linear, _, ref_lab in pairs:
        if ref_lab is None:
            continue
        corrected = np.clip(meas_linear @ ccm, 0.0, 1.0)
        lab = np.array(linear_rgb_to_lab_d65(corrected), dtype=np.float32)
        errors.append(float(np.linalg.norm(lab - ref_lab)))

    stats: Dict[str, object] = {
        "patch_count": int(len(pairs)),
    }
    if errors:
        stats.update(
            {
                "mean_delta_e": round(float(np.mean(errors)), 4),
                "max_delta_e": round(float(np.max(errors)), 4),
            }
        )
    return ccm, stats


def _normalize_label(label: str) -> str:
    from modeling.core.io_utils import normalize_label
    return normalize_label(label)


def _points_to_quad(points: Sequence[Sequence[float]]) -> Optional[np.ndarray]:
    if len(points) == 2:
        (x0, y0), (x1, y1) = points
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    if len(points) >= 4:
        return np.array(points[:4], dtype=np.float32)
    return None


def _find_colorchecker_quad(
    shapes: Iterable[Dict[str, object]],
    label_name: str,
) -> Optional[np.ndarray]:
    target = _normalize_label(label_name)
    for shape in shapes:
        shape_label = _normalize_label(str(shape.get("label", "")))
        if shape_label != target:
            continue
        points = shape.get("points") or []
        quad = _points_to_quad(points)
        if quad is not None:
            return quad
    return None


def save_colorchecker_debug_image(
    image: np.ndarray,
    image_path: Path,
    cc_tool: ColorCheckerTool,
    target_corners: np.ndarray,
    patch_colors: Dict[str, Dict[str, object]],
    output_dir: Path,
    shrink_ratio: float,
    method: str,
) -> Path:
    h = cc_tool._select_homography(image, target_corners.tolist(), shrink_ratio, method)
    patches = cc_tool._apply_homography(h)

    overlay = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for patch in patches:
        pts = patch["points"].astype(np.int32)
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 1, cv2.LINE_AA)
        center = pts.mean(axis=0)
        label = str(patch["label"])
        cv2.putText(
            overlay,
            label,
            (int(center[0]), int(center[1])),
            font,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    row_height = 22
    panel_width = 360
    panel_height = max(overlay.shape[0], row_height * len(patches) + 30)
    panel = np.full((panel_height, panel_width, 3), 255, dtype=np.uint8)

    cv2.putText(
        panel,
        "ColorChecker (meas / ref)",
        (10, 18),
        font,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    def to_bgr(color: Optional[Sequence[float]]) -> Tuple[int, int, int]:
        if not color or len(color) < 3:
            return (200, 200, 200)
        r, g, b = color[:3]
        return (int(round(b)), int(round(g)), int(round(r)))

    y = 40
    for patch in patches:
        label = str(patch["label"])
        measured = patch_colors.get(label, {}).get("sRGB")
        reference = cc_tool.reference_colors.get(label, {}).get("sRGB")

        cv2.putText(panel, label, (10, y), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        meas_color = to_bgr(measured)
        ref_color = to_bgr(reference)
        cv2.rectangle(panel, (200, y - 12), (220, y + 6), meas_color, -1)
        cv2.rectangle(panel, (228, y - 12), (248, y + 6), ref_color, -1)
        cv2.rectangle(panel, (200, y - 12), (220, y + 6), (0, 0, 0), 1)
        cv2.rectangle(panel, (228, y - 12), (248, y + 6), (0, 0, 0), 1)

        if measured:
            text = f"{measured[0]},{measured[1]},{measured[2]}"
            cv2.putText(panel, text, (256, y), font, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

        y += row_height

    out_height = max(overlay.shape[0], panel.shape[0])
    out_width = overlay.shape[1] + panel.shape[1]
    canvas = np.full((out_height, out_width, 3), 255, dtype=np.uint8)
    canvas[: overlay.shape[0], : overlay.shape[1]] = overlay
    canvas[: panel.shape[0], overlay.shape[1] :] = panel

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{image_path.stem}_colorchecker_debug.png"
    cv2.imwrite(str(out_path), canvas)
    return out_path


def _sanitize_filename(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    return cleaned.strip("_") or "patch"


def save_colorchecker_patch_debugs(
    image: np.ndarray,
    image_path: Path,
    cc_tool: ColorCheckerTool,
    target_corners: np.ndarray,
    patch_colors: Dict[str, Dict[str, object]],
    ccm: np.ndarray,
    output_dir: Path,
    shrink_ratio: float,
    method: str,
) -> None:
    h = cc_tool._select_homography(image, target_corners.tolist(), shrink_ratio, method)
    patches = cc_tool._apply_homography(h)

    patches_dir = output_dir / "colorchecker_patches" / image_path.stem
    patches_dir.mkdir(parents=True, exist_ok=True)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for patch in patches:
        label = str(patch["label"])
        measured = patch_colors.get(label, {}).get("sRGB")
        reference = cc_tool.reference_colors.get(label, {}).get("sRGB")
        ref_lab = cc_tool.reference_colors.get(label, {}).get("Lab_d65")
        if not measured or not reference:
            continue

        meas_linear = srgb_to_linear(np.array(measured, dtype=np.float32) / 255.0)
        corrected_linear = np.clip(meas_linear @ ccm, 0.0, 1.0)
        corrected_srgb = linear_to_srgb(corrected_linear)
        corrected_8bit = [int(round(float(v) * 255.0)) for v in corrected_srgb.tolist()]

        delta_e_text = "DeltaE: n/a"
        if ref_lab:
            corrected_lab = linear_rgb_to_lab_d65(corrected_linear)
            diff = np.asarray(corrected_lab, dtype=np.float32) - np.asarray(
                ref_lab, dtype=np.float32
            )
            delta_e = float(np.linalg.norm(diff))
            delta_e_text = f"DeltaE: {delta_e:.2f}"

        canvas = np.full((120, 360, 3), 255, dtype=np.uint8)
        swatch_w = 80
        swatch_h = 50
        y0 = 40
        labels = [("raw", measured), ("corrected", corrected_8bit), ("ref", reference)]
        for i, (name, color) in enumerate(labels):
            x0 = 20 + i * 110
            bgr = (int(color[2]), int(color[1]), int(color[0]))
            cv2.rectangle(canvas, (x0, y0), (x0 + swatch_w, y0 + swatch_h), bgr, -1)
            cv2.rectangle(
                canvas, (x0, y0), (x0 + swatch_w, y0 + swatch_h), (0, 0, 0), 1
            )
            cv2.putText(
                canvas,
                name,
                (x0, y0 - 8),
                font,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            label,
            (10, 18),
            font,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            delta_e_text,
            (10, 110),
            font,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        out_name = f"{_sanitize_filename(label)}.png"
        cv2.imwrite(str(patches_dir / out_name), canvas)


def calibrate_image_with_colorchecker(
    image: np.ndarray,
    cc_quad: np.ndarray,
    cc_tool: ColorCheckerTool,
    shrink_ratio: float = 0.85,
    method: str = "mean",
    debug_dir: Optional[Path] = None,
    image_path: Optional[Path] = None,
) -> Tuple[np.ndarray, Dict[str, object], Dict[str, Dict[str, object]]]:
    patch_colors = cc_tool.extract_patch_colors(
        image, cc_quad.tolist(), shrink_ratio=shrink_ratio, method=method
    )
    ccm, stats = compute_ccm_from_colorchecker(patch_colors, cc_tool.reference_colors)

    if debug_dir is not None and image_path is not None:
        save_colorchecker_debug_image(
            image,
            image_path,
            cc_tool,
            cc_quad,
            patch_colors,
            debug_dir,
            shrink_ratio,
            method,
        )
        save_colorchecker_patch_debugs(
            image,
            image_path,
            cc_tool,
            cc_quad,
            patch_colors,
            ccm,
            debug_dir,
            shrink_ratio,
            method,
        )

    return ccm, stats, patch_colors


def apply_ccm_to_image(image_bgr: np.ndarray, ccm: np.ndarray) -> np.ndarray:
    rgb = np.clip(image_bgr[..., ::-1], 0, 255).astype(np.float32) / 255.0
    linear_rgb = srgb_to_linear(rgb)
    corrected_linear = np.clip(linear_rgb @ ccm, 0.0, 1.0)
    corrected_srgb = linear_to_srgb(corrected_linear)
    corrected = np.clip(corrected_srgb * 255.0, 0, 255).astype(np.uint8)
    return corrected[..., ::-1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate an image using ColorChecker and label JSON."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--label",
        type=Path,
        default=None,
        help="Path to label JSON (default: <image>.json in same folder).",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=None,
        help="Output calibrated image path (default: <image>_calibrated.png).",
    )
    parser.add_argument(
        "--label-name",
        type=str,
        default="ColorChecker",
        help="Label name for ColorChecker polygon in JSON.",
    )
    parser.add_argument(
        "--ref-json",
        type=Path,
        default=DEFAULT_REF_JSON,
        help="Reference ColorChecker LabelMe JSON.",
    )
    parser.add_argument(
        "--ref-desc",
        type=Path,
        default=None,
        help="Optional ColorChecker desc.txt path.",
    )
    parser.add_argument(
        "--shrink",
        type=float,
        default=0.85,
        help="Shrink ratio for patch sampling polygon.",
    )
    parser.add_argument(
        "--method",
        choices=("mean", "median"),
        default="mean",
        help="Sampling method within each patch.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Optional directory to save ColorChecker debug images.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.ref_json.exists():
        raise FileNotFoundError(f"Reference JSON not found: {args.ref_json}")

    label_path = args.label
    if label_path is None:
        candidate = args.image.with_suffix(".json")
        if candidate.exists():
            label_path = candidate
        else:
            candidate_upper = args.image.with_suffix(".JSON")
            if candidate_upper.exists():
                label_path = candidate_upper
            else:
                raise FileNotFoundError(
                    f"Label JSON not found. Tried {candidate} and {candidate_upper}"
                )
    if not label_path.exists():
        raise FileNotFoundError(f"Label JSON not found: {label_path}")

    label_data = json.loads(label_path.read_text(encoding="utf-8"))
    shapes = label_data.get("shapes", [])
    cc_quad = _find_colorchecker_quad(shapes, args.label_name)
    if cc_quad is None:
        raise ValueError(
            f"ColorChecker label '{args.label_name}' not found in {label_path}"
        )

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {args.image}")

    cc_tool = ColorCheckerTool(args.ref_json, reference_desc_path=args.ref_desc)
    ccm, stats, _ = calibrate_image_with_colorchecker(
        image=image,
        cc_quad=cc_quad,
        cc_tool=cc_tool,
        shrink_ratio=args.shrink,
        method=args.method,
        debug_dir=args.debug_dir,
        image_path=args.image,
    )

    output_path = args.output_image
    if output_path is None:
        output_path = args.image.with_suffix(".calibrated.png")

    corrected_bgr = apply_ccm_to_image(image, ccm)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), corrected_bgr)
    print(f"Saved calibrated image to {output_path}")

    mean_delta = stats.get("mean_delta_e")
    max_delta = stats.get("max_delta_e")
    if mean_delta is not None or max_delta is not None:
        print(f"Calibration DeltaE mean={mean_delta} max={max_delta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
