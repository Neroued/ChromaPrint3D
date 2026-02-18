#!/usr/bin/env python3
"""
Evaluate preview quality against source images.

Metric:
  BADE = mean(DeltaE00) + w * sqrt((mean_da)^2 + (mean_db)^2)
where mean_da / mean_db are global chroma-axis biases in Lab.

Example:
  python -m modeling.eval.preview_quality \
    --data-dir data \
    --preview-dir modeling \
    --images xhs1 columbina
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class EvalResult:
    image: str
    mode: str
    reference: str
    preview: str
    mask: Optional[str]
    pixels: int
    bade: float
    mean_de00: float
    median_de00: float
    p90_de00: float
    max_de00: float
    mean_dL: float
    mean_da: float
    mean_db: float
    chroma_bias: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate preview quality against source images."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing source images.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=Path("modeling"),
        help="Directory containing preview/mask outputs.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Image stems to evaluate, e.g. xhs1 columbina.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["colordb_only", "model_only", "mixed"],
        help="Preview modes to evaluate.",
    )
    parser.add_argument(
        "--weight-chroma",
        type=float,
        default=0.7,
        help="Weight for chroma bias penalty in BADE.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for JSON report.",
    )
    return parser.parse_args()


def find_reference_image(data_dir: Path, stem: str) -> Path:
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"):
        candidate = data_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Reference image not found for stem '{stem}' in {data_dir}")


def load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError(f"Failed to load image: {path}")
    return img


def load_mask(mask_path: Path, shape_hw: tuple[int, int]) -> Optional[np.ndarray]:
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.size == 0:
        return None
    h, w = shape_hw
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    valid = mask > 0
    if int(np.count_nonzero(valid)) == 0:
        return None
    return valid


def bgr_to_lab_float(bgr: np.ndarray) -> np.ndarray:
    bgr_f = bgr.astype(np.float32) / 255.0
    return cv2.cvtColor(bgr_f, cv2.COLOR_BGR2Lab)


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    # Vectorized CIEDE2000 implementation.
    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

    C1 = np.sqrt(a1 * a1 + b1 * b1)
    C2 = np.sqrt(a2 * a2 + b2 * b2)
    C_bar = 0.5 * (C1 + C2)
    C_bar7 = np.power(C_bar, 7.0)
    G = 0.5 * (1.0 - np.sqrt(C_bar7 / (C_bar7 + np.power(25.0, 7.0) + 1e-12)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.sqrt(a1p * a1p + b1 * b1)
    C2p = np.sqrt(a2p * a2p + b2 * b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p

    hp_diff = h2p - h1p
    zero_chroma = (C1p * C2p) == 0
    hp_diff = np.where(zero_chroma, 0.0, hp_diff)
    hp_diff = np.where(hp_diff > 180.0, hp_diff - 360.0, hp_diff)
    hp_diff = np.where(hp_diff < -180.0, hp_diff + 360.0, hp_diff)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(hp_diff * 0.5))

    Lp_bar = 0.5 * (L1 + L2)
    Cp_bar = 0.5 * (C1p + C2p)

    hp_sum = h1p + h2p
    hp_abs_diff = np.abs(h1p - h2p)
    hp_bar = np.where(
        zero_chroma,
        hp_sum,
        np.where(
            hp_abs_diff <= 180.0,
            0.5 * hp_sum,
            np.where(hp_sum < 360.0, 0.5 * (hp_sum + 360.0), 0.5 * (hp_sum - 360.0)),
        ),
    )

    T = (
        1.0
        - 0.17 * np.cos(np.radians(hp_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hp_bar))
        + 0.32 * np.cos(np.radians(3.0 * hp_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hp_bar - 63.0))
    )

    delta_theta = 30.0 * np.exp(-np.square((hp_bar - 275.0) / 25.0))
    Rc = 2.0 * np.sqrt(
        np.power(Cp_bar, 7.0) / (np.power(Cp_bar, 7.0) + np.power(25.0, 7.0) + 1e-12)
    )
    Sl = 1.0 + (0.015 * np.square(Lp_bar - 50.0)) / np.sqrt(20.0 + np.square(Lp_bar - 50.0))
    Sc = 1.0 + 0.045 * Cp_bar
    Sh = 1.0 + 0.015 * Cp_bar * T
    Rt = -np.sin(np.radians(2.0 * delta_theta)) * Rc

    dE = np.sqrt(
        np.square(dLp / Sl)
        + np.square(dCp / Sc)
        + np.square(dHp / Sh)
        + Rt * (dCp / Sc) * (dHp / Sh)
    )
    return dE.astype(np.float32)


def evaluate_pair(
    image_stem: str,
    mode: str,
    ref_path: Path,
    preview_path: Path,
    mask_path: Path,
    weight_chroma: float,
) -> EvalResult:
    ref_bgr = load_bgr(ref_path)
    pred_bgr = load_bgr(preview_path)

    h, w = pred_bgr.shape[:2]
    ref_resized = cv2.resize(ref_bgr, (w, h), interpolation=cv2.INTER_AREA)

    ref_lab = bgr_to_lab_float(ref_resized)
    pred_lab = bgr_to_lab_float(pred_bgr)

    valid_mask = load_mask(mask_path, (h, w))
    if valid_mask is None:
        valid_mask = np.ones((h, w), dtype=bool)

    ref_flat = ref_lab[valid_mask].reshape(-1, 3)
    pred_flat = pred_lab[valid_mask].reshape(-1, 3)
    if ref_flat.shape[0] == 0:
        raise ValueError(f"No valid pixels after masking: {preview_path}")

    de00 = ciede2000(ref_flat, pred_flat)
    dlab = pred_flat - ref_flat
    mean_dL = float(np.mean(dlab[:, 0]))
    mean_da = float(np.mean(dlab[:, 1]))
    mean_db = float(np.mean(dlab[:, 2]))
    chroma_bias = float(np.sqrt(mean_da * mean_da + mean_db * mean_db))
    mean_de = float(np.mean(de00))
    bade = mean_de + float(weight_chroma) * chroma_bias

    return EvalResult(
        image=image_stem,
        mode=mode,
        reference=str(ref_path),
        preview=str(preview_path),
        mask=str(mask_path) if mask_path.exists() else None,
        pixels=int(ref_flat.shape[0]),
        bade=float(bade),
        mean_de00=mean_de,
        median_de00=float(np.median(de00)),
        p90_de00=float(np.percentile(de00, 90.0)),
        max_de00=float(np.max(de00)),
        mean_dL=mean_dL,
        mean_da=mean_da,
        mean_db=mean_db,
        chroma_bias=chroma_bias,
    )


def print_report(results: List[EvalResult]) -> None:
    by_image: Dict[str, List[EvalResult]] = {}
    for r in results:
        by_image.setdefault(r.image, []).append(r)

    for image in sorted(by_image.keys()):
        rows = sorted(by_image[image], key=lambda x: x.bade)
        print(f"\n[{image}]")
        print("mode               BADE    meanDE00  p90DE00  chromaBias  mean_dL  mean_da  mean_db")
        for r in rows:
            print(
                f"{r.mode:<18} {r.bade:7.3f}  {r.mean_de00:8.3f}  {r.p90_de00:7.3f}  "
                f"{r.chroma_bias:10.3f}  {r.mean_dL:7.3f}  {r.mean_da:7.3f}  {r.mean_db:7.3f}"
            )

    by_mode: Dict[str, List[EvalResult]] = {}
    for r in results:
        by_mode.setdefault(r.mode, []).append(r)
    print("\n[Overall mean by mode]")
    for mode in sorted(by_mode.keys()):
        rows = by_mode[mode]
        print(
            f"{mode:<18} BADE={np.mean([x.bade for x in rows]):.3f}  "
            f"meanDE00={np.mean([x.mean_de00 for x in rows]):.3f}  "
            f"chromaBias={np.mean([x.chroma_bias for x in rows]):.3f}"
        )


def main() -> int:
    args = parse_args()
    results: List[EvalResult] = []

    for image_stem in args.images:
        ref_path = find_reference_image(args.data_dir, image_stem)
        for mode in args.modes:
            preview_path = args.preview_dir / f"{image_stem}_preview_{mode}.png"
            if not preview_path.exists():
                print(f"[warn] missing preview, skip: {preview_path}")
                continue
            mask_path = args.preview_dir / f"{image_stem}_{mode}_source_mask.png"
            try:
                result = evaluate_pair(
                    image_stem=image_stem,
                    mode=mode,
                    ref_path=ref_path,
                    preview_path=preview_path,
                    mask_path=mask_path,
                    weight_chroma=float(args.weight_chroma),
                )
                results.append(result)
            except Exception as exc:
                print(f"[warn] failed {image_stem}/{mode}: {exc}")

    if not results:
        raise RuntimeError("No valid results generated.")

    print_report(results)

    if args.output_json is not None:
        output = {
            "meta": {
                "images": args.images,
                "modes": args.modes,
                "weight_chroma": float(args.weight_chroma),
            },
            "results": [asdict(r) for r in results],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        print(f"\nSaved report to {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

