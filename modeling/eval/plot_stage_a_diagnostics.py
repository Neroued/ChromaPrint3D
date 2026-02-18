#!/usr/bin/env python3
"""
Plot Stage A fitting diagnostics from stage_A_parameters.json.

Example:
  python -m modeling.eval.plot_stage_a_diagnostics \
    --stage-json modeling/output/params/1_single_stage.json \
    --params-json modeling/output/params/stage_A_parameters.json \
    --output-dir modeling/output/reports/plots
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE_JSON = REPO_ROOT / "modeling" / "1_single_stage.json"
DEFAULT_PARAMS_JSON = REPO_ROOT / "modeling" / "stage_A_parameters.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "modeling" / "plots" / "stage_A_diagnostics"


@dataclass(frozen=True)
class Sample:
    step_layers: int
    measured_linear_rgb: np.ndarray
    measured_lab_d65: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot diagnostics for Stage A fitting results."
    )
    parser.add_argument(
        "--stage-json",
        type=Path,
        default=DEFAULT_STAGE_JSON,
        help="Path to 1_single_stage.json.",
    )
    parser.add_argument(
        "--params-json",
        type=Path,
        default=DEFAULT_PARAMS_JSON,
        help="Path to stage_A_parameters.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save diagnostic plots.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Optional max step_layers to include in plots.",
    )
    parser.add_argument(
        "--include-step-25",
        action="store_true",
        help="Always include step_layers==25 even if max-step < 25.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=8.0,
        help="Figure width (inches).",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=5.0,
        help="Figure height (inches).",
    )
    return parser.parse_args()


def normalize_key(value: str) -> str:
    return "".join(ch for ch in value.lower().strip() if ch.isalnum() or ch in ("-", "_", " "))


def sanitize_filename(value: str) -> str:
    safe: List[str] = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        elif ch in (" ", "."):
            safe.append("_")
    return "".join(safe).strip("_") or "unknown"


def linear_rgb_to_xyz(linear_rgb: np.ndarray) -> np.ndarray:
    rgb_to_xyz = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    return rgb_to_xyz @ linear_rgb


def xyz_to_lab(xyz: np.ndarray, white: np.ndarray) -> np.ndarray:
    delta = 6.0 / 29.0
    delta_cubed = delta ** 3
    scale = 1.0 / (3.0 * delta * delta)

    xyz_n = xyz / white
    f = np.where(
        xyz_n > delta_cubed,
        np.cbrt(xyz_n),
        xyz_n * scale + 4.0 / 29.0,
    )

    l = 116.0 * f[1] - 16.0
    a = 500.0 * (f[0] - f[1])
    b = 200.0 * (f[1] - f[2])
    return np.array([l, a, b], dtype=np.float32)


def linear_rgb_to_lab_d65(linear_rgb: np.ndarray) -> np.ndarray:
    white_d65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    linear = np.clip(np.asarray(linear_rgb, dtype=np.float32), 0.0, 1.0)
    xyz_d65 = linear_rgb_to_xyz(linear)
    lab = xyz_to_lab(xyz_d65, white_d65)
    return np.array([round(float(v), 2) for v in lab], dtype=np.float32)


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_stage_samples(
    datasets: Sequence[Dict[str, object]],
    max_step: Optional[int],
    include_step_25: bool,
) -> Dict[str, Dict[str, List[Sample]]]:
    by_color: Dict[str, Dict[str, List[Sample]]] = {}
    for dataset in datasets:
        substrate_id = str(dataset.get("substrate_id", "unknown"))
        color_name = str(dataset.get("color_name", "unknown"))
        aggregated = dataset.get("aggregated") or []
        samples: List[Sample] = []
        for entry in aggregated:
            if "step_layers" not in entry or "measured_linear_rgb" not in entry:
                continue
            try:
                step_layers = int(entry["step_layers"])
            except (TypeError, ValueError):
                continue
            if max_step is not None and step_layers > max_step:
                if not (include_step_25 and step_layers == 25):
                    continue
            meas_linear = entry.get("measured_linear_rgb")
            if not isinstance(meas_linear, (list, tuple)) or len(meas_linear) != 3:
                continue
            meas_linear_arr = np.array(meas_linear, dtype=np.float32)
            meas_lab = entry.get("measured_lab_d65")
            if isinstance(meas_lab, (list, tuple)) and len(meas_lab) == 3:
                meas_lab_arr = np.array(meas_lab, dtype=np.float32)
            else:
                meas_lab_arr = linear_rgb_to_lab_d65(meas_linear_arr)
            samples.append(Sample(step_layers, meas_linear_arr, meas_lab_arr))
        if samples:
            samples.sort(key=lambda s: s.step_layers)
            by_color.setdefault(color_name, {})[substrate_id] = samples
    return by_color


def compute_c0_from_stage(
    datasets: Sequence[Dict[str, object]],
) -> Dict[str, np.ndarray]:
    c0_values: Dict[str, List[np.ndarray]] = {}
    for dataset in datasets:
        substrate_id = str(dataset.get("substrate_id", "unknown"))
        aggregated = dataset.get("aggregated") or []
        for entry in aggregated:
            try:
                step_layers = int(entry.get("step_layers"))
            except (TypeError, ValueError):
                continue
            if step_layers != 0:
                continue
            meas_linear = entry.get("measured_linear_rgb")
            if not isinstance(meas_linear, (list, tuple)) or len(meas_linear) != 3:
                continue
            c0_values.setdefault(substrate_id, []).append(
                np.array(meas_linear, dtype=np.float32)
            )
    c0_by_substrate: Dict[str, np.ndarray] = {}
    for substrate_id, values in c0_values.items():
        if values:
            c0_by_substrate[substrate_id] = np.stack(values, axis=0).mean(axis=0)
    return c0_by_substrate


def load_c0_from_params(params_json: Dict[str, object]) -> Dict[str, np.ndarray]:
    fitted = params_json.get("fitted_substrates") or {}
    c0_by_substrate: Dict[str, np.ndarray] = {}
    for substrate_id, entry in fitted.items():
        if not isinstance(entry, dict):
            continue
        c0 = entry.get("C0_linear_rgb")
        if isinstance(c0, (list, tuple)) and len(c0) == 3:
            c0_by_substrate[substrate_id] = np.array(c0, dtype=np.float32)
    return c0_by_substrate


def build_param_map(params_json: Dict[str, object]) -> Dict[str, Dict[str, np.ndarray]]:
    channels = params_json.get("parameters", {}).get("channels", [])
    param_map: Dict[str, Dict[str, np.ndarray]] = {}
    for channel in channels:
        if not isinstance(channel, dict):
            continue
        color_name = str(channel.get("color_name", "unknown"))
        E = channel.get("E")
        k = channel.get("k")
        if not (isinstance(E, (list, tuple)) and isinstance(k, (list, tuple))):
            continue
        if len(E) != 3 or len(k) != 3:
            continue
        key = normalize_key(color_name)
        if key in param_map:
            continue
        param_map[key] = {
            "color_name": color_name,
            "E": np.array(E, dtype=np.float32),
            "k": np.array(k, dtype=np.float32),
        }
    return param_map


def predict_linear_rgb(
    E: np.ndarray, k: np.ndarray, c0: np.ndarray, layer_height_mm: float, step_layers: int
) -> np.ndarray:
    if step_layers <= 0:
        return c0
    thickness = float(step_layers) * float(layer_height_mm)
    decay = np.exp(-k * thickness)
    return E + (c0 - E) * decay


def delta_e_lab(pred_lab: np.ndarray, meas_lab: np.ndarray) -> float:
    return float(np.linalg.norm(pred_lab - meas_lab))


def plot_delta_e(
    steps: Sequence[int],
    deltas: Sequence[float],
    title: str,
    output_path: Path,
    dpi: int,
    width: float,
    height: float,
) -> None:
    plt.figure(figsize=(width, height))
    plt.plot(steps, deltas, marker="o", linewidth=1.8)
    plt.xlabel("step_layers")
    plt.ylabel("Delta E (Lab D65)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_linear_rgb_fit(
    steps: Sequence[int],
    measured: np.ndarray,
    predicted: np.ndarray,
    title: str,
    output_path: Path,
    dpi: int,
    width: float,
    height: float,
) -> None:
    plt.figure(figsize=(width, height))
    colors = ["tab:red", "tab:green", "tab:blue"]
    labels = ["R", "G", "B"]
    for idx in range(3):
        plt.plot(
            steps,
            measured[:, idx],
            color=colors[idx],
            marker="o",
            linewidth=1.8,
            label=f"{labels[idx]} measured",
        )
        plt.plot(
            steps,
            predicted[:, idx],
            color=colors[idx],
            linestyle="--",
            linewidth=1.6,
            label=f"{labels[idx]} predicted",
        )
    plt.xlabel("step_layers")
    plt.ylabel("Linear RGB")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_delta_e_histogram(
    deltas: Sequence[float],
    title: str,
    output_path: Path,
    dpi: int,
    width: float,
    height: float,
) -> None:
    plt.figure(figsize=(width, height))
    plt.hist(deltas, bins=30, color="tab:blue", alpha=0.75)
    plt.xlabel("Delta E (Lab D65)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_delta_e_by_color(
    color_names: Sequence[str],
    mean_deltas: Sequence[float],
    title: str,
    output_path: Path,
    dpi: int,
    width: float,
    height: float,
) -> None:
    plt.figure(figsize=(max(width, 10.0), height))
    x = np.arange(len(color_names))
    plt.bar(x, mean_deltas, color="tab:orange", alpha=0.8)
    plt.xticks(x, color_names, rotation=45, ha="right")
    plt.ylabel("Mean Delta E (Lab D65)")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def main() -> int:
    args = parse_args()

    stage_json = load_json(args.stage_json)
    params_json = load_json(args.params_json)
    layer_height_mm = float(
        stage_json.get("stage_config", {}).get("layer_height_mm", 0.04)
    )

    datasets = stage_json.get("datasets") or []
    samples_by_color = collect_stage_samples(
        datasets,
        max_step=args.max_step,
        include_step_25=args.include_step_25,
    )
    param_map = build_param_map(params_json)

    c0_by_substrate = load_c0_from_params(params_json)
    if not c0_by_substrate:
        c0_by_substrate = compute_c0_from_stage(datasets)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    delta_e_all: List[float] = []
    delta_e_by_substrate: Dict[str, List[float]] = {}
    mean_delta_by_color_sub: Dict[str, Dict[str, float]] = {}

    for color_name, substrate_map in samples_by_color.items():
        param = param_map.get(normalize_key(color_name))
        if not param:
            warnings.append(f"Missing parameters for color {color_name}")
            continue
        E = param["E"]
        k = param["k"]

        for substrate_id, samples in substrate_map.items():
            c0 = c0_by_substrate.get(substrate_id)
            if c0 is None:
                warnings.append(
                    f"Missing C0 for substrate {substrate_id} (color {color_name})"
                )
                continue

            steps = [s.step_layers for s in samples]
            measured = np.stack([s.measured_linear_rgb for s in samples], axis=0)
            predicted = np.stack(
                [
                    predict_linear_rgb(E, k, c0, layer_height_mm, s.step_layers)
                    for s in samples
                ],
                axis=0,
            )
            deltas = []
            for s, pred in zip(samples, predicted):
                pred_lab = linear_rgb_to_lab_d65(pred)
                deltas.append(delta_e_lab(pred_lab, s.measured_lab_d65))
            delta_e_all.extend(deltas)
            delta_e_by_substrate.setdefault(substrate_id, []).extend(deltas)
            mean_delta_by_color_sub.setdefault(substrate_id, {})[color_name] = float(
                np.mean(deltas)
            )

            substrate_slug = sanitize_filename(substrate_id)
            color_slug = sanitize_filename(color_name)

            plot_delta_e(
                steps=steps,
                deltas=deltas,
                title=f"Delta E vs step | {color_name} | {substrate_id}",
                output_path=output_dir
                / "per_color"
                / substrate_slug
                / f"deltaE__{color_slug}.png",
                dpi=args.dpi,
                width=args.width,
                height=args.height,
            )

            plot_linear_rgb_fit(
                steps=steps,
                measured=measured,
                predicted=predicted,
                title=f"Linear RGB fit | {color_name} | {substrate_id}",
                output_path=output_dir
                / "per_color"
                / substrate_slug
                / f"linear_rgb__{color_slug}.png",
                dpi=args.dpi,
                width=args.width,
                height=args.height,
            )

    if delta_e_all:
        plot_delta_e_histogram(
            deltas=delta_e_all,
            title="Delta E distribution (all substrates)",
            output_path=output_dir / "summary" / "deltaE_hist__all.png",
            dpi=args.dpi,
            width=args.width,
            height=args.height,
        )

    for substrate_id, deltas in delta_e_by_substrate.items():
        substrate_slug = sanitize_filename(substrate_id)
        plot_delta_e_histogram(
            deltas=deltas,
            title=f"Delta E distribution | {substrate_id}",
            output_path=output_dir
            / "summary"
            / f"deltaE_hist__{substrate_slug}.png",
            dpi=args.dpi,
            width=args.width,
            height=args.height,
        )

        color_means = mean_delta_by_color_sub.get(substrate_id, {})
        if color_means:
            names_sorted = sorted(color_means.keys())
            means_sorted = [color_means[name] for name in names_sorted]
            plot_delta_e_by_color(
                color_names=names_sorted,
                mean_deltas=means_sorted,
                title=f"Mean Delta E by color | {substrate_id}",
                output_path=output_dir
                / "summary"
                / f"deltaE_by_color__{substrate_slug}.png",
                dpi=args.dpi,
                width=args.width,
                height=args.height,
            )

    if warnings:
        warning_path = output_dir / "warnings.txt"
        warning_path.write_text("\n".join(warnings), encoding="utf-8")

    print(f"Saved diagnostics to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
