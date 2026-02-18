#!/usr/bin/env python3
"""
Fit Stage A parameters (E, k) from single-color stair data.

Example:
  python -m modeling.pipeline.step2_fit_stage_a \
    --input modeling/output/params/1_single_stage.json \
    --output modeling/output/params/stage_A_parameters.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from modeling.core.math_utils import (
    logit,
    round_list,
    sigmoid,
    softplus,
    softplus_grad,
    softplus_inv,
)
from modeling.core.color_space import linear_rgb_to_lab_d65
from modeling.core.io_utils import load_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "modeling" / "output" / "params" / "1_single_stage.json"
DEFAULT_OUTPUT = REPO_ROOT / "modeling" / "output" / "params" / "stage_A_parameters.json"


@dataclass(frozen=True)
class Sample:
    step_layers: int
    measured_linear_rgb: np.ndarray
    measured_lab_d65: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit Stage A parameters from single-color stair data."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Path to 1_single_stage.json.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output JSON path for fitted parameters.")
    parser.add_argument("--fit-max-step", type=int, default=16,
                        help="Max step_layers to include in fitting loss.")
    parser.add_argument("--include-step-25", action="store_true",
                        help="Include step_layers==25 in fitting loss.")
    parser.add_argument("--weight-alpha", type=float, default=1.0,
                        help="Weight alpha for w_n = 1 + alpha/(n+1).")
    parser.add_argument("--lr", type=float, default=0.03,
                        help="Adam learning rate.")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Adam optimization steps per color.")
    parser.add_argument("--tol", type=float, default=1e-8,
                        help="Early stop if loss improves less than tol.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization noise.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def collect_stage_samples(
    datasets: Sequence[Dict[str, object]],
) -> Tuple[Dict[str, Dict[str, List[Sample]]], Dict[str, List[np.ndarray]]]:
    by_color: Dict[str, Dict[str, List[Sample]]] = {}
    c0_values: Dict[str, List[np.ndarray]] = {}

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

            meas_linear = entry.get("measured_linear_rgb")
            if not isinstance(meas_linear, (list, tuple)) or len(meas_linear) != 3:
                continue

            meas_linear_arr = np.array(meas_linear, dtype=np.float32)
            meas_lab = entry.get("measured_lab_d65")
            if isinstance(meas_lab, (list, tuple)) and len(meas_lab) == 3:
                meas_lab_arr = np.array(meas_lab, dtype=np.float32)
            else:
                meas_lab_arr = np.array(
                    linear_rgb_to_lab_d65(meas_linear_arr), dtype=np.float32
                )

            samples.append(Sample(step_layers, meas_linear_arr, meas_lab_arr))

            if step_layers == 0:
                c0_values.setdefault(substrate_id, []).append(meas_linear_arr)

        if samples:
            samples.sort(key=lambda s: s.step_layers)
            by_color.setdefault(color_name, {})[substrate_id] = samples

    return by_color, c0_values


def compute_c0_by_substrate(
    c0_values: Dict[str, List[np.ndarray]]
) -> Dict[str, np.ndarray]:
    c0_by_substrate: Dict[str, np.ndarray] = {}
    for substrate_id, values in c0_values.items():
        if not values:
            continue
        stacked = np.stack(values, axis=0)
        c0_by_substrate[substrate_id] = stacked.mean(axis=0)
    return c0_by_substrate


# ---------------------------------------------------------------------------
# Training arrays
# ---------------------------------------------------------------------------

def weight_for_step(step_layers: int, alpha: float) -> float:
    return 1.0 + alpha / (float(step_layers) + 1.0)


def build_training_arrays(
    samples_by_substrate: Dict[str, List[Sample]],
    c0_by_substrate: Dict[str, np.ndarray],
    fit_max_step: int,
    include_step_25: bool,
    weight_alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_list: List[int] = []
    meas_list: List[np.ndarray] = []
    c0_list: List[np.ndarray] = []
    weights_list: List[float] = []

    for substrate_id, samples in samples_by_substrate.items():
        if substrate_id not in c0_by_substrate:
            continue
        c0 = c0_by_substrate[substrate_id]
        for sample in samples:
            n = int(sample.step_layers)
            if n <= fit_max_step or (include_step_25 and n == 25):
                n_list.append(n)
                meas_list.append(sample.measured_linear_rgb)
                c0_list.append(c0)
                weights_list.append(weight_for_step(n, weight_alpha))

    if not n_list:
        raise ValueError("No training samples found for fitting.")

    n_arr = np.asarray(n_list, dtype=np.float32)
    meas_arr = np.stack(meas_list, axis=0).astype(np.float32)
    c0_arr = np.stack(c0_list, axis=0).astype(np.float32)
    weights = np.asarray(weights_list, dtype=np.float32)
    weight_scale = float(np.mean(weights)) if weights.size > 0 else 1.0
    weights = weights / max(weight_scale, 1e-6)
    return n_arr, meas_arr, c0_arr, weights


# ---------------------------------------------------------------------------
# Forward / loss / optimiser
# ---------------------------------------------------------------------------

def predict_linear_rgb(
    E: np.ndarray, k: np.ndarray, c0: np.ndarray,
    layer_height_mm: float, step_layers: int,
) -> np.ndarray:
    if step_layers <= 0:
        return c0
    thickness = float(step_layers) * float(layer_height_mm)
    decay = np.exp(-k * thickness)
    return E + (c0 - E) * decay


def compute_loss_and_grad(
    u: np.ndarray, v: np.ndarray,
    n_arr: np.ndarray, meas_arr: np.ndarray, c0_arr: np.ndarray,
    weights: np.ndarray, layer_height_mm: float,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    E = sigmoid(u)
    k = softplus(v)

    n = n_arr.reshape(-1, 1)
    weights_col = weights.reshape(-1, 1)
    thickness = n * float(layer_height_mm)
    decay = np.exp(-k * thickness)
    pred = E + (c0_arr - E) * decay
    diff = pred - meas_arr

    weight_sum = float(np.sum(weights_col))
    if weight_sum <= 0.0:
        weight_sum = 1.0
    loss = float(np.sum(weights_col * diff * diff) / weight_sum)

    dC_dE = 1.0 - decay
    dC_dk = -(c0_arr - E) * thickness * decay

    scale = 2.0 / weight_sum
    dL_dE = scale * np.sum(weights_col * diff * dC_dE, axis=0)
    dL_dk = scale * np.sum(weights_col * diff * dC_dk, axis=0)

    dE_du = E * (1.0 - E)
    dk_dv = softplus_grad(v)

    dL_du = dL_dE * dE_du
    dL_dv = dL_dk * dk_dv
    return loss, dL_du, dL_dv, E, k


def adam_optimize(
    u: np.ndarray, v: np.ndarray,
    n_arr: np.ndarray, meas_arr: np.ndarray, c0_arr: np.ndarray,
    weights: np.ndarray, layer_height_mm: float,
    steps: int, lr: float, tol: float,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m_u = np.zeros_like(u)
    v_u = np.zeros_like(u)
    m_v = np.zeros_like(v)
    v_v = np.zeros_like(v)

    prev_loss: Optional[float] = None
    last_loss = float("inf")
    for step in range(1, max(1, steps) + 1):
        loss, grad_u, grad_v, _, _ = compute_loss_and_grad(
            u, v, n_arr, meas_arr, c0_arr, weights, layer_height_mm,
        )
        if prev_loss is not None and abs(prev_loss - loss) < tol:
            return sigmoid(u), softplus(v), loss, step
        prev_loss = loss
        last_loss = loss

        m_u = beta1 * m_u + (1.0 - beta1) * grad_u
        v_u = beta2 * v_u + (1.0 - beta2) * (grad_u * grad_u)
        m_v = beta1 * m_v + (1.0 - beta1) * grad_v
        v_v = beta2 * v_v + (1.0 - beta2) * (grad_v * grad_v)

        u -= lr * (m_u / (1.0 - beta1**step)) / (np.sqrt(v_u / (1.0 - beta2**step)) + eps)
        v -= lr * (m_v / (1.0 - beta1**step)) / (np.sqrt(v_v / (1.0 - beta2**step)) + eps)

    return sigmoid(u), softplus(v), last_loss, steps


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def select_saturation_color(samples: Sequence[Sample]) -> Optional[np.ndarray]:
    for sample in samples:
        if sample.step_layers == 25:
            return sample.measured_linear_rgb
    if samples:
        return max(samples, key=lambda s: s.step_layers).measured_linear_rgb
    return None


def initial_E(samples_by_substrate: Dict[str, List[Sample]]) -> np.ndarray:
    values: List[np.ndarray] = []
    for samples in samples_by_substrate.values():
        candidate = select_saturation_color(samples)
        if candidate is not None:
            values.append(candidate)
    if not values:
        return np.array([0.5, 0.5, 0.5], dtype=np.float32)
    return np.stack(values, axis=0).mean(axis=0)


def evaluate_color(
    samples_by_substrate: Dict[str, List[Sample]],
    E: np.ndarray, k: np.ndarray,
    c0_by_substrate: Dict[str, np.ndarray],
    layer_height_mm: float,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, float], List[float]]:
    per_substrate: Dict[str, Dict[str, object]] = {}
    all_deltas: List[float] = []

    for substrate_id, samples in samples_by_substrate.items():
        c0 = c0_by_substrate.get(substrate_id)
        if c0 is None:
            continue
        deltas: List[float] = []
        delta_by_step: List[Dict[str, object]] = []
        for sample in samples:
            pred_linear = predict_linear_rgb(E, k, c0, layer_height_mm, sample.step_layers)
            pred_lab = np.array(linear_rgb_to_lab_d65(pred_linear), dtype=np.float32)
            meas_lab = sample.measured_lab_d65
            delta = float(np.linalg.norm(pred_lab - meas_lab))
            deltas.append(delta)
            delta_by_step.append({"step_layers": int(sample.step_layers), "delta_e": round(delta, 4)})
            all_deltas.append(delta)

        if deltas:
            per_substrate[substrate_id] = {
                "mean_delta_e": round(float(np.mean(deltas)), 4),
                "max_delta_e": round(float(np.max(deltas)), 4),
                "delta_e_by_step": delta_by_step,
            }

    overall: Dict[str, float] = {}
    if all_deltas:
        overall = {
            "mean_delta_e": round(float(np.mean(all_deltas)), 4),
            "max_delta_e": round(float(np.max(all_deltas)), 4),
        }
    return per_substrate, overall, all_deltas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)

    data = load_json(args.input)
    stage_config = data.get("stage_config") or {}
    layer_height_mm = float(stage_config.get("layer_height_mm", 0.04))
    datasets = data.get("datasets") or []

    color_map, c0_values = collect_stage_samples(datasets)
    c0_by_substrate = compute_c0_by_substrate(c0_values)
    if not c0_by_substrate:
        raise ValueError("Failed to find any step_layers==0 base samples.")

    warnings: List[str] = []
    output_channels: List[Dict[str, object]] = []
    validation_channels: List[Dict[str, object]] = []
    all_delta_e: List[float] = []

    sorted_colors = sorted(color_map.keys())
    for channel_index, color_name in enumerate(sorted_colors):
        samples_by_substrate = color_map[color_name]
        try:
            n_arr, meas_arr, c0_arr, weights = build_training_arrays(
                samples_by_substrate, c0_by_substrate,
                args.fit_max_step, args.include_step_25, args.weight_alpha,
            )
        except ValueError as exc:
            warnings.append(f"Skip color {color_name}: {exc}")
            continue

        E_init = initial_E(samples_by_substrate)
        u = logit(E_init) + np.random.normal(0.0, 0.02, size=(3,)).astype(np.float32)
        v = softplus_inv(np.array([1.0, 1.0, 1.0], dtype=np.float32))
        v += np.random.normal(0.0, 0.02, size=v.shape).astype(np.float32)

        E_fit, k_fit, loss, steps_used = adam_optimize(
            u, v, n_arr, meas_arr, c0_arr, weights, layer_height_mm,
            args.steps, args.lr, args.tol,
        )

        output_channels.append({
            "channel_index": channel_index,
            "color_name": color_name,
            "E": round_list(E_fit, 6),
            "k": round_list(k_fit, 6),
            "loss": round(float(loss), 8),
            "sample_count": int(n_arr.shape[0]),
            "fit_steps": int(steps_used),
            "substrates": sorted(samples_by_substrate.keys()),
        })

        per_substrate, overall, deltas = evaluate_color(
            samples_by_substrate, E_fit, k_fit, c0_by_substrate, layer_height_mm,
        )
        validation_channels.append({
            "channel_index": channel_index,
            "color_name": color_name,
            "substrates": per_substrate,
            "overall": overall,
        })
        all_delta_e.extend(deltas)

    output = {
        "meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_json": str(args.input),
            "fitting_method": "stage_A",
            "optimizer": {
                "name": "adam", "lr": args.lr, "steps": args.steps,
                "tol": args.tol, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8,
            },
            "loss_space": "linear_rgb",
            "fit_max_step": args.fit_max_step,
            "include_step_25": bool(args.include_step_25),
            "weight_alpha": args.weight_alpha,
            "layer_height_mm": layer_height_mm,
            "substrates": sorted(c0_by_substrate.keys()),
            "warnings": warnings,
        },
        "fitted_substrates": {
            sid: {"C0_linear_rgb": round_list(c0, 6)}
            for sid, c0 in c0_by_substrate.items()
        },
        "parameters": {"channels": output_channels},
        "validation": {
            "per_channel": validation_channels,
            "overall": {
                "mean_delta_e": round(float(np.mean(all_delta_e)), 4) if all_delta_e else None,
                "max_delta_e": round(float(np.max(all_delta_e)), 4) if all_delta_e else None,
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Saved Stage A parameters to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
