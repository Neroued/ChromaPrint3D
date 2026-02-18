#!/usr/bin/env python3
"""
Fit Stage B parameters (E, k, gamma, delta, C0) from ColorDB combinations.

Example:
  python -m modeling.pipeline.step3_fit_stage_b \
    --db modeling/dbs/RYBW_008_corrected.json \
    --stage-a modeling/output/params/stage_A_parameters.json \
    --output modeling/output/params/stage_B_parameters.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from modeling.core.math_utils import (
    logit,
    round_list,
    sigmoid,
    softplus,
    softplus_grad,
    softplus_inv,
)
from modeling.core.color_space import (
    lab_grad_from_linear_batch,
    linear_rgb_to_opencv_lab,
    linear_rgb_to_opencv_lab_batch,
)
from modeling.core.io_utils import load_json, normalize_label, parse_layer_order, resolve_db_paths


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO_ROOT / "data" / "RYBW_008_corrected.json"
DEFAULT_STAGE_A = REPO_ROOT / "modeling" / "output" / "params" / "stage_A_parameters.json"
DEFAULT_OUTPUT = REPO_ROOT / "modeling" / "output" / "params" / "stage_B_parameters.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DbEntry:
    recipe: List[int]
    lab: np.ndarray
    micro_layers: np.ndarray
    base_channel_idx: int
    source_db: str
    substrate_idx: int
    layer_height_mm: float


@dataclass(frozen=True)
class OptimConfig:
    micro_layer_height: float
    micro_layer_count: Optional[int]
    substrate_mode: str
    lambda_reg: float
    lambda_c0: float
    lambda_height_scale: float
    lambda_neighbor: float
    lab_eps: float
    height_ref_mm: float
    enable_height_scale: bool
    E0: np.ndarray
    k0: np.ndarray
    c0_anchor: np.ndarray


@dataclass(frozen=True)
class StageAParams:
    names: List[str]
    name_to_idx: Dict[str, int]
    E0: np.ndarray
    k0: np.ndarray
    measured_c0: Dict[str, np.ndarray]  # fitted_substrates 实测基板颜色


@dataclass(frozen=True)
class PreparedEntries:
    target_lab: np.ndarray
    base_channel_idx: np.ndarray
    substrate_idx: np.ndarray
    layer_height_mm: np.ndarray
    micro_layers_matrix: Optional[np.ndarray]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit Stage B parameters from ColorDB combinations."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--stage-a", type=Path, default=DEFAULT_STAGE_A)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--micro-layer-height", type=float, default=0.04)
    parser.add_argument("--micro-layer-count", type=int, default=20)
    parser.add_argument("--substrate-mode", choices=("boundary", "material"), default="boundary")
    parser.add_argument("--lambda-reg", type=float, default=0.1)
    parser.add_argument("--lambda-c0", type=float, default=2.0,
                        help="C0 regularization strength (recommended 1.0~5.0; "
                             "higher values keep C0 closer to Stage A measured substrate color)")
    parser.add_argument("--lambda-height-scale", type=float, default=0.05)
    parser.add_argument("--lambda-neighbor", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--stage1-steps", type=int, default=600)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lab-eps", type=float, default=1e-4)
    parser.add_argument("--height-ref-mm", type=float, default=0.04)
    parser.add_argument("--disable-learn-c0", action="store_true")
    parser.add_argument("--disable-height-scale", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Stage A loading
# ---------------------------------------------------------------------------

def load_stage_a_params(path: Path) -> StageAParams:
    data = load_json(path)
    channels = data.get("parameters", {}).get("channels", [])
    if not isinstance(channels, list) or not channels:
        raise ValueError("Stage A file missing parameters.channels")

    names: List[str] = []
    name_to_idx: Dict[str, int] = {}
    E_list: List[np.ndarray] = []
    k_list: List[np.ndarray] = []
    for idx, entry in enumerate(channels):
        color_name = str(entry.get("color_name", f"channel_{idx}"))
        normalized = normalize_label(color_name)
        if not normalized:
            raise ValueError(f"Invalid Stage A color name: {color_name}")
        if normalized in name_to_idx:
            raise ValueError(f"Duplicate Stage A color name: {color_name}")
        E = np.array(entry.get("E", [0.5, 0.5, 0.5]), dtype=np.float32)
        k = np.array(entry.get("k", [1.0, 1.0, 1.0]), dtype=np.float32)
        if E.shape != (3,) or k.shape != (3,):
            raise ValueError(f"Invalid E/k shape for color {color_name}")
        names.append(color_name)
        name_to_idx[normalized] = idx
        E_list.append(E)
        k_list.append(k)

    # 解析 fitted_substrates 中的实测基板颜色
    measured_c0: Dict[str, np.ndarray] = {}
    fitted_substrates = data.get("fitted_substrates", {})
    for key, val in fitted_substrates.items():
        c0_rgb = val.get("C0_linear_rgb")
        if c0_rgb is not None:
            measured_c0[key] = np.array(c0_rgb, dtype=np.float32)

    return StageAParams(
        names=names, name_to_idx=name_to_idx,
        E0=np.stack(E_list, axis=0), k0=np.stack(k_list, axis=0),
        measured_c0=measured_c0,
    )


# ---------------------------------------------------------------------------
# ColorDB loading
# ---------------------------------------------------------------------------

def map_palette_to_stage(
    palette: Sequence[Dict[str, object]], stage: StageAParams, db_path: Path,
) -> List[int]:
    mapping: List[int] = []
    missing: List[str] = []
    for channel in palette:
        color = str(channel.get("color", ""))
        key = normalize_label(color)
        if key not in stage.name_to_idx:
            missing.append(color)
            mapping.append(-1)
            continue
        mapping.append(stage.name_to_idx[key])
    if missing:
        available = sorted(stage.name_to_idx.keys())
        raise ValueError(
            f"{db_path}: Stage A missing palette colors: "
            + ", ".join(missing) + f". Available: {available}"
        )
    return mapping


def expand_recipe_to_micro_layers(
    recipe: Sequence[int], layer_order: str, layer_height_mm: float,
    micro_layer_height: float, micro_layer_count: Optional[int],
    base_channel_idx: int, substrate_mode: str, base_layers: int,
) -> List[int]:
    if layer_order == "Top2Bottom":
        recipe_bt = list(reversed(recipe))
    elif layer_order == "Bottom2Top":
        recipe_bt = list(recipe)
    else:
        raise ValueError(f"Unsupported layer_order: {layer_order}")

    ratio = float(layer_height_mm) / float(micro_layer_height)
    n_u = int(round(ratio))
    if n_u <= 0:
        raise ValueError("Invalid layer height ratio")
    if not math.isfinite(ratio) or abs(ratio - n_u) > 1e-3:
        raise ValueError(f"Layer height {layer_height_mm} is not compatible with {micro_layer_height}")

    micro: List[int] = []
    for ch in recipe_bt:
        micro.extend([int(ch)] * n_u)

    base_micro = 0
    if substrate_mode == "material" and base_layers > 0:
        base_thickness = float(base_layers) * float(layer_height_mm)
        base_micro = int(round(base_thickness / float(micro_layer_height)))

    total = base_micro + len(micro)
    if substrate_mode == "boundary":
        # boundary 模式: C0 已经是边界色 (基板+底层叠完后的颜色)，
        # 不需要额外填充白色微层，仅返回配方本身的微层序列。
        return micro
    if micro_layer_count is not None and micro_layer_count > 0:
        if micro_layer_count < total:
            raise ValueError(f"micro_layer_count {micro_layer_count} < required {total}")
        pad = micro_layer_count - total
        return [base_channel_idx] * (base_micro + pad) + micro
    return [base_channel_idx] * base_micro + micro


def load_db_entries(
    db_path: Path, stage: StageAParams, micro_layer_height: float,
    micro_layer_count: Optional[int], substrate_mode: str, substrate_idx: int,
) -> Tuple[List[DbEntry], Dict[str, object], List[str], List[int]]:
    db = load_json(db_path)
    palette = db.get("palette", [])
    if not isinstance(palette, list) or not palette:
        raise ValueError(f"{db_path}: ColorDB palette missing or empty")

    palette_map = map_palette_to_stage(palette, stage, db_path)
    layer_height_mm = float(db.get("layer_height_mm", 0.08))
    layer_order = parse_layer_order(db.get("layer_order", "Top2Bottom"))
    base_layers = int(db.get("base_layers", 0))
    base_channel_idx = int(db.get("base_channel_idx", 0))
    if base_channel_idx < 0 or base_channel_idx >= len(palette_map):
        raise ValueError(f"{db_path}: base_channel_idx out of range: {base_channel_idx}")
    base_channel_mapped = palette_map[base_channel_idx]

    entries_raw = db.get("entries", [])
    if not isinstance(entries_raw, list) or not entries_raw:
        raise ValueError(f"{db_path}: ColorDB entries missing or empty")

    entries: List[DbEntry] = []
    used_channels: List[int] = []
    skipped = 0
    for item in entries_raw:
        recipe = item.get("recipe")
        lab = item.get("lab")
        if not isinstance(recipe, list) or not isinstance(lab, list):
            skipped += 1; continue
        if len(lab) != 3:
            skipped += 1; continue

        mapped_recipe: List[int] = []
        invalid = False
        for v in recipe:
            idx = int(v)
            if idx < 0 or idx >= len(palette_map):
                invalid = True; break
            mapped_recipe.append(palette_map[idx])
        if invalid:
            skipped += 1; continue

        micro_layers = expand_recipe_to_micro_layers(
            recipe=mapped_recipe, layer_order=layer_order,
            layer_height_mm=layer_height_mm, micro_layer_height=micro_layer_height,
            micro_layer_count=micro_layer_count, base_channel_idx=base_channel_mapped,
            substrate_mode=substrate_mode, base_layers=base_layers,
        )
        entries.append(DbEntry(
            recipe=mapped_recipe, lab=np.array(lab, dtype=np.float32),
            micro_layers=np.array(micro_layers, dtype=np.int32),
            base_channel_idx=base_channel_mapped, source_db=str(db_path),
            substrate_idx=int(substrate_idx), layer_height_mm=layer_height_mm,
        ))
        used_channels.extend(mapped_recipe)

    warnings: List[str] = []
    if skipped:
        warnings.append(f"{db_path}: skipped {skipped} invalid entries")
    if substrate_mode == "material":
        used_channels.append(base_channel_mapped)

    summary = {
        "db_path": str(db_path), "entry_count": len(entries),
        "layer_height_mm": layer_height_mm, "layer_order": layer_order,
        "base_layers": base_layers, "base_channel_idx": base_channel_idx,
        "base_channel_stage_idx": base_channel_mapped,
        "substrate_idx": int(substrate_idx),
    }
    return entries, summary, warnings, used_channels


def prepare_entries(entries: Sequence[DbEntry]) -> PreparedEntries:
    count = len(entries)
    if count == 0:
        return PreparedEntries(
            target_lab=np.zeros((0, 3), dtype=np.float32),
            base_channel_idx=np.zeros((0,), dtype=np.int32),
            substrate_idx=np.zeros((0,), dtype=np.int32),
            layer_height_mm=np.zeros((0,), dtype=np.float32),
            micro_layers_matrix=None,
        )
    target_lab = np.stack([e.lab for e in entries], axis=0).astype(np.float32)
    base_channel_idx = np.array([e.base_channel_idx for e in entries], dtype=np.int32)
    substrate_idx = np.array([e.substrate_idx for e in entries], dtype=np.int32)
    layer_height_mm = np.array([e.layer_height_mm for e in entries], dtype=np.float32)
    lengths = {int(e.micro_layers.shape[0]) for e in entries}
    micro_layers_matrix: Optional[np.ndarray] = None
    if len(lengths) == 1:
        layer_count = next(iter(lengths))
        if layer_count > 0:
            micro_layers_matrix = np.stack([e.micro_layers for e in entries], axis=0).astype(np.int32)
    return PreparedEntries(
        target_lab=target_lab, base_channel_idx=base_channel_idx,
        substrate_idx=substrate_idx, layer_height_mm=layer_height_mm,
        micro_layers_matrix=micro_layers_matrix,
    )


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def _height_delta(layer_height_mm: np.ndarray, height_ref_mm: float) -> np.ndarray:
    ref = max(float(height_ref_mm), 1e-6)
    return np.asarray(layer_height_mm, dtype=np.float32) / ref - 1.0


def forward_with_grad(
    micro_layers: Sequence[int], base_channel_idx: int, substrate_idx: int,
    layer_height_mm: float, E: np.ndarray, k: np.ndarray, c0_boundary: np.ndarray,
    gamma: np.ndarray, delta: np.ndarray, micro_layer_height: float,
    substrate_mode: str, height_ref_mm: float, enable_height_scale: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_channels = E.shape[0]
    num_substrates = c0_boundary.shape[0]
    t = float(micro_layer_height)
    h_delta = float(_height_delta(np.array([layer_height_mm], dtype=np.float32), height_ref_mm)[0])

    c = np.zeros(3, dtype=np.float32) if substrate_mode == "material" else np.asarray(c0_boundary[substrate_idx], dtype=np.float32).copy()
    dE = np.zeros_like(E); dK = np.zeros_like(k)
    dC0 = np.zeros((num_substrates, 3), dtype=np.float32)
    dGamma = np.zeros((num_channels, 3), dtype=np.float32)
    dDelta = np.zeros((num_channels, num_channels, 3), dtype=np.float32)
    if substrate_mode != "material":
        dC0[substrate_idx, :] = 1.0

    prev_idx = int(base_channel_idx)
    for idx in micro_layers:
        i = int(idx)
        if i < 0 or i >= num_channels:
            raise ValueError(f"Channel index out of range: {i}")
        scale = float(np.exp(np.clip(gamma[i] * h_delta, -20.0, 20.0))) if enable_height_scale else 1.0
        k_scaled = k[i] * scale
        k_eff = k_scaled + delta[i, prev_idx]
        T = np.exp(-np.minimum(k_eff * t, 50.0))
        one_minus_T = 1.0 - T
        c_prev = c

        dE *= T; dK *= T; dC0 *= T; dGamma *= T; dDelta *= T
        dE[i] += one_minus_T
        common = (c_prev - E[i]) * (-t * T)
        dK[i] += common * scale
        if enable_height_scale:
            dGamma[i] += common * (k_scaled * h_delta)
        dDelta[i, prev_idx] += common
        c = one_minus_T * E[i] + T * c_prev
        prev_idx = i

    return c, dE, dK, dC0, dGamma, dDelta


def forward_predict_batch(
    micro_layers: np.ndarray, base_channel_idx: np.ndarray,
    substrate_idx: np.ndarray, layer_height_mm: np.ndarray,
    E: np.ndarray, k: np.ndarray, c0_boundary: np.ndarray,
    gamma: np.ndarray, delta: np.ndarray, micro_layer_height: float,
    substrate_mode: str, height_ref_mm: float, enable_height_scale: bool,
) -> np.ndarray:
    num_entries = micro_layers.shape[0]
    num_channels = E.shape[0]
    t = float(micro_layer_height)
    h_delta = _height_delta(layer_height_mm, height_ref_mm)
    prev_idx = base_channel_idx.astype(np.int32).copy()
    c = np.zeros((num_entries, 3), dtype=np.float32) if substrate_mode == "material" else np.asarray(c0_boundary[substrate_idx], dtype=np.float32).copy()

    for layer in range(micro_layers.shape[1]):
        idx = micro_layers[:, layer]
        scale = np.exp(np.clip(gamma[idx] * h_delta, -20.0, 20.0)).astype(np.float32) if enable_height_scale else np.ones_like(h_delta, dtype=np.float32)
        k_scaled = k[idx] * scale[:, None]
        k_eff = k_scaled + delta[idx, prev_idx, :]
        T = np.exp(-np.minimum(k_eff * t, 50.0))
        c = (1.0 - T) * E[idx] + T * c
        prev_idx = idx
    return c


def forward_with_grad_batch(
    micro_layers: np.ndarray, base_channel_idx: np.ndarray,
    substrate_idx: np.ndarray, layer_height_mm: np.ndarray,
    E: np.ndarray, k: np.ndarray, c0_boundary: np.ndarray,
    gamma: np.ndarray, delta: np.ndarray, micro_layer_height: float,
    substrate_mode: str, height_ref_mm: float, enable_height_scale: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_entries = micro_layers.shape[0]
    num_channels = E.shape[0]
    num_substrates = c0_boundary.shape[0]
    t = float(micro_layer_height)
    rows = np.arange(num_entries, dtype=np.int64)
    h_delta = _height_delta(layer_height_mm, height_ref_mm)
    prev_idx = base_channel_idx.astype(np.int32).copy()

    c = np.zeros((num_entries, 3), dtype=np.float32) if substrate_mode == "material" else np.asarray(c0_boundary[substrate_idx], dtype=np.float32).copy()
    dE = np.zeros((num_entries, num_channels, 3), dtype=np.float32)
    dK = np.zeros((num_entries, num_channels, 3), dtype=np.float32)
    dC0 = np.zeros((num_entries, num_substrates, 3), dtype=np.float32)
    dGamma = np.zeros((num_entries, num_channels, 3), dtype=np.float32)
    dDelta = np.zeros((num_entries, num_channels, num_channels, 3), dtype=np.float32)
    if substrate_mode != "material":
        dC0[rows, substrate_idx, :] = 1.0

    for layer in range(micro_layers.shape[1]):
        idx = micro_layers[:, layer]
        E_sel = E[idx]
        scale = np.exp(np.clip(gamma[idx] * h_delta, -20.0, 20.0)).astype(np.float32) if enable_height_scale else np.ones_like(h_delta, dtype=np.float32)
        k_scaled = k[idx] * scale[:, None]
        k_eff = k_scaled + delta[idx, prev_idx, :]
        T = np.exp(-np.minimum(k_eff * t, 50.0))
        one_minus_T = 1.0 - T
        c_prev = c

        dE *= T[:, None, :]; dK *= T[:, None, :]; dC0 *= T[:, None, :]
        dGamma *= T[:, None, :]; dDelta *= T[:, None, None, :]

        dE[rows, idx, :] += one_minus_T
        common = (c_prev - E_sel) * (-t * T)
        dK[rows, idx, :] += common * scale[:, None]
        if enable_height_scale:
            dGamma[rows, idx, :] += common * (k_scaled * h_delta[:, None])
        dDelta[rows, idx, prev_idx, :] += common
        c = one_minus_T * E_sel + T * c_prev
        prev_idx = idx

    return c, dE, dK, dC0, dGamma, dDelta


# ---------------------------------------------------------------------------
# Loss, evaluation, grouping
# ---------------------------------------------------------------------------

def predict_linear_entries(
    entries: Sequence[DbEntry], E: np.ndarray, k: np.ndarray,
    c0: np.ndarray, gamma: np.ndarray, delta: np.ndarray,
    config: OptimConfig, prepared: Optional[PreparedEntries] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    count = len(entries)
    if count == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    use_fast = (prepared is not None and prepared.micro_layers_matrix is not None
                and prepared.micro_layers_matrix.shape[0] == count)
    if use_fast:
        pred_linear = forward_predict_batch(
            prepared.micro_layers_matrix, prepared.base_channel_idx,
            prepared.substrate_idx, prepared.layer_height_mm,
            E, k, c0, gamma, delta, config.micro_layer_height,
            config.substrate_mode, config.height_ref_mm, config.enable_height_scale,
        )
        return pred_linear, prepared.target_lab
    pred_linear = np.zeros((count, 3), dtype=np.float32)
    target_lab = np.zeros((count, 3), dtype=np.float32)
    for idx, entry in enumerate(entries):
        pred, *_ = forward_with_grad(
            entry.micro_layers, entry.base_channel_idx, entry.substrate_idx,
            entry.layer_height_mm, E, k, c0, gamma, delta,
            config.micro_layer_height, config.substrate_mode,
            config.height_ref_mm, config.enable_height_scale,
        )
        pred_linear[idx] = pred; target_lab[idx] = entry.lab
    return pred_linear, target_lab


def compute_loss_and_grad(
    u: np.ndarray, v: np.ndarray, c0_u: np.ndarray,
    gamma: np.ndarray, delta: np.ndarray,
    entries: Sequence[DbEntry], config: OptimConfig,
    prepared: Optional[PreparedEntries] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if not entries:
        raise ValueError("entries must not be empty")
    E = sigmoid(u); k = softplus(v); c0 = sigmoid(c0_u)
    count = len(entries)
    num_channels = E.shape[0]
    use_fast = (prepared is not None and prepared.micro_layers_matrix is not None
                and prepared.micro_layers_matrix.shape[0] == count)

    if use_fast:
        pred_linear, dE_arr, dK_arr, dC0_arr, dGamma_arr, dDelta_arr = forward_with_grad_batch(
            prepared.micro_layers_matrix, prepared.base_channel_idx,
            prepared.substrate_idx, prepared.layer_height_mm,
            E, k, c0, gamma, delta, config.micro_layer_height,
            config.substrate_mode, config.height_ref_mm, config.enable_height_scale,
        )
        target_lab = prepared.target_lab
    else:
        pred_linear = np.zeros((count, 3), dtype=np.float32)
        target_lab = np.zeros((count, 3), dtype=np.float32)
        dE_arr = np.zeros((count, num_channels, 3), dtype=np.float32)
        dK_arr = np.zeros((count, num_channels, 3), dtype=np.float32)
        dC0_arr = np.zeros((count, c0.shape[0], 3), dtype=np.float32)
        dGamma_arr = np.zeros((count, num_channels, 3), dtype=np.float32)
        dDelta_arr = np.zeros((count, num_channels, num_channels, 3), dtype=np.float32)
        for idx, entry in enumerate(entries):
            pred, dE, dK, dC0_, dGamma_, dDelta_ = forward_with_grad(
                entry.micro_layers, entry.base_channel_idx, entry.substrate_idx,
                entry.layer_height_mm, E, k, c0, gamma, delta,
                config.micro_layer_height, config.substrate_mode,
                config.height_ref_mm, config.enable_height_scale,
            )
            pred_linear[idx] = pred; target_lab[idx] = entry.lab
            dE_arr[idx] = dE; dK_arr[idx] = dK; dC0_arr[idx] = dC0_
            dGamma_arr[idx] = dGamma_; dDelta_arr[idx] = dDelta_

    pred_lab = linear_rgb_to_opencv_lab_batch(pred_linear)
    diff = pred_lab - target_lab
    total_loss = float(np.mean(np.sum(diff * diff, axis=1)))
    dL_dLab = (2.0 / float(count)) * diff
    dL_dC = lab_grad_from_linear_batch(pred_linear, dL_dLab, config.lab_eps)
    grad_E = np.sum(dE_arr * dL_dC[:, None, :], axis=0)
    grad_K = np.sum(dK_arr * dL_dC[:, None, :], axis=0)
    grad_C0 = np.sum(dC0_arr * dL_dC[:, None, :], axis=0)
    grad_gamma = np.sum(dGamma_arr * dL_dC[:, None, :], axis=(0, 2))
    grad_delta = np.sum(dDelta_arr * dL_dC[:, None, None, :], axis=0)

    reg_loss = 0.0
    if config.lambda_reg > 0.0:
        diff_E = E - config.E0; diff_K = k - config.k0
        reg = config.lambda_reg * (np.mean(diff_E * diff_E) + np.mean(diff_K * diff_K))
        reg_loss += float(reg)
        grad_E += (2.0 * config.lambda_reg / diff_E.size) * diff_E
        grad_K += (2.0 * config.lambda_reg / diff_K.size) * diff_K
    if config.substrate_mode == "boundary" and config.lambda_c0 > 0.0:
        diff_C0 = c0 - config.c0_anchor
        reg = config.lambda_c0 * np.mean(diff_C0 * diff_C0)
        reg_loss += float(reg)
        grad_C0 += (2.0 * config.lambda_c0 / diff_C0.size) * diff_C0
    if config.enable_height_scale and config.lambda_height_scale > 0.0:
        reg = config.lambda_height_scale * np.mean(gamma * gamma)
        reg_loss += float(reg)
        grad_gamma += (2.0 * config.lambda_height_scale / gamma.size) * gamma
    if config.lambda_neighbor > 0.0:
        reg = config.lambda_neighbor * np.mean(delta * delta)
        reg_loss += float(reg)
        grad_delta += (2.0 * config.lambda_neighbor / delta.size) * delta

    total_loss += reg_loss
    grad_u = grad_E * (E * (1.0 - E))
    grad_v = grad_K * softplus_grad(v)
    grad_c0_u = grad_C0 * (c0 * (1.0 - c0))
    return total_loss, grad_u, grad_v, grad_c0_u, grad_gamma, grad_delta, E, k, c0, reg_loss


def adam_optimize(
    u, v, c0_u, gamma, delta, entries, config, prepared,
    steps, lr, tol, train_u, train_v, train_c0, train_gamma, train_delta, desc,
):
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    ms = {n: np.zeros_like(a) for n, a in [("u", u), ("v", v), ("c0", c0_u), ("g", gamma), ("d", delta)]}
    vs = {n: np.zeros_like(a) for n, a in [("u", u), ("v", v), ("c0", c0_u), ("g", gamma), ("d", delta)]}
    prev_loss = None; last_loss = float("inf"); last_reg = 0.0
    total_steps = max(1, int(steps))
    with tqdm(total=total_steps, desc=desc, unit="step", dynamic_ncols=True) as pbar:
        for step in range(1, total_steps + 1):
            loss, gu, gv, gc0, gg, gd, _, _, _, reg = compute_loss_and_grad(
                u, v, c0_u, gamma, delta, entries, config, prepared,
            )
            last_loss = loss; last_reg = reg
            if step == 1 or step % 10 == 0 or step == total_steps:
                pbar.set_postfix(loss=f"{last_loss:.6f}", reg=f"{last_reg:.6f}")
            if prev_loss is not None and abs(prev_loss - loss) < tol:
                pbar.update(1)
                return u, v, c0_u, gamma, delta, last_loss, last_reg, step
            prev_loss = loss
            grads = {"u": gu if train_u else np.zeros_like(gu),
                     "v": gv if train_v else np.zeros_like(gv),
                     "c0": gc0 if train_c0 else np.zeros_like(gc0),
                     "g": gg if train_gamma else np.zeros_like(gg),
                     "d": gd if train_delta else np.zeros_like(gd)}
            arrays = {"u": u, "v": v, "c0": c0_u, "g": gamma, "d": delta}
            for n in arrays:
                g = grads[n]
                ms[n] = beta1 * ms[n] + (1.0 - beta1) * g
                vs[n] = beta2 * vs[n] + (1.0 - beta2) * (g * g)
                mh = ms[n] / (1.0 - beta1 ** step)
                vh = vs[n] / (1.0 - beta2 ** step)
                arrays[n] -= lr * mh / (np.sqrt(vh) + eps)
            pbar.update(1)
    return u, v, c0_u, gamma, delta, last_loss, last_reg, total_steps


def compute_eval_stats(pred_lab: np.ndarray, target_lab: np.ndarray) -> Dict[str, Any]:
    if pred_lab.shape != target_lab.shape or pred_lab.shape[0] == 0:
        return {}
    residual = pred_lab - target_lab
    deltas_arr = np.linalg.norm(residual, axis=1).astype(np.float32)
    return {
        "count": int(pred_lab.shape[0]),
        "mean_delta_e": round(float(np.mean(deltas_arr)), 4),
        "median_delta_e": round(float(np.median(deltas_arr)), 4),
        "p90_delta_e": round(float(np.percentile(deltas_arr, 90)), 4),
        "max_delta_e": round(float(np.max(deltas_arr)), 4),
        "mean_residual_lab": round_list(np.mean(residual, axis=0), 4),
        "std_residual_lab": round_list(np.std(residual, axis=0), 4),
    }


def evaluate_entries(entries, E, k, c0, gamma, delta, config, prepared=None):
    pred_linear, target_lab = predict_linear_entries(entries, E, k, c0, gamma, delta, config, prepared)
    pred_lab = linear_rgb_to_opencv_lab_batch(pred_linear)
    return compute_eval_stats(pred_lab, target_lab)


def group_entries_by_source(entries):
    grouped = {}
    for e in entries:
        grouped.setdefault(e.source_db, []).append(e)
    return grouped


def group_entries_by_base_channel(entries):
    grouped = {}
    for e in entries:
        grouped.setdefault(int(e.base_channel_idx), []).append(e)
    return grouped


def group_entries_by_layer_height(entries):
    grouped = {}
    for e in entries:
        grouped.setdefault(round(float(e.layer_height_mm), 6), []).append(e)
    return grouped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    db_paths = resolve_db_paths(args.db)
    stage = load_stage_a_params(args.stage_a)
    enable_learn_c0 = (not args.disable_learn_c0) and args.substrate_mode == "boundary"
    enable_height_scale = not args.disable_height_scale
    micro_layer_count = args.micro_layer_count
    if micro_layer_count is not None and micro_layer_count <= 0:
        micro_layer_count = None

    # --- 第一遍: 加载所有 DB，收集 base_channel_stage_idx 用于 substrate 分组 ---
    # 不同 DB 的调色板顺序可能不同 (e.g. CMYK 的 White 在本地索引 0,
    # repre 的 white 在本地索引 6)，所以必须先做 palette 映射再分组。
    db_load_results: List[Tuple[Path, List[DbEntry], Dict[str, object], List[str], List[int]]] = []
    substrate_group_map: Dict[int, int] = {}  # base_channel_stage_idx -> substrate_idx
    for db_path in db_paths:
        # 临时加载，先用 substrate_idx=0 占位 (之后会修正)
        db_entries, summary, db_warnings, db_used = load_db_entries(
            db_path, stage, float(args.micro_layer_height), micro_layer_count, args.substrate_mode, 0,
        )
        base_stage_idx = int(summary["base_channel_stage_idx"])
        if base_stage_idx not in substrate_group_map:
            substrate_group_map[base_stage_idx] = len(substrate_group_map)
        db_load_results.append((db_path, db_entries, summary, db_warnings, db_used))

    # --- 第二遍: 用正确的 substrate_idx 重新赋值 ---
    entries: List[DbEntry] = []; db_summaries = []; warnings = []; used_channels = []
    substrate_defs: List[Dict[str, object]] = []
    seen_substrates: set[int] = set()
    for db_path, db_entries, summary, db_warnings, db_used in db_load_results:
        base_stage_idx = int(summary["base_channel_stage_idx"])
        substrate_idx = substrate_group_map[base_stage_idx]
        # 用正确的 substrate_idx 替换每个 entry
        corrected_entries = [
            DbEntry(
                recipe=e.recipe, lab=e.lab, micro_layers=e.micro_layers,
                base_channel_idx=e.base_channel_idx, source_db=e.source_db,
                substrate_idx=substrate_idx, layer_height_mm=e.layer_height_mm,
            )
            for e in db_entries
        ]
        entries.extend(corrected_entries)
        summary["substrate_idx"] = substrate_idx
        db_summaries.append(summary)
        warnings.extend(db_warnings); used_channels.extend(db_used)
        if substrate_idx not in seen_substrates:
            substrate_defs.append({
                "substrate_idx": substrate_idx,
                "source_dbs": [str(db_path)],
                "base_channel_idx": int(summary["base_channel_idx"]),
                "base_channel_stage_idx": base_stage_idx,
            })
            seen_substrates.add(substrate_idx)
        else:
            # 追加此 DB 的 source 到已有 substrate 定义
            for sd in substrate_defs:
                if int(sd["substrate_idx"]) == substrate_idx:
                    sd["source_dbs"].append(str(db_path))
                    break

    if not entries:
        raise ValueError("No valid entries for optimization")
    prepared_entries = prepare_entries(entries)
    num_substrates = len(substrate_defs)
    c0_anchor = np.zeros((num_substrates, 3), dtype=np.float32)
    for info in substrate_defs:
        sidx = int(info["substrate_idx"])
        base_stage_idx = int(info["base_channel_stage_idx"])
        color_name = normalize_label(stage.names[base_stage_idx])
        measured_key = f"base_{color_name}"
        if measured_key in stage.measured_c0:
            # 优先使用 Stage A 实测的基板颜色 (更准确)
            c0_anchor[sidx] = stage.measured_c0[measured_key]
            print(f"  C0 anchor for substrate {sidx} ({color_name}): "
                  f"using measured value {stage.measured_c0[measured_key].tolist()}")
        else:
            # 回退到 E (平衡色)
            c0_anchor[sidx] = stage.E0[base_stage_idx]
            print(f"  C0 anchor for substrate {sidx} ({color_name}): "
                  f"fallback to E = {stage.E0[base_stage_idx].tolist()}"
                  f" (no measured C0 found for '{measured_key}')")

    config = OptimConfig(
        micro_layer_height=float(args.micro_layer_height), micro_layer_count=micro_layer_count,
        substrate_mode=args.substrate_mode, lambda_reg=float(args.lambda_reg),
        lambda_c0=float(args.lambda_c0), lambda_height_scale=float(args.lambda_height_scale),
        lambda_neighbor=float(args.lambda_neighbor), lab_eps=float(args.lab_eps),
        height_ref_mm=float(args.height_ref_mm), enable_height_scale=enable_height_scale,
        E0=stage.E0, k0=stage.k0, c0_anchor=c0_anchor,
    )

    u = logit(stage.E0); v = softplus_inv(stage.k0)
    if used_channels:
        mask = np.zeros((stage.E0.shape[0], 1), dtype=np.float32)
        mask[sorted(set(used_channels)), 0] = 1.0
        u += np.random.normal(0.0, 0.01, size=u.shape).astype(np.float32) * mask
        v += np.random.normal(0.0, 0.01, size=v.shape).astype(np.float32) * mask

    c0_u = logit(c0_anchor)
    if enable_learn_c0:
        c0_u += np.random.normal(0.0, 0.005, size=c0_u.shape).astype(np.float32)
    gamma = np.zeros((stage.E0.shape[0],), dtype=np.float32)
    delta = np.zeros((stage.E0.shape[0], stage.E0.shape[0], 3), dtype=np.float32)

    total_steps_used = 0; last_loss = float("inf"); last_reg_loss = 0.0
    stage1_steps = max(0, int(args.stage1_steps))

    if stage1_steps > 0:
        u, v, c0_u, gamma, delta, last_loss, last_reg_loss, steps_used = adam_optimize(
            u, v, c0_u, gamma, delta, entries, config, prepared_entries,
            stage1_steps, args.lr, args.tol, False, False, enable_learn_c0,
            enable_height_scale, True, "Stage B fit (stage1 aux)",
        )
        total_steps_used += int(steps_used)

    u, v, c0_u, gamma, delta, last_loss, last_reg_loss, steps_used = adam_optimize(
        u, v, c0_u, gamma, delta, entries, config, prepared_entries,
        args.steps, args.lr, args.tol, True, True, enable_learn_c0,
        enable_height_scale, True, "Stage B fit (stage2 joint)",
    )
    total_steps_used += int(steps_used)

    E_fit = sigmoid(u); k_fit = softplus(v); c0_fit = sigmoid(c0_u)

    # Evaluation
    stats = evaluate_entries(entries, E_fit, k_fit, c0_fit, gamma, delta, config, prepared_entries)
    per_db_stats = {}
    for db_path in db_paths:
        source = str(db_path)
        db_entries = group_entries_by_source(entries).get(source, [])
        if db_entries:
            per_db_stats[source] = evaluate_entries(db_entries, E_fit, k_fit, c0_fit, gamma, delta, config, prepare_entries(db_entries))

    per_base_channel_stats = {}
    for base_idx, grouped in sorted(group_entries_by_base_channel(entries).items()):
        per_base_channel_stats[str(base_idx)] = {
            "base_color": stage.names[base_idx] if 0 <= base_idx < len(stage.names) else f"channel_{base_idx}",
            **evaluate_entries(grouped, E_fit, k_fit, c0_fit, gamma, delta, config, prepare_entries(grouped)),
        }

    per_layer_height_stats = {}
    for layer_h, grouped in sorted(group_entries_by_layer_height(entries).items()):
        per_layer_height_stats[f"{layer_h:.6f}"] = evaluate_entries(
            grouped, E_fit, k_fit, c0_fit, gamma, delta, config, prepare_entries(grouped),
        )

    if per_db_stats:
        print("Per-DB validation:")
        for s, st in per_db_stats.items():
            print(f"  {s}: mean_delta_e={st['mean_delta_e']:.4f}, p90={st['p90_delta_e']:.4f}, max={st['max_delta_e']:.4f}")
    if per_base_channel_stats:
        print("Per-base-channel validation:")
        for bk in sorted(per_base_channel_stats, key=int):
            st = per_base_channel_stats[bk]
            print(f"  base={bk} ({st['base_color']}): mean={st['mean_delta_e']:.4f}, p90={st['p90_delta_e']:.4f}, max={st['max_delta_e']:.4f}")

    # Output
    output_channels = []
    for idx, color_name in enumerate(stage.names):
        output_channels.append({
            "channel_index": idx, "color_name": color_name,
            "E": round_list(E_fit[idx], 6), "k": round_list(k_fit[idx], 6),
            "k_height_scale_gamma": round(float(gamma[idx]), 6),
        })

    output_substrates = []
    for info in sorted(substrate_defs, key=lambda x: int(x["substrate_idx"])):
        sidx = int(info["substrate_idx"])
        output_substrates.append({
            "substrate_idx": sidx,
            "source_dbs": info["source_dbs"],
            "base_channel_idx": int(info["base_channel_idx"]),
            "base_channel_stage_idx": int(info["base_channel_stage_idx"]),
            "base_channel_color": stage.names[int(info["base_channel_stage_idx"])],
            "C0_anchor": round_list(c0_anchor[sidx], 6),
            "C0_fitted": round_list(c0_fit[sidx], 6),
        })

    delta_summary = {
        "enabled": True,
        "l2_norm": round(float(np.linalg.norm(delta)), 6),
        "max_abs": round(float(np.max(np.abs(delta))), 6),
        "delta_k": [[round_list(delta[i, j], 6) for j in range(delta.shape[1])] for i in range(delta.shape[0])],
    }

    output = {
        "meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_dbs": [str(p) for p in db_paths],
            "input_stage_a": str(args.stage_a),
            "fitting_method": "stage_B",
            "optimizer": {"name": "adam", "lr": args.lr, "stage1_steps": stage1_steps,
                         "stage2_steps": int(args.steps), "tol": args.tol},
            "loss_space": "opencv_lab",
            "micro_layer_height_mm": config.micro_layer_height,
            "height_ref_mm": config.height_ref_mm,
            "enable_height_scale": bool(enable_height_scale),
            "enable_learn_c0": bool(enable_learn_c0),
            "final_loss": round(float(last_loss), 8),
            "final_reg_loss": round(float(last_reg_loss), 8),
            "fit_steps": int(total_steps_used),
            "db_summaries": db_summaries,
            "warnings": warnings,
        },
        "parameters": {
            "channels": output_channels,
            "substrates": output_substrates,
            "neighbor_delta": delta_summary,
        },
        "validation": {
            "overall": stats,
            "per_db": per_db_stats,
            "per_base_channel": per_base_channel_stats,
            "per_layer_height_mm": per_layer_height_stats,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Saved Stage B parameters to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
