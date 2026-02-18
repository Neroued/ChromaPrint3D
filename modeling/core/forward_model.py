#!/usr/bin/env python3
"""
Shared forward model utilities for Stage-B style parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from modeling.core.io_utils import load_json


def _normalize_source(value: str) -> str:
    return str(value).replace("\\", "/").strip().lower()


@dataclass(frozen=True)
class StageForwardModel:
    E: np.ndarray
    k: np.ndarray
    gamma: np.ndarray
    delta: np.ndarray
    channel_names: List[str]
    micro_layer_height_mm: float
    height_ref_mm: float
    enable_height_scale: bool
    c0_boundary: np.ndarray
    substrate_sources: List[str]
    substrate_base_channel_idx: List[int]


def _parse_channels(data: Dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    channels = data.get("parameters", {}).get("channels", [])
    if not isinstance(channels, list) or not channels:
        raise ValueError("Stage file missing parameters.channels")

    sorted_channels = sorted(channels, key=lambda item: int(item.get("channel_index", 0)))
    E_list: List[np.ndarray] = []
    k_list: List[np.ndarray] = []
    gamma_list: List[float] = []
    names: List[str] = []
    for idx, ch in enumerate(sorted_channels):
        E = np.asarray(ch.get("E", [0.5, 0.5, 0.5]), dtype=np.float32)
        k = np.asarray(ch.get("k", [1.0, 1.0, 1.0]), dtype=np.float32)
        gamma = float(ch.get("k_height_scale_gamma", 0.0))
        if E.shape != (3,) or k.shape != (3,):
            raise ValueError("Invalid stage E/k shape")
        E_list.append(E)
        k_list.append(k)
        gamma_list.append(gamma)
        names.append(str(ch.get("color_name", f"channel_{idx}")))
    return (
        np.stack(E_list, axis=0),
        np.stack(k_list, axis=0),
        np.asarray(gamma_list, dtype=np.float32),
        names,
    )


def _parse_neighbor_delta(data: Dict[str, object], num_channels: int) -> np.ndarray:
    params = data.get("parameters", {})
    neighbor = params.get("neighbor_delta", {}) if isinstance(params, dict) else {}
    raw_delta = neighbor.get("delta_k", []) if isinstance(neighbor, dict) else []
    if isinstance(raw_delta, list) and raw_delta:
        delta = np.asarray(raw_delta, dtype=np.float32)
        expected_shape = (num_channels, num_channels, 3)
        if delta.shape != expected_shape:
            raise ValueError(
                f"Invalid neighbor delta shape: {delta.shape}, expected {expected_shape}"
            )
        return delta
    return np.zeros((num_channels, num_channels, 3), dtype=np.float32)


def _parse_substrates(data: Dict[str, object], E: np.ndarray) -> tuple[np.ndarray, List[str], List[int]]:
    params = data.get("parameters", {})
    raw_substrates = params.get("substrates", []) if isinstance(params, dict) else []
    if not isinstance(raw_substrates, list) or not raw_substrates:
        # Fallback for legacy stage files.
        return np.asarray([E[0]], dtype=np.float32), [""], [0]

    max_idx = max(int(item.get("substrate_idx", 0)) for item in raw_substrates)
    c0 = np.zeros((max_idx + 1, 3), dtype=np.float32)
    sources: List[str] = [""] * (max_idx + 1)
    base_channels: List[int] = [0] * (max_idx + 1)
    for item in raw_substrates:
        idx = int(item.get("substrate_idx", 0))
        c0_item = np.asarray(item.get("C0_fitted", [0.0, 0.0, 0.0]), dtype=np.float32)
        if c0_item.shape != (3,):
            raise ValueError(f"Invalid substrate C0 shape at substrate_idx={idx}")
        c0[idx] = c0_item
        sources[idx] = str(item.get("source_db", ""))
        base_channels[idx] = int(item.get("base_channel_stage_idx", 0))
    return c0, sources, base_channels


def load_stage_forward_model(path: Path) -> StageForwardModel:
    data = load_json(path)
    E, k, gamma, names = _parse_channels(data)
    delta = _parse_neighbor_delta(data, E.shape[0])
    c0, substrate_sources, substrate_base_channels = _parse_substrates(data, E)

    meta = data.get("meta", {})
    micro_h = float(meta.get("micro_layer_height_mm", 0.04)) if isinstance(meta, dict) else 0.04
    height_ref = float(meta.get("height_ref_mm", 0.04)) if isinstance(meta, dict) else 0.04
    if not np.isfinite(micro_h) or micro_h <= 0:
        micro_h = 0.04
    if not np.isfinite(height_ref) or height_ref <= 0:
        height_ref = 0.04
    enable_height_scale = True
    if isinstance(meta, dict) and "enable_height_scale" in meta:
        enable_height_scale = bool(meta.get("enable_height_scale"))

    return StageForwardModel(
        E=E,
        k=k,
        gamma=gamma,
        delta=delta,
        channel_names=names,
        micro_layer_height_mm=micro_h,
        height_ref_mm=height_ref,
        enable_height_scale=enable_height_scale,
        c0_boundary=c0,
        substrate_sources=substrate_sources,
        substrate_base_channel_idx=substrate_base_channels,
    )


def resolve_substrate_idx(
    model: StageForwardModel,
    db_paths: Sequence[Path],
    fallback_base_channel_idx: int,
) -> int:
    source_map: Dict[int, str] = {
        idx: _normalize_source(src) for idx, src in enumerate(model.substrate_sources) if src
    }
    for db_path in db_paths:
        db_norm = _normalize_source(str(db_path))
        db_name = db_path.name.lower()
        for idx, src_norm in source_map.items():
            if db_norm == src_norm or db_norm.endswith("/" + src_norm) or src_norm.endswith("/" + db_norm):
                return idx
            if Path(src_norm).name.lower() == db_name:
                return idx

    for idx, base_idx in enumerate(model.substrate_base_channel_idx):
        if int(base_idx) == int(fallback_base_channel_idx):
            return idx
    return 0


def substrate_source_for_idx(model: StageForwardModel, substrate_idx: int) -> str:
    if 0 <= substrate_idx < len(model.substrate_sources):
        return model.substrate_sources[substrate_idx]
    return ""


def predict_linear_batch(
    recipes: np.ndarray,
    model: StageForwardModel,
    layer_height_mm: float,
    micro_layer_height: float,
    base_channel_idx: int,
    layer_order: str,
    substrate_idx: int,
) -> np.ndarray:
    """Predict linear RGB for a batch of recipes (boundary mode).

    Each recipe layer is expanded to *n_u* micro-layers (layer_height /
    micro_layer_height) so that the inference model exactly matches the
    micro-layer forward model used during Stage B training.
    """
    if recipes.ndim != 2:
        raise ValueError("recipes must be shape (N, L)")
    if base_channel_idx < 0 or base_channel_idx >= model.E.shape[0]:
        raise ValueError("base_channel_idx out of range")

    ratio_f = float(layer_height_mm) / float(micro_layer_height)
    n_u = int(round(ratio_f))
    if n_u <= 0 or abs(ratio_f - n_u) > 1e-3:
        raise ValueError("layer_height is not compatible with micro_layer_height")

    if layer_order == "Top2Bottom":
        recipe_bt = recipes[:, ::-1]
    elif layer_order == "Bottom2Top":
        recipe_bt = recipes
    else:
        raise ValueError(f"Unsupported layer_order: {layer_order}")

    # Expand each color layer to n_u micro-layers for consistency with
    # the training forward model.  E.g. 0.08mm layer → 2 × 0.04mm micro.
    if n_u > 1:
        recipe_bt = np.repeat(recipe_bt, n_u, axis=1)

    num_samples = recipe_bt.shape[0]
    if 0 <= substrate_idx < model.c0_boundary.shape[0]:
        C = np.tile(model.c0_boundary[substrate_idx], (num_samples, 1)).astype(np.float32)
    else:
        C = np.tile(model.E[base_channel_idx], (num_samples, 1)).astype(np.float32)

    prev_idx = np.full((num_samples,), int(base_channel_idx), dtype=np.int32)
    height_ref = max(float(model.height_ref_mm), 1e-6)
    h_delta = float(layer_height_mm) / height_ref - 1.0
    t = float(micro_layer_height)

    for layer in range(recipe_bt.shape[1]):
        ch = recipe_bt[:, layer]
        Ei = model.E[ch]
        if model.enable_height_scale:
            scale = np.exp(np.clip(model.gamma[ch] * h_delta, -20.0, 20.0)).astype(np.float32)
        else:
            scale = np.ones((num_samples,), dtype=np.float32)
        k_scaled = model.k[ch] * scale[:, None]
        k_eff = k_scaled + model.delta[ch, prev_idx, :]
        T = np.exp(-np.minimum(k_eff * t, 50.0))
        C = (1.0 - T) * Ei + T * C
        prev_idx = ch
    return C
