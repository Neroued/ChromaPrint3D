#!/usr/bin/env python3
"""
Build v2 model package (MessagePack binary format) for runtime model fallback.

Example:
  python -m modeling.pipeline.step5_build_model_package \
    --stage modeling/output/params/stage_B_retrained.json \
    --db modeling/dbs \
    --output modeling/output/packages/pla_basic.msgpack
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import msgpack
import numpy as np

from modeling.core.color_space import linear_rgb_to_opencv_lab_batch
from modeling.core.forward_model import (
    load_stage_forward_model,
    predict_linear_batch as predict_with_stage_model,
    resolve_substrate_idx,
    substrate_source_for_idx,
)
from modeling.core.io_utils import load_json, normalize_label, parse_layer_order


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODES = "2:0.08,3:0.08,4:0.08,5:0.08,6:0.04,7:0.04,8:0.04,9:0.04,10:0.04"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v2 model package (msgpack).")
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--modes", type=str, default=DEFAULT_MODES,
                        help="Comma-separated layer configs: '5:0.08,10:0.04'.")
    parser.add_argument("--candidate-count", type=int, default=65536)
    parser.add_argument("--material", type=str, default="PLA Basic")
    parser.add_argument("--layer-order", choices=("Top2Bottom", "Bottom2Top"), default="Top2Bottom")
    parser.add_argument("--base-channel-key", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--margin", type=float, default=0.7)
    parser.add_argument("--micro-layer-height", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vendor", type=str, default="BambuLab")
    parser.add_argument("--material-type", type=str, default="PLA")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "modeling" / "output" / "packages" / "pla_basic.msgpack")
    return parser.parse_args()


def normalize_channel_key(value: str) -> str:
    if "|" not in value:
        return normalize_label(value)
    color, material = value.split("|", 1)
    return f"{normalize_label(color)}|{normalize_label(material)}"


def build_channel_key(color: str, material: str) -> str:
    return normalize_channel_key(f"{color}|{material}")


def parse_modes(modes_text: str) -> List[Tuple[float, int]]:
    """Parse mode specifications: '5:0.08,10:0.04' (layers:height)."""
    out: List[Tuple[float, int]] = []
    for raw in modes_text.split(","):
        mode = raw.strip()
        if not mode:
            continue
        if ":" in mode:
            parts = mode.split(":")
            layers = int(parts[0])
            height = float(parts[1])
            out.append((height, layers))
        else:
            raise ValueError(f"Unsupported mode format: {mode} (use 'layers:height')")
    if not out:
        raise ValueError("No valid modes")
    return out


def resolve_db_paths(db_path: Optional[Path]) -> List[Path]:
    if db_path is None:
        return []
    if db_path.is_file():
        return [db_path]
    if db_path.is_dir():
        return sorted([p for p in db_path.iterdir() if p.is_file() and p.suffix == ".json"])
    raise FileNotFoundError(f"Invalid db path: {db_path}")


def _pad_to_target(
    layers: List[int], target: int, base_ch: int, layer_order: str,
) -> Optional[List[int]]:
    if len(layers) > target:
        return None
    if len(layers) == target:
        return layers
    pad_count = target - len(layers)
    if layer_order == "Top2Bottom":
        return layers + [base_ch] * pad_count
    return [base_ch] * pad_count + layers


def expand_recipe_to_mode(
    recipe: Sequence[int], db_layer_height: float, db_layer_order: str,
    target_layer_height: float, target_layers: int, target_layer_order: str,
    base_channel_idx: int = 0,
) -> Optional[List[int]]:
    recipe_list = list(recipe)
    if db_layer_order != target_layer_order:
        recipe_list = list(reversed(recipe_list))
    eps = 1e-3
    if abs(db_layer_height - target_layer_height) <= eps:
        return _pad_to_target(recipe_list, target_layers, base_channel_idx, target_layer_order)
    if db_layer_height > target_layer_height:
        ratio_f = db_layer_height / target_layer_height
        ratio = int(round(ratio_f))
        if ratio <= 0 or abs(ratio_f - ratio) > eps:
            return None
        expanded: List[int] = []
        for ch in recipe_list:
            expanded.extend([int(ch)] * ratio)
        return _pad_to_target(expanded, target_layers, base_channel_idx, target_layer_order)
    ratio_f = target_layer_height / db_layer_height
    ratio = int(round(ratio_f))
    if ratio <= 0 or abs(ratio_f - ratio) > eps or len(recipe_list) % ratio != 0:
        return None
    merged: List[int] = []
    merged_count = len(recipe_list) // ratio
    for i in range(merged_count):
        begin = i * ratio
        ref = int(recipe_list[begin])
        if any(int(recipe_list[begin + j]) != ref for j in range(1, ratio)):
            return None
        merged.append(ref)
    return _pad_to_target(merged, target_layers, base_channel_idx, target_layer_order)


def collect_seed_recipes(
    db_paths: Sequence[Path], stage_key_to_idx: Dict[str, int],
    target_layer_height: float, target_layers: int, target_layer_order: str,
    base_channel_idx: int = 0,
) -> Set[Tuple[int, ...]]:
    recipes: Set[Tuple[int, ...]] = set()
    for path in db_paths:
        data = load_json(path)
        entries = data.get("entries", [])
        palette = data.get("palette", [])
        if not isinstance(entries, list) or not isinstance(palette, list) or not palette:
            continue
        db_to_stage: List[int] = []
        valid_db = True
        for p in palette:
            key = build_channel_key(str(p.get("color", "")), str(p.get("material", "")))
            if key not in stage_key_to_idx:
                valid_db = False; break
            db_to_stage.append(stage_key_to_idx[key])
        if not valid_db:
            continue
        db_layer_height = float(data.get("layer_height_mm", target_layer_height))
        db_layer_order = parse_layer_order(data.get("layer_order", "Top2Bottom"))
        for entry in entries:
            raw = entry.get("recipe")
            if not isinstance(raw, list):
                continue
            try:
                stage_recipe = [db_to_stage[int(v)] for v in raw]
            except Exception:
                continue
            converted = expand_recipe_to_mode(
                stage_recipe, db_layer_height, db_layer_order,
                target_layer_height, target_layers, target_layer_order,
                base_channel_idx,
            )
            if converted is not None:
                recipes.add(tuple(converted))
    return recipes


def sample_candidate_recipes(
    seed_recipes: Set[Tuple[int, ...]], num_channels: int,
    color_layers: int, target_count: int, seed: int,
) -> np.ndarray:
    total_space = int(num_channels ** color_layers)
    desired_count = min(target_count, total_space)
    rng = np.random.default_rng(seed)
    recipes = list(seed_recipes)
    if len(recipes) > desired_count:
        choice = rng.choice(len(recipes), size=desired_count, replace=False)
        return np.asarray([recipes[int(i)] for i in choice], dtype=np.int32)
    seen = set(seed_recipes)
    max_attempts = max(desired_count * 50, 1000)
    for _ in range(max_attempts):
        sample = tuple(int(v) for v in rng.integers(0, num_channels, size=color_layers))
        if sample not in seen:
            seen.add(sample); recipes.append(sample)
        if len(recipes) >= desired_count:
            break
    if len(recipes) < desired_count and total_space <= 2_000_000:
        for idx in range(total_space):
            raw = idx; sample_list = [0] * color_layers
            for layer in range(color_layers):
                sample_list[layer] = int(raw % num_channels); raw //= num_channels
            sample = tuple(sample_list)
            if sample not in seen:
                seen.add(sample); recipes.append(sample)
            if len(recipes) >= desired_count:
                break
    return np.asarray(recipes, dtype=np.int32)


def main() -> int:
    args = parse_args()
    modes = parse_modes(args.modes)
    stage_model = load_stage_forward_model(args.stage)
    channel_keys = [build_channel_key(name, args.material) for name in stage_model.channel_names]
    stage_key_to_idx = {key: idx for idx, key in enumerate(channel_keys)}
    micro_h = float(args.micro_layer_height) if args.micro_layer_height is not None else float(stage_model.micro_layer_height_mm)

    base_channel_key = (
        normalize_channel_key(args.base_channel_key) if args.base_channel_key is not None
        else next((k for k in channel_keys if k.startswith("white|")), channel_keys[0])
    )
    if base_channel_key not in stage_key_to_idx:
        raise ValueError(f"base_channel_key not found: {base_channel_key}")
    base_channel_idx = stage_key_to_idx[base_channel_key]

    db_paths = resolve_db_paths(args.db)
    substrate_idx = resolve_substrate_idx(stage_model, db_paths, base_channel_idx)
    substrate_source = substrate_source_for_idx(stage_model, substrate_idx)

    pack_name = args.name or f"{args.vendor}_{args.material_type}_ModelPackage"
    modes_list: List[dict] = []

    for layer_height_mm, color_layers in modes:
        seed_recipes = collect_seed_recipes(
            db_paths, stage_key_to_idx, layer_height_mm, color_layers, args.layer_order,
            base_channel_idx,
        )
        recipes = sample_candidate_recipes(
            seed_recipes, stage_model.E.shape[0], color_layers, args.candidate_count,
            args.seed + color_layers,
        )
        linear_rgb = predict_with_stage_model(
            recipes, stage_model, layer_height_mm, micro_h, base_channel_idx,
            args.layer_order, substrate_idx,
        )
        pred_lab = linear_rgb_to_opencv_lab_batch(linear_rgb)

        recipes_bin = recipes.astype(np.uint8).tobytes()
        pred_lab_bin = pred_lab.astype("<f4").tobytes()

        modes_list.append({
            "color_layers": color_layers,
            "layer_height_mm": layer_height_mm,
            "layer_order": args.layer_order,
            "base_channel_key": base_channel_key,
            "num_candidates": int(recipes.shape[0]),
            "candidate_recipes": recipes_bin,
            "pred_lab": pred_lab_bin,
        })
        print(f"[{color_layers}L/{layer_height_mm}mm] "
              f"db_seed={len(seed_recipes)} candidates={recipes.shape[0]}")

    output = {
        "schema_version": 2,
        "name": pack_name,
        "scope": {
            "vendor": args.vendor,
            "material_type": args.material_type,
        },
        "channel_keys": channel_keys,
        "defaults": {"threshold": float(args.threshold), "margin": float(args.margin)},
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_from": str(args.stage),
            "dbs": [str(p) for p in db_paths],
            "seed": args.seed,
            "candidate_count_per_mode": args.candidate_count,
            "micro_layer_height_mm": micro_h,
            "layer_order": args.layer_order,
            "substrate_idx": int(substrate_idx),
            "substrate_source_db": substrate_source,
        },
        "modes": modes_list,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    packed = msgpack.packb(output, use_bin_type=True)
    args.output.write_bytes(packed)
    print(f"Saved v2 model package ({len(packed)} bytes) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
