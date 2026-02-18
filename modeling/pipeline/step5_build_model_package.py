#!/usr/bin/env python3
"""
Build Phase A model package for runtime model fallback.

Example:
  python -m modeling.pipeline.step5_build_model_package \
    --stage modeling/output/params/stage_B_retrained.json \
    --db modeling/dbs \
    --output modeling/output/packages/model_package_phaseA.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase A model package json.")
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--modes", type=str, default="0.08x5,0.04x10")
    parser.add_argument("--candidate-count", type=int, default=65536)
    parser.add_argument("--material", type=str, default="PLA Basic")
    parser.add_argument("--layer-order", choices=("Top2Bottom", "Bottom2Top"), default="Top2Bottom")
    parser.add_argument("--base-channel-key", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--margin", type=float, default=0.7)
    parser.add_argument("--micro-layer-height", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "modeling" / "output" / "packages" / "model_package_phaseA.json")
    return parser.parse_args()


def normalize_channel_key(value: str) -> str:
    if "|" not in value:
        return normalize_label(value)
    color, material = value.split("|", 1)
    return f"{normalize_label(color)}|{normalize_label(material)}"


def build_channel_key(color: str, material: str) -> str:
    return normalize_channel_key(f"{color}|{material}")


def parse_modes(modes_text: str) -> List[Tuple[str, float, int]]:
    out: List[Tuple[str, float, int]] = []
    for raw in modes_text.split(","):
        mode = raw.strip()
        if mode in ("0.08x5", "0p08x5"):
            out.append(("0.08x5", 0.08, 5))
        elif mode in ("0.04x10", "0p04x10"):
            out.append(("0.04x10", 0.04, 10))
        elif mode:
            raise ValueError(f"Unsupported mode: {mode}")
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


def expand_recipe_to_mode(
    recipe: Sequence[int], db_layer_height: float, db_layer_order: str,
    target_layer_height: float, target_layers: int, target_layer_order: str,
) -> Optional[List[int]]:
    recipe_list = list(recipe)
    if db_layer_order != target_layer_order:
        recipe_list = list(reversed(recipe_list))
    eps = 1e-3
    if abs(db_layer_height - target_layer_height) <= eps:
        return recipe_list if len(recipe_list) == target_layers else None
    if db_layer_height > target_layer_height:
        ratio_f = db_layer_height / target_layer_height
        ratio = int(round(ratio_f))
        if ratio <= 0 or abs(ratio_f - ratio) > eps or len(recipe_list) * ratio != target_layers:
            return None
        out: List[int] = []
        for ch in recipe_list:
            out.extend([int(ch)] * ratio)
        return out
    ratio_f = target_layer_height / db_layer_height
    ratio = int(round(ratio_f))
    if ratio <= 0 or abs(ratio_f - ratio) > eps or len(recipe_list) != target_layers * ratio:
        return None
    out = []
    for i in range(target_layers):
        begin = i * ratio; ref = int(recipe_list[begin])
        if any(int(recipe_list[begin + j]) != ref for j in range(1, ratio)):
            return None
        out.append(ref)
    return out


def collect_seed_recipes(
    db_paths: Sequence[Path], stage_key_to_idx: Dict[str, int],
    target_layer_height: float, target_layers: int, target_layer_order: str,
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
    output_modes: Dict[str, Dict[str, object]] = {}

    for mode_name, layer_height_mm, color_layers in modes:
        seed_recipes = collect_seed_recipes(
            db_paths, stage_key_to_idx, layer_height_mm, color_layers, args.layer_order,
        )
        recipes = sample_candidate_recipes(
            seed_recipes, stage_model.E.shape[0], color_layers, args.candidate_count, args.seed + color_layers,
        )
        linear_rgb = predict_with_stage_model(
            recipes, stage_model, layer_height_mm, micro_h, base_channel_idx, args.layer_order, substrate_idx,
        )
        pred_lab = linear_rgb_to_opencv_lab_batch(linear_rgb)
        output_modes[mode_name] = {
            "layer_height_mm": layer_height_mm, "color_layers": color_layers,
            "layer_order": args.layer_order, "base_channel_key": base_channel_key,
            "substrate_idx": int(substrate_idx), "substrate_source_db": substrate_source,
            "candidate_recipes": recipes.astype(np.int32).tolist(),
            "pred_lab": pred_lab.astype(np.float32).tolist(),
        }
        print(f"[{mode_name}] db_seed={len(seed_recipes)} candidates={recipes.shape[0]}")

    output = {
        "name": "PhaseA_ModelPackage",
        "meta": {
            "schema_version": "1.0",
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
        "channel_keys": channel_keys,
        "defaults": {"threshold": float(args.threshold), "margin": float(args.margin)},
        "modes": output_modes,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Saved model package to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
