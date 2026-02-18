#!/usr/bin/env python3
"""
Select representative recipes using k-center coverage in OpenCV Lab space.

Example:
  python -m modeling.pipeline.step4_select_recipes \
    --stage-b modeling/output/params/stage_B_retrained.json \
    --db modeling/dbs \
    --mode 0.04x10 \
    --output modeling/output/recipes/recipes_0p04_10L.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from modeling.core.color_space import linear_rgb_to_opencv_lab_batch
from modeling.core.forward_model import (
    load_stage_forward_model,
    predict_linear_batch as predict_with_stage_model,
    resolve_substrate_idx,
    substrate_source_for_idx,
)
from modeling.core.io_utils import load_json, normalize_label, parse_layer_order, resolve_db_paths


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE_B = REPO_ROOT / "modeling" / "output" / "params" / "stage_B_retrained.json"
DEFAULT_OUTPUT = REPO_ROOT / "modeling" / "output" / "recipes" / "representative_recipes.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select representative recipes with k-center coverage."
    )
    parser.add_argument("--stage-b", type=Path, default=DEFAULT_STAGE_B)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--mode", choices=("0.04x10", "0.08x5"), default="0.04x10")
    parser.add_argument("--layer-height-mm", type=float, default=None)
    parser.add_argument("--color-layers", type=int, default=None)
    parser.add_argument("--micro-layer-height", type=float, default=0.04)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--prefilter-size", type=int, default=50000)
    parser.add_argument("--max-enumerate", type=int, default=2000000)
    parser.add_argument("--layer-order", choices=("Top2Bottom", "Bottom2Top"), default=None)
    parser.add_argument("--base-channel-idx", type=int, default=None)
    parser.add_argument("--palette", type=Path, default=None)
    parser.add_argument("--material", type=str, default="PLA Basic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sort-output-by", choices=("none", "lab", "linear_rgb"), default="lab")
    parser.add_argument("--sort-output-order", choices=("asc", "desc"), default="asc")
    return parser.parse_args()


def find_white_channel_idx(palette: Sequence[Dict[str, str]]) -> Optional[int]:
    for idx, item in enumerate(palette):
        if normalize_label(str(item.get("color", ""))) == "white":
            return idx
    return None


def load_palette_override(path: Optional[Path]) -> Optional[List[Dict[str, str]]]:
    if path is None:
        return None
    data = load_json(path)
    palette_data = data.get("palette") if isinstance(data, dict) else data
    if not isinstance(palette_data, list) or not palette_data:
        raise ValueError("palette file must be an array or {palette: [...]} object")
    palette: List[Dict[str, str]] = []
    for item in palette_data:
        color = str(item.get("color", ""))
        material = str(item.get("material", ""))
        if not color:
            raise ValueError("palette item missing color")
        palette.append({"color": color, "material": material})
    return palette


def collect_existing_recipes(
    db_paths: Sequence[Path], target_layers: int, target_layer_height_mm: float,
    desired_layer_order: Optional[str], stageb_color_map: Dict[str, int],
) -> Tuple[set[Tuple[int, ...]], Optional[str], Optional[int], Optional[float], Optional[int]]:
    recipes: set[Tuple[int, ...]] = set()
    layer_order: Optional[str] = None
    base_channel_idx: Optional[int] = None
    line_width_mm: Optional[float] = None
    base_layers: Optional[int] = None
    for path in db_paths:
        data = load_json(path)
        entries = data.get("entries", [])
        if not isinstance(entries, list):
            continue
        db_layer_order = parse_layer_order(data.get("layer_order", "Top2Bottom"))
        if layer_order is None:
            layer_order = db_layer_order
        db_base_channel_idx = int(data.get("base_channel_idx", 0))
        if base_layers is None and "base_layers" in data:
            base_layers = int(data.get("base_layers", 0))
        if line_width_mm is None and "line_width_mm" in data:
            line_width_mm = float(data.get("line_width_mm", 0.0))
        raw_palette = data.get("palette", [])
        if not isinstance(raw_palette, list) or not raw_palette:
            raise ValueError(f"{path}: palette missing or empty")
        db_to_stage: List[int] = []
        for item in raw_palette:
            color = normalize_label(str(item.get("color", "")))
            if color not in stageb_color_map:
                raise ValueError(f"{path}: palette color '{color}' not in Stage B colors")
            db_to_stage.append(stageb_color_map[color])
        if db_base_channel_idx < 0 or db_base_channel_idx >= len(db_to_stage):
            raise ValueError(f"{path}: base_channel_idx out of range")
        if base_channel_idx is None:
            base_channel_idx = db_to_stage[db_base_channel_idx]
        target_order = desired_layer_order or layer_order or db_layer_order
        db_layer_height = float(data.get("layer_height_mm", target_layer_height_mm))
        ratio = db_layer_height / float(target_layer_height_mm)
        ratio_int = int(round(ratio)) if ratio > 0 else 0
        ratio_ok = ratio_int >= 1 and abs(ratio - ratio_int) <= 1e-3
        for entry in entries:
            recipe = entry.get("recipe")
            if not isinstance(recipe, list):
                continue
            recipe_list = [db_to_stage[int(v)] for v in recipe]
            if db_layer_order != target_order:
                recipe_list = list(reversed(recipe_list))
            if len(recipe_list) == target_layers:
                recipes.add(tuple(recipe_list))
            elif ratio_ok and len(recipe_list) * ratio_int == target_layers:
                expanded: List[int] = []
                for v in recipe_list:
                    expanded.extend([v] * ratio_int)
                recipes.add(tuple(expanded))
    return recipes, layer_order, base_channel_idx, line_width_mm, base_layers


def index_to_recipe(index: int, num_channels: int, color_layers: int, layer_order: str) -> List[int]:
    recipe = [0] * color_layers
    idx = index
    for i in range(color_layers):
        v = idx % num_channels; idx //= num_channels
        layer = color_layers - 1 - i if layer_order == "Top2Bottom" else i
        recipe[layer] = int(v)
    return recipe


def generate_candidate_recipes(
    num_channels: int, color_layers: int, layer_order: str,
    existing: set[Tuple[int, ...]], prefilter_size: int, max_enumerate: int, seed: int,
) -> np.ndarray:
    total = int(num_channels ** color_layers)
    available = total - len(existing)
    if available <= 0:
        raise ValueError("No candidate recipes after excluding existing colorDB recipes.")
    prefilter_size = min(prefilter_size, available)
    if total <= max_enumerate:
        candidates: List[List[int]] = []
        for idx in range(total):
            recipe = index_to_recipe(idx, num_channels, color_layers, layer_order)
            if tuple(recipe) not in existing:
                candidates.append(recipe)
        if len(candidates) > prefilter_size:
            rng = np.random.default_rng(seed)
            choice = rng.choice(len(candidates), size=prefilter_size, replace=False)
            candidates = [candidates[int(i)] for i in choice]
        return np.asarray(candidates, dtype=np.int32)
    rng = np.random.default_rng(seed)
    candidates = []
    seen: set[Tuple[int, ...]] = set()
    max_attempts = prefilter_size * 50
    for _ in range(max_attempts):
        recipe = tuple(int(v) for v in rng.integers(0, num_channels, size=color_layers))
        if recipe in existing or recipe in seen:
            continue
        seen.add(recipe)
        candidates.append(list(recipe))
        if len(candidates) >= prefilter_size:
            break
    return np.asarray(candidates, dtype=np.int32)


def kcenter_select(labs: np.ndarray, k: int, seed: int) -> List[int]:
    if labs.shape[0] < k:
        raise ValueError("Not enough candidates to select.")
    mean = np.mean(labs, axis=0)
    dist = np.sum((labs - mean) ** 2, axis=1)
    first = int(np.argmax(dist))
    selected = [first]
    min_dist = np.sum((labs - labs[first]) ** 2, axis=1)
    for _ in range(1, k):
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        new_dist = np.sum((labs - labs[idx]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, new_dist)
    return selected


def sort_selected_outputs(recipes, linear, labs, sort_by, sort_order):
    if sort_by == "none":
        return recipes, linear, labs
    values = labs if sort_by == "lab" else linear
    order = np.lexsort((values[:, 2], values[:, 1], values[:, 0]))
    if sort_order == "desc":
        order = order[::-1]
    return recipes[order], linear[order], labs[order]


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)

    default_layers, default_height = (10, 0.04) if args.mode == "0.04x10" else (5, 0.08)
    color_layers = args.color_layers if args.color_layers is not None else default_layers
    layer_height_mm = args.layer_height_mm if args.layer_height_mm is not None else default_height
    if args.prefilter_size < args.count:
        raise ValueError("--prefilter-size must be >= --count")

    stage_model = load_stage_forward_model(args.stage_b)
    stageb_names = stage_model.channel_names
    num_channels = stage_model.E.shape[0]
    palette_override = load_palette_override(args.palette)
    palette = palette_override if palette_override is not None else [
        {"color": name, "material": args.material} for name in stageb_names
    ]
    if palette_override and len(palette_override) != num_channels:
        raise ValueError("palette length must match Stage B channels")

    stageb_color_map = {normalize_label(item["color"]): idx for idx, item in enumerate(palette)}
    db_paths = resolve_db_paths(args.db)
    existing, db_layer_order, db_base_idx, line_width_mm, base_layers = collect_existing_recipes(
        db_paths, color_layers, layer_height_mm, args.layer_order, stageb_color_map,
    )
    layer_order = args.layer_order or db_layer_order or "Top2Bottom"
    white_idx = find_white_channel_idx(palette)
    if args.base_channel_idx is not None:
        base_channel_idx = args.base_channel_idx
    elif white_idx is not None:
        base_channel_idx = white_idx
    elif db_base_idx is not None:
        base_channel_idx = db_base_idx
    else:
        base_channel_idx = 0

    substrate_idx = resolve_substrate_idx(stage_model, db_paths, base_channel_idx)
    substrate_source = substrate_source_for_idx(stage_model, substrate_idx)

    candidates = generate_candidate_recipes(
        num_channels, color_layers, layer_order, existing,
        args.prefilter_size, args.max_enumerate, args.seed,
    )
    print(f"Candidates: {candidates.shape[0]}")

    linear = predict_with_stage_model(
        candidates, stage_model, layer_height_mm, float(args.micro_layer_height),
        base_channel_idx, layer_order, substrate_idx,
    )
    labs = linear_rgb_to_opencv_lab_batch(linear)
    selected_idx = kcenter_select(labs, args.count, args.seed)
    si = np.asarray(selected_idx, dtype=np.int64)
    sr, sl, slab = sort_selected_outputs(
        candidates[si], linear[si], labs[si], args.sort_output_by, args.sort_output_order,
    )

    output = {
        "meta": {
            "generated_from": str(args.stage_b), "dbs": [str(p) for p in db_paths],
            "mode": args.mode, "layer_height_mm": layer_height_mm,
            "color_layers": color_layers, "num_channels": num_channels,
            "layer_order": layer_order, "base_layers": base_layers,
            "base_channel_idx": base_channel_idx,
            "substrate_idx": int(substrate_idx), "substrate_source_db": substrate_source,
            "line_width_mm": line_width_mm,
            "existing_recipe_count": len(existing),
        },
        "base_layers": base_layers, "base_channel_idx": base_channel_idx,
        "layer_height_mm": layer_height_mm, "line_width_mm": line_width_mm,
        "layer_order": layer_order, "palette": palette,
        "recipes": sr.tolist(),
        "predicted_linear_rgb": sl.astype(np.float64).tolist(),
        "predicted_lab_opencv": slab.astype(np.float64).tolist(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Saved {len(sr)} recipes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
