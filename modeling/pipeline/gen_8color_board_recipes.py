#!/usr/bin/env python3
"""
Generate pre-computed recipe selections for two 8-color calibration boards.

Each board is a 40x40 grid (1600 patches), for a total of 3200 recipes
selected from 8^5 = 32768 possible combinations.  The k-center greedy
algorithm in Lab space guarantees that Board 1 (the first 1600 picks)
covers the gamut broadly, while Board 2 (picks 1601-3200) fills the
remaining gaps.

Example:
  python -m modeling.pipeline.gen_8color_board_recipes \
    --stage-b modeling/output/params/stage_B_retrained.json \
    --db data/dbs \
    --output data/recipes/8color_boards.json
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
)
from modeling.core.io_utils import load_json, normalize_label, parse_layer_order, resolve_db_paths


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE_B = REPO_ROOT / "modeling" / "output" / "params" / "stage_B_retrained.json"
DEFAULT_DB_DIR = REPO_ROOT / "data" / "dbs"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "recipes" / "8color_boards.json"

GRID_ROWS = 40
GRID_COLS = 40
BOARD_SIZE = GRID_ROWS * GRID_COLS  # 1600
TOTAL_SELECT = BOARD_SIZE * 2        # 3200
NUM_CHANNELS = 8
COLOR_LAYERS = 5
LAYER_HEIGHT_MM = 0.08
MICRO_LAYER_HEIGHT = 0.04
LAYER_ORDER = "Top2Bottom"
MATERIAL = "PLA Basic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 8-color calibration board recipe selections."
    )
    parser.add_argument("--stage-b", type=Path, default=DEFAULT_STAGE_B)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def find_white_channel_idx(palette: Sequence[Dict[str, str]]) -> Optional[int]:
    for idx, item in enumerate(palette):
        if normalize_label(str(item.get("color", ""))) == "white":
            return idx
    return None


def collect_db_params(
    db_paths: Sequence[Path],
    stageb_color_map: Dict[str, int],
) -> Tuple[Optional[str], Optional[int], Optional[float], Optional[int]]:
    """Extract layer_order, base_channel_idx, line_width_mm, base_layers from DBs."""
    layer_order: Optional[str] = None
    base_channel_idx: Optional[int] = None
    line_width_mm: Optional[float] = None
    base_layers: Optional[int] = None
    for path in db_paths:
        data = load_json(path)
        db_layer_order = parse_layer_order(data.get("layer_order", "Top2Bottom"))
        if layer_order is None:
            layer_order = db_layer_order
        if base_layers is None and "base_layers" in data:
            base_layers = int(data.get("base_layers", 0))
        if line_width_mm is None and "line_width_mm" in data:
            line_width_mm = float(data.get("line_width_mm", 0.0))
        raw_palette = data.get("palette", [])
        if not isinstance(raw_palette, list) or not raw_palette:
            continue
        db_base_channel_idx = int(data.get("base_channel_idx", 0))
        db_to_stage: List[int] = []
        for item in raw_palette:
            color = normalize_label(str(item.get("color", "")))
            if color not in stageb_color_map:
                continue
            db_to_stage.append(stageb_color_map[color])
        if db_to_stage and base_channel_idx is None:
            if 0 <= db_base_channel_idx < len(db_to_stage):
                base_channel_idx = db_to_stage[db_base_channel_idx]
    return layer_order, base_channel_idx, line_width_mm, base_layers


def index_to_recipe(index: int, num_channels: int, color_layers: int, layer_order: str) -> List[int]:
    recipe = [0] * color_layers
    idx = index
    for i in range(color_layers):
        v = idx % num_channels
        idx //= num_channels
        layer = color_layers - 1 - i if layer_order == "Top2Bottom" else i
        recipe[layer] = int(v)
    return recipe


def enumerate_all_recipes(num_channels: int, color_layers: int, layer_order: str) -> np.ndarray:
    total = int(num_channels ** color_layers)
    recipes: List[List[int]] = []
    for idx in range(total):
        recipes.append(index_to_recipe(idx, num_channels, color_layers, layer_order))
    return np.asarray(recipes, dtype=np.int32)


def kcenter_select(labs: np.ndarray, k: int) -> List[int]:
    """Greedy k-center: iteratively pick the point farthest from any selected."""
    if labs.shape[0] < k:
        raise ValueError(f"Not enough candidates ({labs.shape[0]}) to select {k}.")
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


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)

    stage_model = load_stage_forward_model(args.stage_b)
    num_channels = stage_model.E.shape[0]
    if num_channels != NUM_CHANNELS:
        raise ValueError(f"Expected {NUM_CHANNELS} channels in Stage B, got {num_channels}")

    palette = [
        {"color": name, "material": MATERIAL}
        for name in stage_model.channel_names
    ]
    stageb_color_map = {normalize_label(item["color"]): idx for idx, item in enumerate(palette)}

    db_paths = resolve_db_paths(args.db)
    db_layer_order, db_base_idx, line_width_mm, base_layers = collect_db_params(
        db_paths, stageb_color_map,
    )
    layer_order = db_layer_order or LAYER_ORDER
    white_idx = find_white_channel_idx(palette)
    base_channel_idx = white_idx if white_idx is not None else (db_base_idx if db_base_idx is not None else 0)
    if base_layers is None:
        base_layers = 10
    if line_width_mm is None or line_width_mm <= 0:
        line_width_mm = 0.42

    substrate_idx = resolve_substrate_idx(stage_model, db_paths, base_channel_idx)

    print(f"Palette: {[p['color'] for p in palette]}")
    print(f"base_channel_idx={base_channel_idx}, layer_order={layer_order}")
    print(f"substrate_idx={substrate_idx}")

    total = int(NUM_CHANNELS ** COLOR_LAYERS)
    print(f"Enumerating all {total} recipes ...")
    all_recipes = enumerate_all_recipes(NUM_CHANNELS, COLOR_LAYERS, layer_order)

    print("Predicting Lab colors ...")
    linear = predict_with_stage_model(
        all_recipes, stage_model, LAYER_HEIGHT_MM, MICRO_LAYER_HEIGHT,
        base_channel_idx, layer_order, substrate_idx,
    )
    labs = linear_rgb_to_opencv_lab_batch(linear)

    print(f"Running k-center to select {TOTAL_SELECT} recipes ...")
    selected_order = kcenter_select(labs, TOTAL_SELECT)

    board1_idx = selected_order[:BOARD_SIZE]
    board2_idx = selected_order[BOARD_SIZE:]

    board1_labs = labs[board1_idx]
    board2_labs = labs[board2_idx]
    both_labs = labs[np.array(selected_order)]

    board1_sort = np.lexsort((board1_labs[:, 2], board1_labs[:, 1], board1_labs[:, 0]))
    board2_sort = np.lexsort((board2_labs[:, 2], board2_labs[:, 1], board2_labs[:, 0]))

    board1_recipes = all_recipes[board1_idx][board1_sort].tolist()
    board2_recipes = all_recipes[board2_idx][board2_sort].tolist()
    board1_labs = board1_labs[board1_sort]
    board2_labs = board2_labs[board2_sort]

    print(f"Board 1: {len(board1_recipes)} recipes, "
          f"Lab L range [{board1_labs[:,0].min():.1f}, {board1_labs[:,0].max():.1f}]")
    print(f"Board 2: {len(board2_recipes)} recipes, "
          f"Lab L range [{board2_labs[:,0].min():.1f}, {board2_labs[:,0].max():.1f}]")
    print(f"Combined Lab L range [{both_labs[:,0].min():.1f}, {both_labs[:,0].max():.1f}]")

    output = {
        "meta": {
            "generated_from": str(args.stage_b),
            "dbs": [str(p) for p in db_paths],
            "mode": "0.08x5",
            "layer_height_mm": LAYER_HEIGHT_MM,
            "color_layers": COLOR_LAYERS,
            "num_channels": NUM_CHANNELS,
            "layer_order": layer_order,
            "base_layers": base_layers,
            "base_channel_idx": base_channel_idx,
            "substrate_idx": int(substrate_idx),
            "line_width_mm": line_width_mm,
            "total_possible": total,
            "total_selected": TOTAL_SELECT,
        },
        "base_channel_idx": base_channel_idx,
        "base_layers": base_layers,
        "layer_height_mm": LAYER_HEIGHT_MM,
        "line_width_mm": line_width_mm,
        "layer_order": layer_order,
        "palette": palette,
        "boards": [
            {
                "board_index": 1,
                "grid_rows": GRID_ROWS,
                "grid_cols": GRID_COLS,
                "recipes": board1_recipes,
            },
            {
                "board_index": 2,
                "grid_rows": GRID_ROWS,
                "grid_cols": GRID_COLS,
                "recipes": board2_recipes,
            },
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
