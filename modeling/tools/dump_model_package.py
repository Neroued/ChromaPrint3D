#!/usr/bin/env python3
"""
Convert a v2 MessagePack model package to readable JSON for debugging.

Example:
  python -m modeling.tools.dump_model_package \
    --input model_package.msgpack \
    --output model_package_readable.json \
    --max-candidates 10
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import msgpack


def decode_lab_blob(data: bytes, num_candidates: int) -> list:
    labs = []
    for i in range(min(num_candidates, len(data) // 12)):
        L, a, b = struct.unpack_from("<fff", data, i * 12)
        labs.append([round(L, 4), round(a, 4), round(b, 4)])
    return labs


def decode_recipes_blob(data: bytes, num_candidates: int, color_layers: int) -> list:
    recipes = []
    stride = color_layers
    for i in range(min(num_candidates, len(data) // stride)):
        offset = i * stride
        recipes.append(list(data[offset:offset + stride]))
    return recipes


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump v2 model package to JSON.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=0,
                        help="Limit candidates per mode (0 = all)")
    args = parser.parse_args()

    raw = args.input.read_bytes()
    data = msgpack.unpackb(raw, raw=False)

    for mode in data.get("modes", []):
        num = mode.get("num_candidates", 0)
        cl = mode.get("color_layers", 0)
        limit = args.max_candidates if args.max_candidates > 0 else num

        recipes_bin = mode.get("candidate_recipes", b"")
        if isinstance(recipes_bin, (bytes, bytearray)):
            mode["candidate_recipes"] = decode_recipes_blob(recipes_bin, limit, cl)
            mode["candidate_recipes_truncated"] = limit < num
        lab_bin = mode.get("pred_lab", b"")
        if isinstance(lab_bin, (bytes, bytearray)):
            mode["pred_lab"] = decode_lab_blob(lab_bin, limit)
            mode["pred_lab_truncated"] = limit < num

    text = json.dumps(data, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Written to {args.output}")
    else:
        sys.stdout.write(text)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
