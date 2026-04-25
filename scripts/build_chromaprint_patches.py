#!/usr/bin/env python3
"""Generate a draft `data/presets/chromaprint_patches.json` by diffing the
4 reference P2S preset JSONs against BambuStudio system process presets.

This is a maintenance tool: produce a fresh draft that the ChromaPrint3D
team reviews and edits (especially DD TPU HF / Bowden Std / Bowden HF
variant values, which cannot be inferred from a single-extruder P2S
reference).

The script does NOT overwrite an existing patches file. Output goes to
stdout (or `--output`) for hand-merging.

Plan reference: v13 §4.2 / §9 step 2.

Usage:
    python3 scripts/build_chromaprint_patches.py \
        --bambu-resources path/to/BambuStudio/resources/profiles \
        --reference-dir   data/presets \
        --output          /tmp/chromaprint_patches.draft.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# BambuStudio PrintConfig.cpp set extraction
# ---------------------------------------------------------------------------


def _parse_string_set(text: str, set_name: str) -> List[str]:
    """Extract `std::set<std::string> <set_name> = { "a", "b", ... };` from C++."""
    pat = re.compile(
        rf"std::set<std::string>\s+{set_name}\s*=\s*\{{(.*?)\}};",
        re.DOTALL,
    )
    m = pat.search(text)
    if not m:
        return []
    body = m.group(1)
    body = re.sub(r"//[^\n]*", "", body)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
    return re.findall(r'"([^"]+)"', body)


def load_field_sets(print_config_path: str) -> Dict[str, set]:
    with open(print_config_path, "r", encoding="utf-8") as f:
        text = f.read()
    return {
        "print_options_with_variant": set(_parse_string_set(text, "print_options_with_variant")),
        "filament_options_with_variant": set(_parse_string_set(text, "filament_options_with_variant")),
        "printer_extruder_options": set(_parse_string_set(text, "printer_extruder_options")),
        "printer_options_with_variant_1": set(_parse_string_set(text, "printer_options_with_variant_1")),
        "printer_options_with_variant_2": set(_parse_string_set(text, "printer_options_with_variant_2")),
    }


# ---------------------------------------------------------------------------
# BambuStudio system process inherits-chain merge
# ---------------------------------------------------------------------------


def chain_merge_process(profile_dir: str, name: str) -> Dict[str, Any]:
    """Recursively follow `inherits` and return the fully merged dict."""
    merged: Dict[str, Any] = {}
    chain: List[Dict[str, Any]] = []
    cur: Optional[str] = name
    proc_dir = os.path.join(profile_dir, "BBL", "process")
    while cur:
        path = os.path.join(proc_dir, f"{cur}.json")
        if not os.path.exists(path):
            break
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        chain.append(d)
        cur = d.get("inherits") or None
    for d in reversed(chain):
        merged.update(d)
    return merged


# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_values_equal(x, y) for x, y in zip(a, b))
    return str(a) == str(b)


def diff_against_system(preset: Dict[str, Any], system: Dict[str, Any]) -> Dict[str, Any]:
    """Return preset_value for every key whose value differs from the system default."""
    out: Dict[str, Any] = {}
    for k, v in preset.items():
        if k.startswith("_") or k in {
            "from", "name", "version", "is_custom_defined", "type", "setting_id",
            "filament_settings_id", "printer_settings_id", "print_settings_id",
            "different_settings_to_system", "inherits_group", "inherits",
            "compatible_printers", "compatible_prints", "compatible_machine_expression_group",
            "compatible_process_expression_group", "print_compatible_printers",
            "print_extruder_id",
        }:
            continue
        if k in {"filament_colour", "filament_multi_colour", "filament_type",
                 "filament_ids", "filament_vendor", "default_filament_colour",
                 "nozzle_temperature", "nozzle_temperature_initial_layer",
                 "filament_extruder_variant", "filament_self_index",
                 "extruder_variant_list", "flush_volumes_matrix",
                 "wipe_tower_x", "wipe_tower_y"}:
            continue
        sv = system.get(k)
        if sv is None:
            out[k] = v
        elif not _values_equal(v, sv):
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Variant-indexed transformation
# ---------------------------------------------------------------------------


_VARIANT_FALLBACKS = [
    "Direct Drive Standard",
    "Direct Drive High Flow",
    "Direct Drive TPU High Flow",
    "Bowden Standard",
    "Bowden High Flow",
]


def to_variant_indexed(values: Sequence[Any], variant: Sequence[str], extra_variants: Sequence[str]) -> Dict[str, Any]:
    """Build {variant_name: value} dict; fill TODO_<variant> for variants not present
    in the reference preset."""
    if len(values) != len(variant):
        raise ValueError(
            f"variant length {len(variant)} does not match value length {len(values)}: {variant} vs {values}"
        )
    out: Dict[str, str] = {}
    for v_name, v_val in zip(variant, values):
        out[v_name] = str(v_val)
    for v_name in extra_variants:
        if v_name not in out:
            # Default to the lowest seen value (e.g. DD Std), so manual review is required.
            fallback = next(iter(out.values())) if out else ""
            out[v_name] = f"TODO_{fallback}"
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


REFERENCE_PRESETS = {
    ("0.2", "FaceUp"):   "bambu_p2s_0.08mm_n02_faceup.json",
    ("0.2", "FaceDown"): "bambu_p2s_0.08mm_n02_facedown.json",
    ("0.4", "FaceUp"):   "bambu_p2s_0.08mm_n04_faceup.json",
    ("0.4", "FaceDown"): "bambu_p2s_0.08mm_n04_facedown.json",
}

SYSTEM_PROCESS_FOR_NOZZLE = {
    "0.2": "0.08mm High Quality @BBL P2S 0.2 nozzle",
    "0.4": "0.08mm High Quality @BBL P2S",
}


def _classify(key: str, sets: Dict[str, set]) -> str:
    if key in sets["print_options_with_variant"]:
        return "print_with_variant"
    if key in sets["filament_options_with_variant"]:
        return "filament_with_variant"
    if key in sets["printer_extruder_options"]:
        return "printer_extruder"
    if key in sets["printer_options_with_variant_1"] or key in sets["printer_options_with_variant_2"]:
        return "printer_with_variant"
    return "scalar"


def build_draft(
    reference_dir: str,
    bambu_resources: str,
) -> Dict[str, Any]:
    sets = load_field_sets(os.path.join(bambu_resources, "..", "..", "src", "libslic3r", "PrintConfig.cpp"))

    preset_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    diff_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for (nozzle, face), fname in REFERENCE_PRESETS.items():
        path = os.path.join(reference_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            preset = json.load(f)
        preset_by_key[(nozzle, face)] = preset

        # Use BambuStudio's authoritative `different_settings_to_system[0]` field, which
        # encodes exactly which fields the user (or ChromaPrint3D) modified vs system
        # defaults. This avoids false positives from filament / printer inheritance.
        diff_field = preset.get("different_settings_to_system", [""])[0]
        diff_keys = [k for k in diff_field.split(";") if k]
        diff_by_key[(nozzle, face)] = {k: preset[k] for k in diff_keys if k in preset}

    # Fields appearing in every (nozzle × face) diff with same value -> process_common
    # Fields appearing in both faces of one nozzle (but not other nozzle) -> process_per_nozzle
    # Fields differing between faces -> process_per_face
    all_keys = set()
    for d in diff_by_key.values():
        all_keys.update(d.keys())

    common: Dict[str, Any] = {}
    per_nozzle: Dict[str, Dict[str, Any]] = {"0.2": {}, "0.4": {}}
    per_face: Dict[str, Dict[str, Any]] = {"FaceUp": {}, "FaceDown": {}}

    def _val(nozzle: str, face: str, key: str) -> Optional[Any]:
        return diff_by_key[(nozzle, face)].get(key)

    def _emit(value: Any, variant: Sequence[str], category: str) -> Any:
        """Convert value to plain string or $variant_indexed form."""
        if category == "print_with_variant" and isinstance(value, list):
            return {
                "$variant_indexed": to_variant_indexed(value, variant, _VARIANT_FALLBACKS)
            }
        if isinstance(value, list):
            return [str(x) for x in value]
        return str(value)

    for key in sorted(all_keys):
        cat = _classify(key, sets)
        if cat in {"printer_extruder", "printer_with_variant", "filament_with_variant"}:
            # ChromaPrint3D does not modify these; if present in diff, skip and warn.
            print(f"warn: skip non-process field {key!r} (category={cat})", file=sys.stderr)
            continue

        # All four diffs have same value? -> process_common
        all_vals = [_val(n, f, key) for n in ("0.2", "0.4") for f in ("FaceUp", "FaceDown")]
        all_present = all(v is not None for v in all_vals)
        if all_present and all(_values_equal(v, all_vals[0]) for v in all_vals):
            variant = preset_by_key[("0.4", "FaceUp")].get("print_extruder_variant", [])
            common[key] = _emit(all_vals[0], variant, cat)
            continue

        # Per nozzle (consistent across faces of same nozzle)
        for nozzle in ("0.2", "0.4"):
            v_up = _val(nozzle, "FaceUp", key)
            v_dn = _val(nozzle, "FaceDown", key)
            if v_up is not None and v_dn is not None and _values_equal(v_up, v_dn):
                variant = preset_by_key[(nozzle, "FaceUp")].get("print_extruder_variant", [])
                per_nozzle[nozzle][key] = _emit(v_up, variant, cat)

        # Per face (set if differs across faces)
        # Collect by face only when the same value across nozzles for that face but
        # differs from the other face.
        for face in ("FaceUp", "FaceDown"):
            v_n02 = _val("0.2", face, key)
            v_n04 = _val("0.4", face, key)
            if v_n02 is not None and v_n04 is not None and _values_equal(v_n02, v_n04):
                # Already covered by process_common if also in other face; otherwise emit per_face.
                other = "FaceDown" if face == "FaceUp" else "FaceUp"
                v_other_n02 = _val("0.2", other, key)
                v_other_n04 = _val("0.4", other, key)
                if not (v_other_n02 is not None and _values_equal(v_n02, v_other_n02) and
                        v_other_n04 is not None and _values_equal(v_n04, v_other_n04)):
                    variant = preset_by_key[("0.4", face)].get("print_extruder_variant", [])
                    per_face[face][key] = _emit(v_n02, variant, cat)

    return {
        "_doc": [
            "Auto-generated DRAFT from `scripts/build_chromaprint_patches.py`.",
            "Hand-review required: fill TODO_* placeholders for DD TPU HF /",
            "Bowden Standard / Bowden High Flow variant values.",
            "Then merge into `data/presets/chromaprint_patches.json`.",
        ],
        "process_common": common,
        "process_per_nozzle": per_nozzle,
        "process_per_face": per_face,
        "filament_common": {},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bambu-resources", required=True,
                   help="Path to BambuStudio/resources/profiles directory")
    p.add_argument("--reference-dir", default="data/presets",
                   help="Directory containing the 4 reference P2S preset JSONs")
    p.add_argument("--output", default="-",
                   help="Output path (default: stdout)")
    args = p.parse_args()

    draft = build_draft(args.reference_dir, args.bambu_resources)
    text = json.dumps(draft, indent=2, ensure_ascii=False)
    if args.output == "-":
        sys.stdout.write(text + "\n")
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"Draft written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
