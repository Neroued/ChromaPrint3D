#!/usr/bin/env python3
"""Generate offline preset base files for ChromaPrint3D multi-machine support.

For every machine x nozzle combination listed in `data/presets/machines.json`,
this script:

  1. Loads BambuStudio's BBL.json registry (machine/process/filament path map).
  2. Recursively merges the three `inherits` chains:
        - process: process_template[nozzle]
        - filament: filament_template[nozzle]
        - printer:  printer_template (formatted with {nozzle})
  3. Combines them into a single flat `base_dict` (mirrors BambuStudio's
     project_settings.config layout).
  4. Step 3.5 (plan v13 §4.1 / CCC) - pre-aligns every `filament_options_with_variant`
     field to K_per_extruder = len(extruder_variant_list[0].split(',')):
        - K_filament_raw == K_per_extruder : pass through
        - K_filament_raw == 1              : replicate single value across K_per_extruder slots
        - K_filament_raw < K_per_extruder  : pattern-replicate by i % K_filament_raw
        - otherwise                        : fail with diagnostic
     (plan v13.2: filament arrays use K_per_extruder, NOT K_process. K_process is
      the print_options_with_variant length and is preserved separately.)
  5. Injects 1-slot filament user-level placeholders (filament_colour,
     filament_multi_colour, filament_ids, default_filament_colour,
     filament_settings_id) - BambuStudio inheritance never produces these.
  6. Drops `EXCLUDED_DYNAMIC_FIELDS` (e.g. wipe_tower_x/y) and runtime-injected
     metadata (different_settings_to_system, inherits, name, etc.).
  7. Writes `data/preset_bases/<model_slug>_<nozzle>_<lh>.json`.

The C++ runtime (`BuildProjectSettings`) consumes these base files and:
  - applies `chromaprint_patches.json`,
  - extends 1-slot/K_process arrays to N/N*K_process by replication,
  - generates the 3 mandatory variant meta-fields (extruder_variant_list,
    filament_extruder_variant, filament_self_index),
  - patches user-supplied filament colours/material slots.

Plan reference: v13 §4.1, §3.3.

Usage:
    python3 scripts/build_preset_bases.py \
        --bambu-resources path/to/BambuStudio/resources/profiles \
        --machines        data/presets/machines.json \
        --output          data/preset_bases \
        [--layer-heights 0.08] \
        [--only "Bambu Lab P2S"] \
        [--validate-only] \
        [--check] \
        [--print-chains] \
        [--list-print-with-variant-keys] \
        [--list-filament-with-variant-keys] \
        [--list-filament-no-variant-keys] \
        [--regen-golden]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


EXCLUDED_DYNAMIC_FIELDS = {
    "wipe_tower_x",
    "wipe_tower_y",
}

# Runtime-injected metadata; not part of base_dict.
RUNTIME_METADATA_FIELDS = {
    "from",
    "name",
    "type",
    "version",
    "is_custom_defined",
    "setting_id",
    "inherits",
    "inherits_group",
    "different_settings_to_system",
    "compatible_printers",
    "compatible_printers_condition",
    "compatible_prints",
    "compatible_prints_condition",
    "compatible_machine_expression_group",
    "compatible_process_expression_group",
    "print_compatible_printers",
    "filament_self_index",
    "print_settings_id",
    "printer_settings_id",
    "filament_settings_id",  # injected per machine below, but never inherited
    "instantiation",
    "include",          # BambuStudio inherits-chain meta (analogous to `inherits`)
    "renamed_from",
    "default_filament_profile",
    "default_print_profile",
}

# 1-slot filament user-level placeholders (plan v13 §3.3).
INJECTED_FILAMENT_FIELDS = {
    "filament_colour": ["#FFFFFF"],
    "filament_multi_colour": ["#FFFFFF"],
    "filament_ids": ["GFA00"],
    "default_filament_colour": [""],
    "filament_type": ["PLA"],
    "filament_vendor": ["Bambu Lab"],
}


# ---------------------------------------------------------------------------
# BambuStudio registry & inherits-chain merge
# ---------------------------------------------------------------------------


class BambuRegistry:
    """Load BBL.json and resolve preset file paths by name."""

    def __init__(self, profiles_dir: str) -> None:
        bbl_path = os.path.join(profiles_dir, "BBL.json")
        if not os.path.exists(bbl_path):
            raise FileNotFoundError(f"BBL.json not found at {bbl_path}")
        with open(bbl_path, "r", encoding="utf-8") as f:
            self.bbl = json.load(f)
        self.profiles_dir = profiles_dir
        self._machine = {a["name"]: a["sub_path"] for a in self.bbl.get("machine_list", [])}
        self._process = {a["name"]: a["sub_path"] for a in self.bbl.get("process_list", [])}
        self._filament = {a["name"]: a["sub_path"] for a in self.bbl.get("filament_list", [])}

    def has_machine(self, name: str) -> bool:
        return name in self._machine

    def has_process(self, name: str) -> bool:
        return name in self._process

    def has_filament(self, name: str) -> bool:
        return name in self._filament

    def list_machines(self) -> List[str]:
        return list(self._machine.keys())

    def list_processes(self) -> List[str]:
        return list(self._process.keys())

    def list_filaments(self) -> List[str]:
        return list(self._filament.keys())

    def _resolve(self, sub_path: str) -> str:
        # `sub_path` is relative to the BBL/ folder, e.g. "filament/P1P/Bambu PLA Basic @BBL P1P.json".
        return os.path.join(self.profiles_dir, "BBL", sub_path)

    def chain(self, kind: str, name: str) -> List[Dict[str, Any]]:
        """Return the inherits chain leaf -> root, fully loaded JSON dicts."""
        out: List[Dict[str, Any]] = []
        cur: Optional[str] = name
        seen: Set[str] = set()
        while cur:
            if cur in seen:
                raise RuntimeError(f"inherits cycle for {kind}: {cur!r}")
            seen.add(cur)
            sub_path = self._lookup(kind, cur)
            if not sub_path:
                if not out:
                    raise FileNotFoundError(f"{kind} preset not registered in BBL.json: {cur!r}")
                # Common ancestors (e.g. fdm_filament_common) may live outside BBL/.
                candidate = os.path.join(self.profiles_dir, "BBL", f"{cur}.json")
                if not os.path.exists(candidate):
                    candidate = os.path.join(self.profiles_dir, f"{cur}.json")
                if not os.path.exists(candidate):
                    # Search recursively as last resort (BBL/<subdir>/<name>.json).
                    for root, _, files in os.walk(os.path.join(self.profiles_dir, "BBL")):
                        if f"{cur}.json" in files:
                            candidate = os.path.join(root, f"{cur}.json")
                            break
                if not os.path.exists(candidate):
                    # Reached root of inheritance.
                    break
                full_path = candidate
            else:
                full_path = self._resolve(sub_path)
            with open(full_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            out.append(d)
            cur = d.get("inherits") or None
        return out

    def _lookup(self, kind: str, name: str) -> Optional[str]:
        if kind == "machine":
            return self._machine.get(name)
        if kind == "process":
            return self._process.get(name)
        if kind == "filament":
            return self._filament.get(name)
        raise ValueError(f"Unknown kind: {kind!r}")


def merge_chain(chain: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge a leaf->root chain so leaf values win (BambuStudio inherits semantics)."""
    merged: Dict[str, Any] = {}
    for d in reversed(chain):
        merged.update(d)
    return merged


# ---------------------------------------------------------------------------
# PrintConfig.cpp field-set extraction
# ---------------------------------------------------------------------------


def _parse_string_set(text: str, set_name: str) -> List[str]:
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


def load_field_sets(profiles_dir: str) -> Dict[str, Set[str]]:
    cpp_path = os.path.join(profiles_dir, "..", "..", "src", "libslic3r", "PrintConfig.cpp")
    cpp_path = os.path.normpath(cpp_path)
    if not os.path.exists(cpp_path):
        raise FileNotFoundError(f"PrintConfig.cpp not found at {cpp_path}")
    with open(cpp_path, "r", encoding="utf-8") as f:
        text = f.read()
    return {
        "print_options_with_variant": set(_parse_string_set(text, "print_options_with_variant")),
        "filament_options_with_variant": set(_parse_string_set(text, "filament_options_with_variant")),
        "printer_extruder_options": set(_parse_string_set(text, "printer_extruder_options")),
        "printer_options_with_variant_1": set(_parse_string_set(text, "printer_options_with_variant_1")),
        "printer_options_with_variant_2": set(_parse_string_set(text, "printer_options_with_variant_2")),
    }


# ---------------------------------------------------------------------------
# Step 3.5: K_per_extruder pre-alignment (plan v13.2)
# ---------------------------------------------------------------------------


def parse_extruder0_variants(extruder_variant_list: List[str]) -> List[str]:
    """Parse `extruder_variant_list[0]` (e.g. 'Direct Drive Standard,Direct Drive High Flow')
    into the list of variants for extruder 0. K_per_extruder = len(result).

    Source-of-truth fact (BambuStudio PrintConfig.cpp:7571-7580 +
    PresetBundle.cpp:225-232 + Print.cpp:8462+):
    BambuStudio's filament arrays use N × K_per_extruder length, and
    K_per_extruder = number of variants on extruder 0 (= filament_variant_count[i]).
    """
    if not isinstance(extruder_variant_list, list) or not extruder_variant_list:
        raise ValueError("extruder_variant_list is empty or missing")
    raw = extruder_variant_list[0]
    if not isinstance(raw, str):
        raise ValueError(f"extruder_variant_list[0] is not a string: {raw!r}")
    variants = [v.strip() for v in raw.split(",") if v.strip()]
    if not variants:
        raise ValueError(f"extruder_variant_list[0] yielded no variants: {raw!r}")
    return variants


def align_filament_with_variant(
    base_dict: Dict[str, Any],
    field_set: Set[str],
    machine_name: str,
    nozzle: str,
) -> Tuple[int, int, List[str]]:
    """Align every `filament_options_with_variant` array in base_dict to K_per_extruder.

    Returns (K_per_extruder, K_filament_raw, modified_keys).
    Raises ValueError on unsupported alignment patterns.

    Plan v13.2 / m-realfile: K is K_per_extruder (= len(extruder_variant_list[0].split(',')))
    not K_process (= len(print_extruder_variant)). For most machines K_process ==
    extruder_count × K_per_extruder; some (H2D) have asymmetric per-extruder variant
    counts where K_process != extruder_count × K_per_extruder, but BambuStudio's
    filament arrays still use K_per_extruder as the per-slot stride.
    """
    extruder_variant_list = base_dict.get("extruder_variant_list", [])
    extruder0_variants = parse_extruder0_variants(extruder_variant_list)
    K_per_extruder = len(extruder0_variants)

    # Use nozzle_temperature as the canonical filament-with-variant length probe.
    nt = base_dict.get("nozzle_temperature", [])
    if not isinstance(nt, list):
        raise ValueError(f"{machine_name} {nozzle}: nozzle_temperature is not an array")
    K_filament_raw = len(nt)

    modified: List[str] = []
    for key in sorted(field_set):
        if key not in base_dict:
            continue
        v = base_dict[key]
        if not isinstance(v, list):
            continue
        cur_len = len(v)
        if cur_len == K_per_extruder:
            continue
        if cur_len == 1:
            base_dict[key] = [v[0]] * K_per_extruder
        elif cur_len < K_per_extruder and K_per_extruder % cur_len == 0:
            base_dict[key] = [v[i % cur_len] for i in range(K_per_extruder)]
        elif cur_len < K_per_extruder:
            # Non-divisor: i%cur_len fallback + warn
            base_dict[key] = [v[i % cur_len] for i in range(K_per_extruder)]
            print(
                f"warn: {machine_name} {nozzle} field {key!r}: "
                f"K_filament={cur_len} not a divisor of K_per_extruder={K_per_extruder}; "
                f"non-divisor pattern-replicate (i%{cur_len} fallback; consider explicit "
                f"override if precision required)",
                file=sys.stderr,
            )
        elif cur_len > K_per_extruder:
            # Truncate longer-than-K_per_extruder arrays.
            base_dict[key] = v[:K_per_extruder]
            print(
                f"warn: {machine_name} {nozzle} field {key!r}: "
                f"K_filament={cur_len} > K_per_extruder={K_per_extruder}; truncated "
                f"(consider explicit override if precision required)",
                file=sys.stderr,
            )
        else:
            raise ValueError(
                f"{machine_name} {nozzle} field {key!r}: "
                f"unsupported length combination cur={cur_len}, K_per_extruder={K_per_extruder}"
            )
        modified.append(key)

    return K_per_extruder, K_filament_raw, modified


# ---------------------------------------------------------------------------
# Filament 1-slot user-level injection
# ---------------------------------------------------------------------------


def inject_filament_user_fields(base_dict: Dict[str, Any], filament_template: str) -> None:
    """Inject 1-slot filament user-level fields (plan v13 §3.3 / proposition 11)."""
    for key, default in INJECTED_FILAMENT_FIELDS.items():
        # Always overwrite to the canonical 1-slot placeholder (BambuStudio
        # inheritance may seed empty strings, e.g. filament_settings_id: [""]).
        base_dict[key] = list(default)
    base_dict["filament_settings_id"] = [filament_template]


# ---------------------------------------------------------------------------
# Build & strip
# ---------------------------------------------------------------------------


def build_base_dict(
    registry: BambuRegistry,
    machine_name: str,
    machine_spec: Dict[str, Any],
    nozzle: str,
    sets: Dict[str, Set[str]],
    print_chains: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (base_dict, diagnostics) for one (machine, nozzle).

    diagnostics carries K_per_extruder / K_process / K_filament_raw / topology / aligned_keys
    for `--validate-only` reporting.
    """
    process_template = machine_spec["process_template"][nozzle]
    filament_template = machine_spec["filament_template"][nozzle]
    printer_template = machine_spec["printer_template"].format(nozzle=nozzle)

    # 1. Validate references are registered.
    if not registry.has_process(process_template):
        raise FileNotFoundError(
            f"{machine_name} {nozzle}: process_template not registered: {process_template!r}"
        )
    if not registry.has_filament(filament_template):
        raise FileNotFoundError(
            f"{machine_name} {nozzle}: filament_template not registered: {filament_template!r}"
        )
    if not registry.has_machine(printer_template):
        raise FileNotFoundError(
            f"{machine_name} {nozzle}: printer_template not registered: {printer_template!r}"
        )

    # 2. Resolve & merge each chain.
    proc_chain = registry.chain("process", process_template)
    fil_chain = registry.chain("filament", filament_template)
    prt_chain = registry.chain("machine", printer_template)

    if print_chains:
        print(f"=== {machine_name} {nozzle} ===", file=sys.stderr)
        for label, ch in (("process", proc_chain), ("filament", fil_chain), ("printer", prt_chain)):
            names = [d.get("name", "?") for d in ch]
            print(f"  {label} chain (leaf->root): {names}", file=sys.stderr)

    proc_merged = merge_chain(proc_chain)
    fil_merged = merge_chain(fil_chain)
    prt_merged = merge_chain(prt_chain)

    # 3. Combine into base_dict. Field categorization:
    #    - print_options_with_variant + scalar process fields -> from process
    #    - filament_options_no_variant + filament_options_with_variant -> from filament
    #    - printer_extruder_options + printer_options_with_variant_1/_2 + scalar printer -> from printer
    # We start from printer_merged (lowest priority), layer filament_merged on top,
    # then process_merged. Conflicts on overlapping fields are extremely rare since
    # the three sets are disjoint by design.
    base_dict: Dict[str, Any] = {}
    for src in (prt_merged, fil_merged, proc_merged):
        for k, v in src.items():
            if k in RUNTIME_METADATA_FIELDS:
                continue
            if k in EXCLUDED_DYNAMIC_FIELDS:
                continue
            base_dict[k] = v

    # 4. Inject filament user-level 1-slot placeholders.
    inject_filament_user_fields(base_dict, filament_template)

    # 5. Step 3.5 K_per_extruder pre-alignment (plan v13.2).
    K_per_extruder, K_filament_raw, aligned = align_filament_with_variant(
        base_dict, sets["filament_options_with_variant"], machine_name, nozzle
    )
    K_process = len(base_dict["print_extruder_variant"])

    # 6. filament_extruder_variant: BambuStudio's filament arrays use
    #    extruder_variant_list[0]'s variants (extruder 0 only, K_per_extruder),
    #    NOT print_extruder_variant (K_process). See plan v13.2 / m-realfile.
    extruder0_variants = parse_extruder0_variants(base_dict["extruder_variant_list"])
    base_dict["filament_extruder_variant"] = list(extruder0_variants)

    # 7. Validate `extruder_variant_list` is present (BambuStudio strictly
    #    checks this on 3MF load - plan v13 proposition 13).
    if "extruder_variant_list" not in base_dict:
        raise ValueError(
            f"{machine_name} {nozzle}: extruder_variant_list missing from printer chain"
        )

    diag = {
        "machine": machine_name,
        "nozzle": nozzle,
        "topology": machine_spec["extruder_topology"],
        "K_process": K_process,
        "K_per_extruder": K_per_extruder,
        "K_filament_raw": K_filament_raw,
        "aligned_keys": aligned,
        "process_template": process_template,
        "filament_template": filament_template,
        "printer_template": printer_template,
        "printer_model": prt_merged.get("printer_model"),
        "extruder_variant_list": base_dict.get("extruder_variant_list"),
        "print_extruder_variant": base_dict.get("print_extruder_variant"),
        "field_count": len(base_dict),
    }
    return base_dict, diag


_SLUG_RE = re.compile(r"^[a-z0-9_]+$")


def validate_machines_schema(machines_json: Dict[str, Any]) -> None:
    """Strict-validate `machines.json` schema; raise ValueError on first issue.

    Required per-entry: extruder_topology, slug, nozzles, process_template,
    filament_template, printer_template. `slug` must match ^[a-z0-9_]+$ and be unique.
    """
    if "machines" not in machines_json or not isinstance(machines_json["machines"], dict):
        raise ValueError("machines.json: missing or invalid 'machines' object")
    if "default_machine" not in machines_json:
        raise ValueError("machines.json: missing 'default_machine'")
    seen_slugs: Set[str] = set()
    required = ("extruder_topology", "slug", "nozzles",
                "process_template", "filament_template", "printer_template")
    for name, spec in machines_json["machines"].items():
        for k in required:
            if k not in spec:
                raise ValueError(f"machines.json: {name!r} missing field {k!r}")
        slug = spec["slug"]
        if not isinstance(slug, str) or not _SLUG_RE.match(slug):
            raise ValueError(
                f"machines.json: {name!r} has invalid slug {slug!r} "
                f"(must match ^[a-z0-9_]+$)"
            )
        if slug in seen_slugs:
            raise ValueError(f"machines.json: duplicate slug {slug!r}")
        seen_slugs.add(slug)


def output_filename(slug: str, nozzle: str, layer_height: float) -> str:
    """Compose preset_bases output filename: e.g. `bambu_p2s_0.08mm_n04.json`.

    `slug` MUST come from `machines.json` `slug` field (plan v13.1 / m1).
    """
    nozzle_tag = "n02" if nozzle == "0.2" else "n04"
    lh = f"{layer_height:.2f}mm"
    return f"{slug}_{lh}_{nozzle_tag}.json"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bambu-resources", required=True,
                   help="Path to BambuStudio/resources/profiles directory")
    p.add_argument("--machines", default="data/presets/machines.json",
                   help="machines.json path (default: data/presets/machines.json)")
    p.add_argument("--output", default="data/preset_bases",
                   help="Output directory (default: data/preset_bases)")
    p.add_argument("--layer-heights", default="0.08",
                   help="Comma-separated layer heights (default: 0.08)")
    p.add_argument("--only", default=None,
                   help="Only process this machine name (incremental mode)")
    p.add_argument("--validate-only", action="store_true",
                   help="Validate references and report K_process/K_filament; do not write files")
    p.add_argument("--check", action="store_true",
                   help="Generate to a temp dir and diff against existing output")
    p.add_argument("--print-chains", action="store_true",
                   help="Print inherits chains for debugging")
    p.add_argument("--list-print-with-variant-keys", action="store_true",
                   help="Dump print_options_with_variant set and exit")
    p.add_argument("--list-filament-with-variant-keys", action="store_true",
                   help="Dump filament_options_with_variant set and exit")
    p.add_argument("--list-filament-no-variant-keys", action="store_true",
                   help="Dump fields appearing in 1-slot N-extension category (filament fields not in the variant set) and exit")
    p.add_argument("--regen-golden", action="store_true",
                   help="Force overwrite of golden files in tests")
    args = p.parse_args()

    sets = load_field_sets(args.bambu_resources)

    if args.list_print_with_variant_keys:
        for k in sorted(sets["print_options_with_variant"]):
            print(k)
        return 0
    if args.list_filament_with_variant_keys:
        for k in sorted(sets["filament_options_with_variant"]):
            print(k)
        return 0
    if args.list_filament_no_variant_keys:
        # Best-effort: derive from a reference filament inheritance merge.
        registry = BambuRegistry(args.bambu_resources)
        chain = registry.chain("filament", "Bambu PLA Basic @BBL P2S")
        merged = merge_chain(chain)
        no_variant = sorted(
            k for k, v in merged.items()
            if k.startswith("filament_") or k in {"nozzle_temperature_range_low",
                                                   "nozzle_temperature_range_high"}
            if k not in sets["filament_options_with_variant"]
        )
        for k in no_variant:
            print(k)
        return 0

    with open(args.machines, "r", encoding="utf-8") as f:
        machines = json.load(f)

    validate_machines_schema(machines)

    registry = BambuRegistry(args.bambu_resources)
    layer_heights = [float(x) for x in args.layer_heights.split(",")]

    machines_to_process = list(machines["machines"].items())
    if args.only:
        machines_to_process = [(n, s) for n, s in machines_to_process if n == args.only]
        if not machines_to_process:
            print(f"error: machine {args.only!r} not found in machines.json", file=sys.stderr)
            return 1

    diagnostics: List[Dict[str, Any]] = []
    failures: List[str] = []
    out_dir = args.output

    if not args.validate_only and not args.check:
        os.makedirs(out_dir, exist_ok=True)

    for machine_name, spec in machines_to_process:
        for nozzle in spec["nozzles"]:
            try:
                base, diag = build_base_dict(
                    registry, machine_name, spec, nozzle, sets,
                    print_chains=args.print_chains,
                )
            except Exception as exc:
                failures.append(f"{machine_name} {nozzle}: {exc}")
                print(f"error: {machine_name} {nozzle}: {exc}", file=sys.stderr)
                continue
            diagnostics.append(diag)

            if args.validate_only:
                continue

            # Embed source machine + nozzle metadata for the C++ catalog loader.
            base["_chromaprint3d_meta"] = {
                "machine_name": machine_name,
                "nozzle": nozzle,
                "extruder_topology": spec["extruder_topology"],
                "process_template": diag["process_template"],
                "filament_template": diag["filament_template"],
                "printer_template": diag["printer_template"],
                "printer_model": diag["printer_model"],
                "K_process": diag["K_process"],
                "K_filament_raw": diag["K_filament_raw"],
            }

            for lh in layer_heights:
                fname = output_filename(spec["slug"], nozzle, lh)
                fpath = os.path.join(out_dir, fname)
                if args.check:
                    if os.path.exists(fpath):
                        with open(fpath, "r", encoding="utf-8") as f:
                            existing = json.load(f)
                        if existing != base:
                            failures.append(f"{fname}: drift detected vs existing file")
                            print(f"drift: {fname}", file=sys.stderr)
                else:
                    with open(fpath, "w", encoding="utf-8") as f:
                        json.dump(base, f, indent=2, ensure_ascii=False)
                        f.write("\n")
                    print(f"wrote {fpath}", file=sys.stderr)

    # Validate-only summary report.
    if args.validate_only:
        print("\n=== K_per_extruder vs K_filament_raw report (plan v13.2) ===")
        mismatches = [d for d in diagnostics if d["K_per_extruder"] != d["K_filament_raw"]]
        for d in diagnostics:
            ok = "OK" if d["K_per_extruder"] == d["K_filament_raw"] else "MISMATCH"
            print(
                f"  {d['machine']:25s} {d['nozzle']}  topology={d['topology']:6s}  "
                f"K_process={d['K_process']}  K_per_extruder={d['K_per_extruder']}  "
                f"K_filament_raw={d['K_filament_raw']}  [{ok}]  "
                f"aligned={len(d['aligned_keys'])} fields"
            )
        if mismatches:
            print(
                f"\n{len(mismatches)} (machine,nozzle) combination(s) require Step 3.5 alignment to K_per_extruder."
            )

    if failures:
        print(f"\n{len(failures)} failure(s):", file=sys.stderr)
        for f_ in failures:
            print(f"  {f_}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
