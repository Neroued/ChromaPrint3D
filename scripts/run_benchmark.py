#!/usr/bin/env python3
"""Vectorization benchmark runner.

Reads a manifest of test images, runs the raster_to_svg CLI on each,
then runs analyze_fragmentation.py for quality metrics.  Results are
compared against baselines, scored, and tracked in history.csv.
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ANALYZE_SCRIPT = SCRIPT_DIR / "analyze_fragmentation.py"
DEFAULT_TEST_DIR = PROJECT_ROOT / "test_data"
DEFAULT_CLI = PROJECT_ROOT / "build" / "bin" / "raster_to_svg"

SCORE_WEIGHTS = {
    "merge": 10,
    "tiny": 20,
    "coverage": 50,
    "delta_e": 2,
}

REGRESSION_THRESHOLD = 1.0
IMPROVEMENT_THRESHOLD = 1.0


# ── helpers ──────────────────────────────────────────────────────────────────


def git_short_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=PROJECT_ROOT,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def make_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{git_short_hash()}"


def load_manifest(test_dir: Path) -> list[dict]:
    manifest_path = test_dir / "images" / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"Manifest not found: {manifest_path}")
    with open(manifest_path) as f:
        data = json.load(f)
    images = data.get("images", [])
    if not images:
        sys.exit("Manifest contains no images.")
    return images


# ── scoring ──────────────────────────────────────────────────────────────────


def compute_score(m: dict) -> float:
    """Weighted aggregate score (100 = perfect)."""
    score = 100.0
    score -= SCORE_WEIGHTS["merge"] * max(0, m.get("mergeable_ratio", 1.0) - 1)
    score -= SCORE_WEIGHTS["tiny"] * m.get("tiny_fragment_rate", 0.0)
    score -= SCORE_WEIGHTS["coverage"] * max(0, 1.0 - m.get("coverage", 1.0))
    score -= SCORE_WEIGHTS["delta_e"] * m.get("delta_e_mean", 0.0)
    return max(0.0, round(score, 2))


# ── expectations ─────────────────────────────────────────────────────────────


def check_expectations(metrics: dict, expectations: dict) -> list[str]:
    failures = []
    if not expectations:
        return failures
    if "max_shapes" in expectations:
        n = metrics.get("total_shapes", 0)
        if n > expectations["max_shapes"]:
            failures.append(f"shapes {n} > max {expectations['max_shapes']}")
    if "min_coverage" in expectations:
        c = metrics.get("coverage", 0)
        if c < expectations["min_coverage"]:
            failures.append(f"coverage {c:.4f} < min {expectations['min_coverage']}")
    if "max_delta_e_mean" in expectations:
        d = metrics.get("delta_e_mean", 0)
        if d > expectations["max_delta_e_mean"]:
            failures.append(f"ΔE_mean {d:.2f} > max {expectations['max_delta_e_mean']}")
    return failures


# ── external tool calls ─────────────────────────────────────────────────────


def run_vectorize(
    cli_path: Path,
    image_path: Path,
    svg_output: Path,
    per_image_params: dict,
    cli_overrides: dict,
) -> None:
    cmd = [
        str(cli_path),
        "--image",
        str(image_path),
        "--out",
        str(svg_output),
        "--log-level",
        "warn",
    ]
    merged = {**per_image_params, **cli_overrides}
    for k, v in merged.items():
        cmd.extend([f"--{k.replace('_', '-')}", str(v)])
    print(f"  vectorize: {' '.join(cmd[-4:])}")
    subprocess.run(cmd, check=True, timeout=120)


def run_analyze(
    svg_path: Path,
    original_path: Path,
    json_path: Path,
    report_path: Path,
    resolution: int = 800,
) -> dict:
    cmd = [
        sys.executable,
        str(ANALYZE_SCRIPT),
        str(svg_path),
        "--original",
        str(original_path),
        "--json",
        str(json_path),
        "--save",
        str(report_path),
        "--resolution",
        str(resolution),
    ]
    subprocess.run(cmd, check=True, timeout=300, capture_output=True)
    with open(json_path) as f:
        return json.load(f)


# ── baseline management ─────────────────────────────────────────────────────


def set_baseline(result_dir: Path, baseline_dir: Path, names: list[str]) -> None:
    baseline_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in names:
        for ext in [".json", ".svg"]:
            src = result_dir / f"{name}{ext}"
            if src.exists():
                shutil.copy2(src, baseline_dir / f"{name}{ext}")
                copied += 1
    print(f"\nBaseline updated: {copied} files → {baseline_dir}")


def compare_with_baseline(
    baseline_dir: Path, results: dict[str, dict]
) -> dict[str, tuple[str, float | None, float | None]]:
    verdicts: dict[str, tuple[str, float | None, float | None]] = {}
    for name, info in results.items():
        base_json = baseline_dir / f"{name}.json"
        curr_score = info["score"]
        if not base_json.exists():
            verdicts[name] = ("NEW", None, curr_score)
            continue
        with open(base_json) as f:
            base_metrics = json.load(f)
        base_score = compute_score(base_metrics)
        diff = curr_score - base_score
        if diff > IMPROVEMENT_THRESHOLD:
            verdict = "IMPROVED"
        elif diff < -REGRESSION_THRESHOLD:
            verdict = "REGRESSED"
        else:
            verdict = "OK"
        verdicts[name] = (verdict, base_score, curr_score)
    return verdicts


# ── terminal report ──────────────────────────────────────────────────────────


def print_report(
    run_id: str,
    images: list[dict],
    results: dict[str, dict],
    verdicts: dict[str, tuple] | None,
) -> None:
    header = f"{'Category':<10} {'Image':<16} {'Shapes':>6} {'MergeR':>7} {'TinyF%':>7} {'Cvg%':>7} {'ΔE_mean':>8} {'Score':>6}"
    if verdicts:
        header += f" {'Verdict':>12}"

    print(f"\n{'=' * 3} Vectorize Benchmark: {run_id} {'=' * 3}")
    print(header)
    print("─" * len(header))

    ok = improved = regressed = failed = new_count = 0
    for img in images:
        name = img["name"]
        cat = img.get("category", "?")
        info = results.get(name)
        if not info:
            continue
        m = info["metrics"]
        score = info["score"]
        fails = info.get("expectation_failures", [])

        line = (
            f"{cat:<10} {name:<16} "
            f"{m.get('total_shapes', 0):>6} "
            f"{m.get('mergeable_ratio', 0):>7.2f} "
            f"{m.get('tiny_fragment_rate', 0):>6.1%} "
            f"{m.get('coverage', 0):>6.2%} "
            f"{m.get('delta_e_mean', 0):>8.1f} "
            f"{score:>6.1f}"
        )
        if verdicts and name in verdicts:
            v, _, _ = verdicts[name]
            marker = ""
            if v == "REGRESSED":
                marker = " \u26a0"
                regressed += 1
            elif v == "IMPROVED":
                marker = " \u2714"
                improved += 1
            elif v == "NEW":
                new_count += 1
            else:
                ok += 1
            line += f" {v:>10}{marker}"
        if fails:
            line += f"  FAIL: {'; '.join(fails)}"
            failed += 1

        print(line)

    total = len(results)
    print(f"\n--- Summary ---")
    parts = [f"Total: {total} images"]
    if verdicts:
        parts.extend(
            [
                f"OK: {ok}",
                f"IMPROVED: {improved}",
                f"REGRESSED: {regressed}",
                f"NEW: {new_count}",
            ]
        )
    if failed:
        parts.append(f"FAIL: {failed}")
    print(" | ".join(parts))

    scores = [info["score"] for info in results.values()]
    avg = sum(scores) / len(scores) if scores else 0
    print(f"Aggregate score: {avg:.1f}")

    if verdicts:
        base_scores = [
            verdicts[n][1] for n in verdicts if verdicts[n][1] is not None
        ]
        if base_scores:
            base_avg = sum(base_scores) / len(base_scores)
            delta = avg - base_avg
            sign = "+" if delta >= 0 else ""
            print(f"  Baseline avg: {base_avg:.1f} → Current: {avg:.1f} ({sign}{delta:.1f})")


# ── history tracking ─────────────────────────────────────────────────────────


def update_history(
    history_path: Path,
    run_id: str,
    results: dict[str, dict],
    note: str,
) -> None:
    scores = [info["score"] for info in results.values()]
    avg_score = sum(scores) / len(scores) if scores else 0

    all_m = [info["metrics"] for info in results.values()]
    avg_merge = (
        sum(m.get("mergeable_ratio", 1) for m in all_m) / len(all_m) if all_m else 0
    )
    avg_cov = sum(m.get("coverage", 1) for m in all_m) / len(all_m) if all_m else 0
    avg_de = sum(m.get("delta_e_mean", 0) for m in all_m) / len(all_m) if all_m else 0

    row = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_short_hash(),
        "images": len(results),
        "score_avg": f"{avg_score:.1f}",
        "merge_ratio_avg": f"{avg_merge:.2f}",
        "coverage_avg": f"{avg_cov:.4f}",
        "delta_e_avg": f"{avg_de:.1f}",
        "note": note,
    }

    fieldnames = list(row.keys())
    write_header = not history_path.exists()
    with open(history_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"History updated: {history_path}")


# ── main pipeline ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run vectorization benchmark suite"
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=DEFAULT_TEST_DIR,
        help="Root test_data directory (default: test_data/)",
    )
    parser.add_argument(
        "--cli",
        type=Path,
        default=DEFAULT_CLI,
        help="Path to raster_to_svg binary",
    )
    parser.add_argument(
        "--category",
        help="Only run images in this category",
    )
    parser.add_argument(
        "--set-baseline",
        action="store_true",
        help="Copy results to baselines/current/ after run",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override vectorization param (e.g. --param colors=24)",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Note appended to history.csv entry",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=800,
        help="Analysis rasterization resolution (default: 800)",
    )
    args = parser.parse_args()

    test_dir: Path = args.test_dir.resolve()
    cli_path: Path = args.cli.resolve()
    if not cli_path.exists():
        sys.exit(f"CLI binary not found: {cli_path}")

    cli_overrides = {}
    for p in args.param:
        if "=" not in p:
            sys.exit(f"Invalid --param format '{p}', expected KEY=VALUE")
        k, v = p.split("=", 1)
        cli_overrides[k] = v

    images = load_manifest(test_dir)
    if args.category:
        images = [i for i in images if i.get("category") == args.category]
        if not images:
            sys.exit(f"No images found in category '{args.category}'")

    run_id = make_run_id()
    result_dir = test_dir / "results" / run_id
    result_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = test_dir / "baselines" / "current"
    history_path = test_dir / "history.csv"

    print(f"Benchmark run: {run_id}")
    print(f"Images: {len(images)} | Test dir: {test_dir}")
    if cli_overrides:
        print(f"Param overrides: {cli_overrides}")
    print()

    results: dict[str, dict] = {}

    for i, img in enumerate(images, 1):
        name = img["name"]
        rel_file = img["file"]
        image_path = test_dir / "images" / rel_file
        if not image_path.exists():
            print(f"[{i}/{len(images)}] {name}: SKIP (file not found: {image_path})")
            continue

        svg_output = result_dir / f"{name}.svg"
        json_output = result_dir / f"{name}.json"
        report_output = result_dir / f"{name}_report.png"

        print(f"[{i}/{len(images)}] {name}")

        try:
            run_vectorize(
                cli_path,
                image_path,
                svg_output,
                img.get("vectorize_params", {}),
                cli_overrides,
            )
        except subprocess.CalledProcessError as e:
            print(f"  VECTORIZE FAILED (exit {e.returncode})")
            continue
        except subprocess.TimeoutExpired:
            print("  VECTORIZE TIMEOUT")
            continue

        try:
            metrics = run_analyze(
                svg_output, image_path, json_output, report_output, args.resolution
            )
        except subprocess.CalledProcessError as e:
            print(f"  ANALYZE FAILED (exit {e.returncode})")
            if json_output.exists():
                json_output.unlink()
            continue
        except subprocess.TimeoutExpired:
            print("  ANALYZE TIMEOUT")
            continue

        score = compute_score(metrics)
        failures = check_expectations(metrics, img.get("expectations", {}))
        results[name] = {
            "metrics": metrics,
            "score": score,
            "expectation_failures": failures,
        }
        status = "OK" if not failures else f"FAIL: {'; '.join(failures)}"
        print(f"  score={score:.1f}  shapes={metrics.get('total_shapes', '?')}  {status}")

    if not results:
        sys.exit("No images processed successfully.")

    has_baseline = baseline_dir.exists() and any(baseline_dir.glob("*.json"))
    verdicts = compare_with_baseline(baseline_dir, results) if has_baseline else None

    print_report(run_id, images, results, verdicts)
    note = args.note or ("baseline" if args.set_baseline else "")
    update_history(history_path, run_id, results, note)

    if args.set_baseline:
        set_baseline(result_dir, baseline_dir, list(results.keys()))

    if verdicts:
        reg_count = sum(1 for v, _, _ in verdicts.values() if v == "REGRESSED")
        if reg_count > 0:
            print(f"\n\u26a0  {reg_count} image(s) regressed vs baseline!")
            sys.exit(2)


if __name__ == "__main__":
    main()
