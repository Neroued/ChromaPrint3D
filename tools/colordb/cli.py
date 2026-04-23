"""
Command-line entry point for the reference implementation.

Only the ``validate`` subcommand is offered; the plan explicitly
scopes out ``convert`` / ``info`` / ``merge`` / ``fallback`` as
Python-API-only features.

Exit codes:

- ``0`` all inputs pass every MUST rule (SHOULD warnings may be
  present and are surfaced in the output).
- ``1`` at least one MUST rule failed on at least one input. Also
  returned under ``--strict`` when any SHOULD warning exists.
- ``2`` every input passes all MUST rules and at least one SHOULD
  warning exists. Only used when ``--strict`` is **not** set.

Output format:

- Default text report, one line per issue, grouped by input path.
- ``--json`` emits a single JSON object of the form
  ``{"results": [{"path": str, "errors": [...], "warnings": [...]}, ...]}``.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

from .codec import load_doc
from .validator import ColorDBValidationError, ValidationReport, validate

__all__ = ["main"]


def _expand_paths(raw_paths: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for pat in raw_paths:
        matches = glob.glob(pat, recursive=True)
        if matches:
            out.extend(Path(m) for m in sorted(matches))
        else:
            # If glob produces nothing, fall through to the literal
            # path so that the loader surfaces a FileNotFoundError the
            # user can act on.
            out.append(Path(pat))
    return out


def _run_validate(
    paths: Iterable[Path],
    *,
    strict: bool,
    as_json: bool,
) -> int:
    results = []
    any_errors = False
    any_warnings = False

    for path in paths:
        try:
            buf = path.read_bytes()
            doc, _ = load_doc(buf)
            report = validate(doc)
        except FileNotFoundError as e:
            any_errors = True
            results.append(
                {
                    "path": str(path),
                    "errors": [
                        {
                            "rule": "S-io",
                            "severity": "MUST",
                            "path": "$",
                            "message": str(e),
                        }
                    ],
                    "warnings": [],
                }
            )
            continue
        except ColorDBValidationError as e:
            # load_doc raises this wrapping the codec-level error with
            # S-json-parse / S-msgpack-parse rules; preserve the report
            # so users see the exact codec failure reason.
            any_errors = True
            results.append(_report_to_dict(str(path), e.report))
            continue
        except Exception as e:  # noqa: BLE001 - unexpected codec errors
            any_errors = True
            results.append(
                {
                    "path": str(path),
                    "errors": [
                        {
                            "rule": "S-parse",
                            "severity": "MUST",
                            "path": "$",
                            "message": f"{type(e).__name__}: {e}",
                        }
                    ],
                    "warnings": [],
                }
            )
            continue

        results.append(_report_to_dict(str(path), report))
        if report.errors:
            any_errors = True
        if report.warnings:
            any_warnings = True

    if as_json:
        json.dump(
            {"results": results},
            sys.stdout,
            ensure_ascii=False,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        _print_text_report(results)

    if any_errors:
        return 1
    if any_warnings:
        return 1 if strict else 2
    return 0


def _report_to_dict(path: str, report: ValidationReport) -> dict:
    return {
        "path": path,
        "errors": [issue.to_dict() for issue in report.errors],
        "warnings": [issue.to_dict() for issue in report.warnings],
    }


def _print_text_report(results: Sequence[dict]) -> None:
    for r in results:
        errors = r["errors"]
        warnings = r["warnings"]
        status = "OK"
        if errors:
            status = "FAIL"
        elif warnings:
            status = "WARN"
        print(f"[{status}] {r['path']}")
        for issue in errors:
            print(
                f"  ERROR  {issue['rule']}  {issue['path']}  {issue['message']}"
            )
        for issue in warnings:
            print(
                f"  WARN   {issue['rule']}  {issue['path']}  {issue['message']}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    """Program entry point, returning an exit code."""
    parser = argparse.ArgumentParser(
        prog="tools.colordb",
        description="ColorDB spec v1 reference validator",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser(
        "validate",
        help="Validate one or more .colordb files against spec v1 V1-V25",
    )
    p_validate.add_argument(
        "paths",
        nargs="+",
        help="Paths (or glob patterns) to .colordb files to validate",
    )
    p_validate.add_argument(
        "--strict",
        action="store_true",
        help="Treat SHOULD warnings as errors (exit 1 instead of 2)",
    )
    p_validate.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit a machine-readable JSON report on stdout",
    )

    args = parser.parse_args(argv)
    if args.command == "validate":
        return _run_validate(
            _expand_paths(args.paths),
            strict=args.strict,
            as_json=args.as_json,
        )
    parser.error(f"unknown command: {args.command!r}")
    return 2  # unreachable
