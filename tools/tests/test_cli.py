"""Tests for the ``validate`` CLI subcommand."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.colordb.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------
# Helpers.
# --------------------------------------------------------------------------


def _write_doc(tmp_path: Path, doc: dict, name: str = "doc.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return p


def _appendix_a_doc() -> dict:
    return json.loads((FIXTURES / "appendix_a.json").read_text(encoding="utf-8"))


def _valid_doc() -> dict:
    """Same minimal doc used in the validator tests."""
    return {
        "schema_version": 1,
        "name": "mini",
        "palette": [
            {"channel_class": "White", "display_name": "White", "material": "m"},
            {"channel_class": "Red", "display_name": "Red", "material": "m"},
        ],
        "defaults": {
            "color_layers": 3,
            "layer_height_mm": 0.08,
            "line_width_mm": 0.42,
            "base_layers": 10,
            "base_channel_idx": 0,
        },
        "sections": [
            {
                "type": "measured",
                "entries": [
                    {"lab": [50.0, 0.0, 0.0], "recipe": [0, 0, 0]},
                    {"lab": [40.0, 10.0, 0.0], "recipe": [1, 1, 1]},
                ],
            }
        ],
    }


# --------------------------------------------------------------------------
# Tests.
# --------------------------------------------------------------------------


class TestExitCodes:
    def test_appendix_a_exits_zero(self) -> None:
        rc = main(["validate", str(FIXTURES / "appendix_a.json")])
        assert rc == 0

    def test_must_violation_exits_one(self, tmp_path: Path) -> None:
        doc = _valid_doc()
        doc["schema_version"] = 2  # V1
        p = _write_doc(tmp_path, doc)
        rc = main(["validate", str(p)])
        assert rc == 1

    def test_should_only_exits_two(self, tmp_path: Path) -> None:
        # Duplicate recipe triggers V12 (SHOULD) but no MUST errors.
        doc = _valid_doc()
        doc["sections"][0]["entries"].append(
            {"lab": [50.0, 0.0, 0.0], "recipe": [0, 0, 0]}
        )
        p = _write_doc(tmp_path, doc)
        rc = main(["validate", str(p)])
        assert rc == 2

    def test_strict_upgrades_should_to_one(self, tmp_path: Path) -> None:
        doc = _valid_doc()
        doc["sections"][0]["entries"].append(
            {"lab": [50.0, 0.0, 0.0], "recipe": [0, 0, 0]}
        )
        p = _write_doc(tmp_path, doc)
        rc = main(["validate", "--strict", str(p)])
        assert rc == 1

    def test_strict_no_issues_still_zero(self, tmp_path: Path) -> None:
        p = _write_doc(tmp_path, _valid_doc())
        rc = main(["validate", "--strict", str(p)])
        assert rc == 0

    def test_missing_file_exits_one(self, tmp_path: Path) -> None:
        rc = main(["validate", str(tmp_path / "does-not-exist.json")])
        assert rc == 1


class TestOutputShapes:
    def test_text_output_ok_label(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        p = _write_doc(tmp_path, _valid_doc())
        main(["validate", str(p)])
        out = capsys.readouterr().out
        assert "[OK]" in out

    def test_text_output_fail_label(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        doc = _valid_doc()
        doc["schema_version"] = 99
        p = _write_doc(tmp_path, doc)
        main(["validate", str(p)])
        out = capsys.readouterr().out
        assert "[FAIL]" in out
        assert "V1" in out

    def test_text_output_warn_label(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        doc = _valid_doc()
        doc["sections"][0]["entries"].append(
            {"lab": [50.0, 0.0, 0.0], "recipe": [0, 0, 0]}
        )
        p = _write_doc(tmp_path, doc)
        main(["validate", str(p)])
        out = capsys.readouterr().out
        assert "[WARN]" in out
        assert "V12" in out

    def test_json_output_structure(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        doc = _valid_doc()
        doc["schema_version"] = 99
        p = _write_doc(tmp_path, doc)
        main(["validate", "--json", str(p)])
        out = capsys.readouterr().out
        report = json.loads(out)
        assert "results" in report
        first = report["results"][0]
        assert first["path"] == str(p)
        assert any(e["rule"] == "V1" for e in first["errors"])


class TestMultiplePaths:
    def test_one_good_one_bad_exits_one(
        self, tmp_path: Path
    ) -> None:
        good = _write_doc(tmp_path, _valid_doc(), "good.json")
        bad_doc = _valid_doc()
        bad_doc["schema_version"] = 99
        bad = _write_doc(tmp_path, bad_doc, "bad.json")
        rc = main(["validate", str(good), str(bad)])
        assert rc == 1


class TestGlob:
    def test_glob_pattern(self, tmp_path: Path) -> None:
        _write_doc(tmp_path, _valid_doc(), "a.json")
        _write_doc(tmp_path, _valid_doc(), "b.json")
        rc = main(["validate", str(tmp_path / "*.json")])
        assert rc == 0


# Bad fixtures live in tests/fixtures/bad_v<N>[_<reason>].(json|msgpack)
# and serve as golden negative samples for cross-implementation
# conformance. Each one MUST fail the CLI with exit code 1 and surface
# at least one error whose rule tag corresponds to the filename.
_BAD_FIXTURES = sorted(
    p for p in FIXTURES.iterdir()
    if p.name.startswith("bad_v") and p.suffix in {".json", ".msgpack"}
)


def _rule_from_fixture_name(name: str) -> str:
    # "bad_v1_schema_version.json" -> "V1"
    token = name.split("_")[1]
    return token.upper()


class TestBadFixtures:
    @pytest.mark.parametrize("fixture", _BAD_FIXTURES, ids=lambda p: p.name)
    def test_exit_code_one(
        self,
        capsys: pytest.CaptureFixture[str],
        fixture: Path,
    ) -> None:
        rc = main(["validate", "--json", str(fixture)])
        assert rc == 1, fixture.name

    @pytest.mark.parametrize("fixture", _BAD_FIXTURES, ids=lambda p: p.name)
    def test_expected_rule_surfaced(
        self,
        capsys: pytest.CaptureFixture[str],
        fixture: Path,
    ) -> None:
        main(["validate", "--json", str(fixture)])
        report = json.loads(capsys.readouterr().out)
        rules = {
            e["rule"] for r in report["results"] for e in r["errors"]
        }
        expected = _rule_from_fixture_name(fixture.name)
        # Some fixtures may also trigger other rules (e.g. V24 coexists
        # with V22 / V23 in edge cases). We only require the headline
        # rule tag to be present; cross-rule interaction is tested
        # separately in test_validator.py.
        assert expected in rules, (
            f"fixture {fixture.name} expected rule {expected} "
            f"in errors; got {sorted(rules)}"
        )
