"""Tests for ``scripts/qs/models.py`` — the per-phase model policy (QS-358)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = REPO_ROOT / "scripts" / "qs" / "agent_templates"

# Module-scope import needs ``scripts/qs`` on ``sys.path`` before the
# per-test conftest fixture fires (same idiom as ``test_render_agents.py``).
sys.path.insert(0, str(REPO_ROOT / "scripts" / "qs"))

import models  # type: ignore[import-not-found]  # noqa: E402
import render_agents  # type: ignore[import-not-found]  # noqa: E402
import targets  # type: ignore[import-not-found]  # noqa: E402
from launchers.phases import PHASE_TO_AGENT  # type: ignore[import-not-found]  # noqa: E402

_ODD_LANES = (None, "", "not-a-lane", "feature-nope")
_EFFORT_LEVELS = {"low", "medium", "high", "xhigh", "max"}
_CLAUDE_ALIASES = {"opus", "fable", "sonnet", "haiku"}

_FLAT_TABLE = [
    ("qs-plan-critic", "deep"),
    ("qs-plan-concrete-planner", "deep"),
    ("qs-plan-dev-proxy", "build"),
    ("qs-plan-scope-guardian", "frontier"),
    ("qs-plan-delta-auditor", "light"),
    ("qs-diag-root-cause-skeptic", "deep"),
    ("qs-diag-fix-minimalist", "frontier"),
    ("qs-implement-task", "build"),
    ("qs-implement-setup-task", "build"),
    ("qs-review-task", "frontier"),
    ("qs-verify-task", "frontier"),
    ("qs-review-blind-hunter", "deep"),
    ("qs-review-edge-case-hunter", "deep"),
    ("qs-review-acceptance-auditor", "frontier"),
    ("qs-review-regression-proof", "frontier"),
    ("qs-review-coderabbit", "fast"),
    ("qs-finish-task", "fast"),
    ("qs-setup-task", "light"),
    ("qs-release", "fast"),
]

_ROW_LITERALS = [
    ("claude", "build", "claude-opus-4-8"),
    ("claude", "deep", "claude-opus-5-5"),
    ("claude", "frontier", "claude-fable-5-1"),
    ("claude", "light", "claude-sonnet-5"),
    ("claude", "fast", "claude-haiku-4-5"),
    ("opencode", "build", "github-copilot/claude-opus-4.8"),
    ("opencode", "deep", "github-copilot/claude-opus-5.5"),
    ("opencode", "frontier", "github-copilot/gpt-6-astra"),
    ("opencode", "light", "github-copilot/claude-sonnet-5"),
    ("opencode", "fast", "github-copilot/claude-haiku-4.5"),
]


# --- resolve() -------------------------------------------------------------


@pytest.mark.parametrize("lane", (*models.LANES, *_ODD_LANES))
def test_every_stem_resolves_to_a_class(lane: str | None) -> None:
    for stem in models.STEMS:
        assert models.resolve(lane, stem) in models.CLASSES


def test_classes_exact() -> None:
    assert {"build", "deep", "frontier", "light", "fast"} == models.CLASSES
    assert "inherit" not in models.CLASSES


def test_stems_match_registry_and_templates() -> None:
    assert len(models.STEMS) == 21
    assert models.STEMS == render_agents.ORCHESTRATORS | render_agents.SUBAGENTS
    assert set(render_agents._discover_stems(TEMPLATES_DIR)) == models.STEMS


def test_phase_agents_have_policy_rows() -> None:
    assert set(PHASE_TO_AGENT.values()) <= models.STEMS


@pytest.mark.parametrize("stem", ["qs-create-plan", "qs-diagnose-task"])
def test_planning_kind_gate(stem: str) -> None:
    for lane in ("bug-product", "bug-factory"):
        assert models.resolve(lane, stem) == "deep"
    for lane in ("feature-product", "feature-factory", "epic-product", "epic-factory"):
        assert models.resolve(lane, stem) == "frontier"
    for lane in _ODD_LANES:
        assert models.resolve(lane, stem) == "deep"


def test_flat_rows_are_lane_invariant() -> None:
    for stem, _cls in _FLAT_TABLE:
        seen = {models.resolve(lane, stem) for lane in (*models.LANES, None)}
        assert len(seen) == 1, stem


@pytest.mark.parametrize(("stem", "cls"), _FLAT_TABLE)
def test_flat_table_literal(stem: str, cls: str) -> None:
    assert models.resolve("feature-factory", stem) == cls
    assert models.resolve(None, stem) == cls


def test_flat_table_covers_every_non_planning_stem() -> None:
    assert {s for s, _ in _FLAT_TABLE} == models.STEMS - {"qs-create-plan", "qs-diagnose-task"}


def test_unknown_stem_raises() -> None:
    with pytest.raises(models.ModelPolicyError, match="qs-nope"):
        models.resolve(None, "qs-nope")
    assert issubclass(models.ModelPolicyError, ValueError)


def test_lanes_derivation_and_lane_files() -> None:
    expected = tuple(f"{k}-{t}" for k in targets.KINDS for t in targets.TARGETS) + tuple(
        f"epic-{t}" for t in targets.TARGETS
    )
    assert expected == models.LANES
    lane_files = {p.stem for p in (REPO_ROOT / "docs" / "workflow" / "lanes").glob("*.md")}
    assert set(models.LANES) == lane_files


# --- harness rows ----------------------------------------------------------


def test_harness_rows_complete() -> None:
    assert set(models.HARNESS_MODELS) == set(render_agents._HARNESSES)
    for harness, row in models.HARNESS_MODELS.items():
        assert set(row) == models.CLASSES, harness


@pytest.mark.parametrize(("harness", "cls", "literal"), _ROW_LITERALS)
def test_row_literals(harness: str, cls: str, literal: str) -> None:
    assert models.model_for(harness, cls) == literal


def test_claude_row_has_no_bare_alias() -> None:
    for value in models.HARNESS_MODELS["claude"].values():
        assert value not in _CLAUDE_ALIASES


def test_model_for_rejects_unknowns() -> None:
    with pytest.raises(KeyError):
        models.model_for("claude", "inherit")
    with pytest.raises(KeyError):
        models.model_for("codex", "deep")


# --- class_of ----------------------------------------------------------------


def test_class_of() -> None:
    for cls in models.CLASSES:
        assert models.class_of(cls) == cls
    for value in ("opus", "claude-opus-4-8", "github-copilot/x", "inherit"):
        assert models.class_of(value) is None


# --- effort (D19) -------------------------------------------------------------


def test_class_effort_complete_and_valid() -> None:
    assert set(models.CLASS_EFFORT) == models.CLASSES
    for value in models.CLASS_EFFORT.values():
        assert value is None or value in _EFFORT_LEVELS


def test_class_effort_literals() -> None:
    assert models.CLASS_EFFORT == {
        "build": "high",
        "deep": "high",
        "frontier": "high",
        "light": "medium",
        "fast": None,
    }
    assert models.effort_for("fast") is None
    assert models.effort_for("light") == "medium"
    with pytest.raises(KeyError):
        models.effort_for("nope")


def test_planning_classes_share_effort() -> None:
    # D16 tripwire: the day these diverge, the ``lane`` passed to the
    # launcher's pin writer becomes load-bearing — update knowingly.
    assert models.CLASS_EFFORT["deep"] == models.CLASS_EFFORT["frontier"]


# --- lockstep (D13/D20) --------------------------------------------------------


def test_opencode_json_follows_deep_row() -> None:
    data = json.loads((REPO_ROOT / "opencode.json").read_text(encoding="utf-8"))
    assert data["model"] == models.HARNESS_MODELS["opencode"]["deep"], (
        "opencode.json 'model' must equal models.HARNESS_MODELS['opencode']['deep']"
    )


def test_settings_json_has_no_alias_env_pins() -> None:
    data = json.loads((REPO_ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    env = data.get("env", {})
    for family in ("OPUS", "FABLE", "SONNET", "HAIKU"):
        key = f"ANTHROPIC_DEFAULT_{family}_MODEL"
        assert key not in env, (
            f"D20 — alias pins via settings env ({key}) are dead on Claude Code "
            "2.1.278 and the policy pins full IDs; do not re-add"
        )
