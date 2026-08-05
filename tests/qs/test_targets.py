"""Tests for ``scripts/qs/targets.py`` — the lane domain module (QS-332).

One truth table, three consumers (``fetch_issue.py``, ``setup_task.py``,
the quality gate). These tests pin the shared table itself; per-caller
tests only cover wiring (review SG-R1).
"""

from __future__ import annotations

import pytest


@pytest.fixture(name="targets")
def targets_fixture():
    """Import inside the fixture: ``tests/qs/conftest.py``'s autouse
    fixture inserts ``scripts/qs/`` on ``sys.path`` at fixture time and
    purges the module on teardown (same pattern as ``test_context.py``).
    """
    import targets as mod

    return mod


# ---------------------------------------------------------------------------
# classify() — the D5 classification, fully enumerated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # --- factory prefixes ---
        ("scripts/qs/quality_gate.py", "factory"),
        ("scripts/worktree-setup.sh", "factory"),
        ("tests/qs/test_context.py", "factory"),
        ("tests/qs/agents/test_lane_steps_parity.py", "factory"),
        (".claude/agents/qs-setup-task.md", "factory"),
        (".cursor/agents/qs-setup-task.md", "factory"),
        (".opencode/agents/qs-setup-task.md", "factory"),
        (".github/workflows/pr-quality.yml", "factory"),
        ("legacy/old_pipeline.py", "factory"),
        ("docs/workflow/project-rules.md", "factory"),
        ("docs/workflow/lanes/bug-product.md", "factory"),
        # --- factory carve-out by the purpose rule (review CR-1) ---
        ("tests/test_quality_gate.py", "factory"),
        # --- factory basenames (top-level only) ---
        ("CLAUDE.md", "factory"),
        ("AGENTS.md", "factory"),
        (".cursorrules", "factory"),
        (".gitignore", "factory"),
        ("pyproject.toml", "factory"),
        ("setup.cfg", "factory"),
        ("requirements.txt", "factory"),
        ("requirements_test.txt", "factory"),
        # --- product prefixes ---
        ("custom_components/quiet_solar/solver.py", "product"),
        ("custom_components/quiet_solar/ui/dashboard.py", "product"),
        ("tests/test_solver.py", "product"),
        ("tests/ha_tests/test_home.py", "product"),
        ("docs/agents/concepts/solver.md", "product"),
        ("docs/product/roadmap.md", "product"),
        # --- enumerated neutral (silent) ---
        ("docs/stories/QS-332.story.md", "neutral"),
        ("docs/epics/QS-321.md", "neutral"),
        ("README.md", "neutral"),
        # --- unknown fallthrough (FYI-printed by the gate) ---
        ("hacs.json", "unknown"),
        ("some/random/path.txt", "unknown"),
        ("info.md", "unknown"),
        # a nested file matching a factory BASENAME is not top-level
        ("somewhere/pyproject.toml", "unknown"),
        ("somewhere/README.md", "unknown"),
    ],
)
def test_classify(targets, path: str, expected: str) -> None:
    assert targets.classify(path) == expected


def test_classify_tests_qs_beats_product_tests(targets) -> None:
    """``tests/qs/`` is factory even though ``tests/`` is product."""
    assert targets.classify("tests/qs/test_targets.py") == "factory"
    assert targets.classify("tests/test_constraints.py") == "product"


# ---------------------------------------------------------------------------
# parse_axes()
# ---------------------------------------------------------------------------


def test_parse_axes_task_lane(targets) -> None:
    axes = targets.parse_axes(["kind:bug", "target:product", "scale:task"])
    assert axes == {
        "kind": "bug",
        "target": "product",
        "scale": "task",
        "lane": "bug-product",
    }


def test_parse_axes_epic_lane(targets) -> None:
    axes = targets.parse_axes(["scale:epic", "target:factory", "pinned"])
    assert axes == {
        "kind": "",
        "target": "factory",
        "scale": "epic",
        "lane": "epic-factory",
    }


def test_parse_axes_unlabelled_is_empty(targets) -> None:
    axes = targets.parse_axes(["bug", "area:solver"])
    assert axes == {"kind": "", "target": "", "scale": "", "lane": ""}


def test_parse_axes_conflicting_axis_is_empty(targets) -> None:
    """Two labels on the same axis → that axis is ambiguous → empty."""
    axes = targets.parse_axes(
        ["target:product", "target:factory", "kind:bug", "scale:task"]
    )
    assert axes["target"] == ""
    assert axes["lane"] == ""


def test_parse_axes_incomplete_has_no_lane(targets) -> None:
    axes = targets.parse_axes(["kind:feature", "target:factory"])
    assert axes["lane"] == ""


# ---------------------------------------------------------------------------
# validate_declaration() — THE truth table (D1)
# ---------------------------------------------------------------------------


def test_validate_declaration_complete_task(targets) -> None:
    ok, missing, message = targets.validate_declaration(
        ["kind:feature", "target:factory", "scale:task", "enhancement"]
    )
    assert ok
    assert missing == []
    assert message == ""


def test_validate_declaration_complete_epic(targets) -> None:
    """An epic-shaped declaration validates as itself — no kind expected."""
    ok, missing, message = targets.validate_declaration(
        ["scale:epic", "target:product"]
    )
    assert ok
    assert missing == []
    assert message == ""


def test_validate_declaration_epic_with_kind_is_invalid(targets) -> None:
    ok, _missing, message = targets.validate_declaration(
        ["scale:epic", "target:product", "kind:bug"]
    )
    assert not ok
    assert "kind" in message


def test_validate_declaration_empty_labels(targets) -> None:
    ok, missing, message = targets.validate_declaration(["bug"])
    assert not ok
    assert set(missing) == {"kind", "target", "scale"}
    # The message carries the shape-aware backfill command with an <N>
    # placeholder callers substitute with the issue number.
    assert "gh issue edit <N> --add-label" in message
    assert "scale:task" in message


def test_validate_declaration_partial_reports_only_missing_axes(targets) -> None:
    ok, missing, message = targets.validate_declaration(
        ["kind:bug", "scale:task"]
    )
    assert not ok
    assert missing == ["target"]
    assert "target:product" in message
    assert "kind:" not in message.split("--add-label")[-1]


def test_validate_declaration_two_targets_is_invalid(targets) -> None:
    ok, missing, _message = targets.validate_declaration(
        ["kind:bug", "target:product", "target:factory", "scale:task"]
    )
    assert not ok
    assert "target" in missing


def test_validate_declaration_two_kinds_is_invalid(targets) -> None:
    ok, missing, _message = targets.validate_declaration(
        ["kind:bug", "kind:feature", "target:product", "scale:task"]
    )
    assert not ok
    assert "kind" in missing


def test_validate_declaration_two_scales_is_invalid(targets) -> None:
    ok, missing, _message = targets.validate_declaration(
        ["kind:bug", "target:product", "scale:task", "scale:epic"]
    )
    assert not ok
    assert "scale" in missing


def test_validate_declaration_epic_shape_message_has_no_kind(targets) -> None:
    """Shape-aware: an epic missing its target is not asked to grow a kind."""
    ok, missing, message = targets.validate_declaration(["scale:epic"])
    assert not ok
    assert missing == ["target"]
    assert "kind:" not in message


# ---------------------------------------------------------------------------
# parse_parent_epic()
# ---------------------------------------------------------------------------


def test_parse_parent_epic_structured_section(targets) -> None:
    body = "intro\n\n### Parent epic\n\n321\n\nmore text\n"
    assert targets.parse_parent_epic(body) == 321


def test_parse_parent_epic_structured_with_hash(targets) -> None:
    body = "### Parent epic\n#321\n"
    assert targets.parse_parent_epic(body) == 321


def test_parse_parent_epic_structured_beats_refs(targets) -> None:
    """The structured field wins over free-text cross-refs (review PC-05)."""
    body = "Refs #999\n\n### Parent epic\n\n321\n"
    assert targets.parse_parent_epic(body) == 321


def test_parse_parent_epic_refs_fallback(targets) -> None:
    body = "some description\n\nRefs #321\n"
    assert targets.parse_parent_epic(body) == 321


def test_parse_parent_epic_no_response_falls_back_to_refs(targets) -> None:
    body = "### Parent epic\n\n_No response_\n\nRefs #321\n"
    assert targets.parse_parent_epic(body) == 321


def test_parse_parent_epic_none(targets) -> None:
    assert targets.parse_parent_epic("plain body, no epic") is None
    assert targets.parse_parent_epic("") is None


# ---------------------------------------------------------------------------
# __main__ classification query (AC-10 / review DP2-08)
# ---------------------------------------------------------------------------


def test_main_classifies_paths(targets, capsys: pytest.CaptureFixture[str]) -> None:
    targets.main(["scripts/qs/targets.py", "custom_components/x.py", "hacs.json"])
    out = capsys.readouterr().out.splitlines()
    assert out == [
        "factory\tscripts/qs/targets.py",
        "product\tcustom_components/x.py",
        "unknown\thacs.json",
    ]


def test_main_no_args_errors(targets) -> None:
    with pytest.raises(SystemExit):
        targets.main([])
