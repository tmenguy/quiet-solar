"""Tests for ``scripts/qs/utils.py`` additions made by factory tasks (QS-340).

Factory-classified on purpose (``tests/qs/``): ``tests/test_qs_utils.py``
is product-classified by path and would trip the lane warning for a
factory task.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest


def test_run_env_is_merged_over_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    """``env=`` overlays the given keys on ``os.environ`` — never replaces it."""
    import utils

    monkeypatch.setenv("QS_UTILS_BASE", "kept")
    monkeypatch.setenv("QS_UTILS_OVERRIDE", "old")
    result = utils.run(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ['QS_UTILS_BASE'], "
            "os.environ['QS_UTILS_OVERRIDE'], os.environ['QS_UTILS_NEW'])",
        ],
        env={"QS_UTILS_OVERRIDE": "new", "QS_UTILS_NEW": "added"},
    )
    assert result.stdout.split() == ["kept", "new", "added"]


def test_run_without_env_passes_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default keeps today's behaviour exactly: ``env=None`` (inherit)."""
    import utils

    seen: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(utils.subprocess, "run", fake_run)
    utils.run(["true"])
    assert seen["env"] is None


@pytest.mark.parametrize(
    ("branch", "expected"),
    [
        ("QS_369", 369),
        ("QS_7", 7),
        ("QS_369_1", None),  # #398: int() accepts "369_1" as 3691
        ("QS_", None),
        ("QS_abc", None),
        ("QS_ 369", None),
        ("main", None),
    ],
)
def test_get_issue_from_branch_accepts_only_pure_digits(branch: str, expected: int | None) -> None:
    """Only ``QS_<digits>`` names an issue, as in ``quality_gate.py`` (#398)."""
    import utils

    assert utils.get_issue_from_branch(branch) == expected


# ---------------------------------------------------------------------------
# QS-400: item branch / worktree helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("branch", "expected"),
    [
        ("QS_400", (400, None)),
        ("QS_400_2", (400, 2)),
        ("QS_7_10", (7, 10)),
        ("QS_400_0", None),
        ("QS_400_02", None),
        ("QS_400_x", None),
        ("QS_400_integration", None),
        ("QS_400_2_integration", None),
        ("QS_", None),
        ("QS_x2", None),
        ("QS_400\n", None),
        ("QS_400_2\n", None),
        ("QS_٣", None),
        ("main", None),
        ("", None),
    ],
)
def test_parse_task_branch(branch: str, expected: tuple[int, int | None] | None) -> None:
    """``QS_<N>`` and ``QS_<N>_<k>`` only; every other form is ``None`` (QS-400 D1)."""
    import utils

    assert utils.parse_task_branch(branch) == expected


@pytest.mark.parametrize(
    ("issue", "item", "expected"),
    [(400, None, "QS_400"), (400, 2, "QS_400_2"), (7, 10, "QS_7_10")],
)
def test_task_branch_name(issue: int, item: int | None, expected: str) -> None:
    """The one place outside the Control Plane where the name is spelled."""
    import utils

    assert utils.task_branch_name(issue, item) == expected


def test_task_branch_name_default_is_the_task_branch() -> None:
    import utils

    assert utils.task_branch_name(12) == "QS_12"


def _fake_main(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    from pathlib import Path

    import utils

    monkeypatch.setattr(utils, "get_main_worktree", lambda: Path(path))


def test_get_worktree_dir_task_path_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    import utils

    _fake_main(monkeypatch, "/x/repo")
    assert utils.get_worktree_dir(42) == Path("/x/repo-worktrees/QS_42")
    assert utils.get_worktree_dir(42, item=None) == Path("/x/repo-worktrees/QS_42")


def test_get_worktree_dir_item(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    import utils

    _fake_main(monkeypatch, "/x/repo")
    assert utils.get_worktree_dir(42, item=3) == Path("/x/repo-worktrees/QS_42_3")


def test_get_integration_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    import utils

    _fake_main(monkeypatch, "/x/repo")
    assert utils.get_integration_dir(42, 3) == Path("/x/repo-worktrees/QS_42_3_integration")


@pytest.mark.parametrize(("raw", "expected"), [("1", 1), ("9", 9), ("10", 10), ("123", 123)])
def test_positive_int_accepts(raw: str, expected: int) -> None:
    import utils

    assert utils.positive_int(raw) == expected


@pytest.mark.parametrize("raw", ["0", "-1", "01", "+1", "1_0", " 1", "1 ", "٣", "x", ""])
def test_positive_int_rejects(raw: str) -> None:
    import argparse

    import utils

    with pytest.raises(argparse.ArgumentTypeError):
        utils.positive_int(raw)
