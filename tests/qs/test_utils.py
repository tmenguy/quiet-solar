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
