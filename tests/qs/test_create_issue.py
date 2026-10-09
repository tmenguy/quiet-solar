"""Tests for ``scripts/qs/create_issue.py`` — the ``--labels`` passthrough.

QS-332 (review R2-06): the passthrough existed untested; the setup
agent's declare-at-birth path (D3 path 3) now depends on it, so it gets
its first pin. No script change was needed — these are characterization
tests.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

_URL = "https://github.com/tmenguy/quiet-solar/issues/42\n"


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> tuple[list[str], dict[str, Any]]:
    import create_issue
    import utils

    seen: list[str] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.extend(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=_URL, stderr="")

    monkeypatch.setattr(utils, "run", fake_run)
    monkeypatch.setattr("sys.argv", ["create_issue.py", *argv])
    create_issue.main()
    return seen, json.loads(capsys.readouterr().out)


def test_labels_passthrough_reaches_gh(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cmd, out = _run_main(
        monkeypatch,
        capsys,
        ["--title", "t", "--body", "b", "--labels", "kind:bug,target:product,scale:task"],
    )
    label_idx = cmd.index("--label")
    assert cmd[label_idx + 1] == "kind:bug,target:product,scale:task"
    assert out["issue_number"] == 42
    assert out["branch"] == "QS_42"


def test_no_labels_means_no_label_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cmd, _out = _run_main(monkeypatch, capsys, ["--title", "t"])
    assert "--label" not in cmd
