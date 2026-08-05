"""Tests for ``scripts/qs/fetch_issue.py`` — axes wiring (QS-332).

The declaration truth table itself is pinned in ``test_targets.py``;
these cover only the wiring: axes + ``declaration_complete`` +
``parent_epic`` in the JSON, and ``story_type`` deleted.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    issue_payload: dict[str, Any],
) -> dict[str, Any]:
    import fetch_issue
    import utils

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert cmd[:3] == ["gh", "issue", "view"]
        return _completed(stdout=json.dumps(issue_payload))

    monkeypatch.setattr(utils, "run", fake_run)
    monkeypatch.setattr("sys.argv", ["fetch_issue.py", "--issue", str(issue_payload["number"])])
    fetch_issue.main()
    return json.loads(capsys.readouterr().out)


def test_labelled_issue_reports_axes_and_parent_epic(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _run_main(
        monkeypatch,
        capsys,
        {
            "number": 332,
            "title": "lanes",
            "body": "intro\n\nRefs #321\n",
            "labels": [
                {"name": "enhancement"},
                {"name": "kind:feature"},
                {"name": "target:factory"},
                {"name": "scale:task"},
            ],
            "state": "OPEN",
        },
    )
    assert out["kind"] == "feature"
    assert out["target"] == "factory"
    assert out["scale"] == "task"
    assert out["lane"] == "feature-factory"
    assert out["parent_epic"] == 321
    assert out["declaration_complete"] is True
    assert out["branch"] == "QS_332"


def test_unlabelled_issue_reports_empty_axes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _run_main(
        monkeypatch,
        capsys,
        {
            "number": 7,
            "title": "legacy",
            "body": "no declarations here",
            "labels": [{"name": "bug"}],
            "state": "OPEN",
        },
    )
    assert out["kind"] == ""
    assert out["target"] == ""
    assert out["scale"] == ""
    assert out["lane"] == ""
    assert out["parent_epic"] is None
    assert out["declaration_complete"] is False


def test_null_body_and_labels_degrade_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review-fix #02 (must-fix): the API can return present-but-null
    fields (`"body": null`, `"labels": null`). Fix plan #01 guarded the
    other two `parse_parent_epic` consumers (`context.py`,
    `create_pr.py`) but missed this one — iterating `None` and
    `re.search(None)` raised a raw traceback instead of the script's
    structured JSON. Twin of
    `test_context.py::test_null_body_from_the_api_degrades_to_no_parent_epic`.
    """
    out = _run_main(
        monkeypatch,
        capsys,
        {
            "number": 9,
            "title": "null fields",
            "body": None,
            "labels": None,
            "state": "OPEN",
        },
    )
    assert out["labels"] == []
    assert out["body"] == ""
    assert out["parent_epic"] is None
    assert out["declaration_complete"] is False
    assert out["lane"] == ""


def test_story_type_is_gone(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``story_type`` had zero consumers and a no-op branch — deleted."""
    out = _run_main(
        monkeypatch,
        capsys,
        {"number": 1, "title": "t", "body": "", "labels": [], "state": "OPEN"},
    )
    assert "story_type" not in out


def test_gh_failure_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import fetch_issue
    import utils

    monkeypatch.setattr(
        utils, "run", lambda cmd, **kw: _completed(returncode=1)
    )
    monkeypatch.setattr("sys.argv", ["fetch_issue.py", "--issue", "1"])
    with pytest.raises(SystemExit) as exc:
        fetch_issue.main()
    assert exc.value.code == 1
    assert "error" in json.loads(capsys.readouterr().out)
