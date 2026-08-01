"""Tests for ``scripts/qs/context.py`` — stdout contract and gh overlap.

Ten of the eleven-ish cases here are *characterization* tests: they pin
``context.py``'s pre-change behaviour (byte-identical stdout, guard
combinations, failure degradation) so that parallelizing the two ``gh``
calls cannot change what agents parse. The barrier test
(:func:`test_gh_calls_run_concurrently`) and the raising case of
:func:`test_gh_failure_paths` are the two that only pass once the calls
actually overlap.

Everything patches **``utils.run`` and only ``utils.run``**: ``run_gh``
and ``run_git`` resolve ``run`` from ``utils.__dict__`` at call time, so
a single patch catches every subprocess. Patching ``utils.run_gh`` would
miss ``context._issue_title`` (it holds its own reference); patching
``context.run_gh`` would miss ``utils.find_pr_for_branch``.

``context`` / ``utils`` are imported *inside* each test because
``tests/qs/conftest.py``'s autouse ``_add_scripts_qs_to_syspath`` fixture
inserts ``scripts/qs/`` at fixture time and purges those modules on
teardown.
"""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

PR_URL = "https://github.com/tmenguy/quiet-solar/pull/7"
PR_JSON = f'[{{"number": 7, "url": "{PR_URL}", "state": "OPEN"}}]'

GIT_BRANCH = ["git", "branch", "--show-current"]
GIT_ROOT = ["git", "rev-parse", "--show-toplevel"]
GH_ISSUE = ["gh", "issue", "view"]
GH_PR = ["gh", "pr", "list"]


def _completed(cmd: list[str], stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    """Build a ``CompletedProcess`` the way ``utils.run`` would."""
    return subprocess.CompletedProcess(args=cmd, returncode=returncode, stdout=stdout, stderr="")


def _make_fake_run(
    *,
    branch: str,
    repo_root: Path,
    title: str = "Fake title",
    gh_rc: dict[str, int] | None = None,
    barrier: threading.Barrier | None = None,
    raise_on: str | None = None,
) -> tuple[Callable[..., subprocess.CompletedProcess[str]], list[list[str]], threading.Lock]:
    """Return ``(fake_run, recorder, lock)`` matching ``utils.run``'s signature.

    Dispatch is on ``cmd[:3]``: the issue number sits at index 3 and the
    branch at index 4, so the first three tokens discriminate all four
    command shapes. ``gh_rc`` maps ``"issue"`` / ``"pr"`` to a return code
    (non-zero only ever for ``gh`` — the fake ignores ``check``, so a
    non-zero ``git`` response would silently return fake stdout instead of
    raising). ``barrier`` gates **only** ``gh`` commands: the pool block
    issues ``git`` concurrently too, so an ungated barrier would be
    satisfied by git+gh and stop discriminating serial from parallel.

    The recorder is written from both pool workers and the main thread, so
    it is guarded by ``lock``; all assertions on it must be
    order-independent.
    """
    codes = gh_rc or {}
    recorder: list[list[str]] = []
    lock = threading.Lock()

    def fake_run(
        cmd: list[str],
        *,
        check: bool = True,
        capture: bool = True,
        cwd: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        with lock:
            recorder.append(list(cmd))
        head = cmd[:3]
        if head == GIT_BRANCH:
            return _completed(cmd, f"{branch}\n" if branch else "")
        if head == GIT_ROOT:
            return _completed(cmd, f"{repo_root}\n")
        if barrier is not None and cmd[0] == "gh":
            barrier.wait()
        if head == GH_ISSUE:
            if raise_on == "issue":
                raise RuntimeError("boom: gh issue view")
            return _completed(cmd, f"{title}\n", returncode=codes.get("issue", 0))
        if head == GH_PR:
            return _completed(cmd, PR_JSON, returncode=codes.get("pr", 0))
        raise AssertionError(f"unexpected command: {cmd}")

    return fake_run, recorder, lock


def _select(recorder: list[list[str]], prefix: list[str]) -> list[list[str]]:
    """Return every recorded command whose first three tokens match ``prefix``."""
    return [cmd for cmd in recorder if cmd[:3] == prefix]


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    """Drive ``context.main()`` under ``argv``; the caller reads stdout via capsys.

    ``sys.argv`` must be set explicitly — otherwise ``parse_args()``
    consumes pytest's own argv and exits 2. ``main()`` always ends in
    ``sys.exit(0)`` on success, so the caller asserts on the exit code.
    """
    import context  # type: ignore[import-not-found]

    monkeypatch.setattr("sys.argv", ["context.py", *argv])
    with pytest.raises(SystemExit) as exc:
        context.main()
    assert exc.value.code == 0


@pytest.fixture(autouse=True)
def _claude_code_harness(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin ``detect()`` to ``claude-code`` — it honours ``QS_HARNESS`` first."""
    monkeypatch.setenv("QS_HARNESS", "claude-code")


# ---------------------------------------------------------------------------
# AC1 — stdout is byte-identical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_story", [False, True], ids=["no-story", "with-story"])
def test_stdout_is_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    with_story: bool,
) -> None:
    """stdout equals ``json.dumps(expected, indent=2) + "\\n"``, key order included."""
    import utils  # type: ignore[import-not-found]

    story = tmp_path / "docs" / "stories" / "QS-42.story.md"
    if with_story:
        story.parent.mkdir(parents=True)
        story.write_text("# story\n")

    fake_run, _recorder, _lock = _make_fake_run(branch="QS_42", repo_root=tmp_path)
    monkeypatch.setattr(utils, "run", fake_run)

    _run_main(monkeypatch, [])

    expected = {
        "harness": "claude-code",
        "branch": "QS_42",
        "issue": 42,
        "title": "Fake title",
        "story_file": str(story) if with_story else "",
        "story_exists": with_story,
        "latest_review_fix": "",
        "pr_number": 7,
        "pr_url": PR_URL,
        "worktree": str(tmp_path),
    }
    assert capsys.readouterr().out == json.dumps(expected, indent=2) + "\n"


# ---------------------------------------------------------------------------
# AC2 — the two gh calls overlap
# ---------------------------------------------------------------------------


def test_gh_calls_run_concurrently(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Both ``gh`` calls must be in flight at once.

    Timing-bounded, not deterministic: against serial code the first
    ``gh`` call's ``barrier.wait()`` raises ``BrokenBarrierError`` after
    the 5s timeout (uncaught by ``_issue_title`` or ``build_context``), so
    a RED run costs >=5s wall — that is the timeout, not a hang.

    *If the two calls are ever merged into one request, delete this test —
    do not "fix" it.*
    """
    import utils  # type: ignore[import-not-found]

    barrier = threading.Barrier(2, timeout=5)
    fake_run, recorder, _lock = _make_fake_run(
        branch="QS_42", repo_root=tmp_path, barrier=barrier
    )
    monkeypatch.setattr(utils, "run", fake_run)

    _run_main(monkeypatch, [])

    assert not barrier.broken
    assert len(_select(recorder, GH_ISSUE)) == 1
    assert len(_select(recorder, GH_PR)) == 1


# ---------------------------------------------------------------------------
# AC3 — guard combinations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "branch", "want_issue_cmd", "want_pr_head", "want_branch_calls"),
    [
        pytest.param([], "QS_42", "42", "QS_42", 1, id="issue-and-branch"),
        pytest.param([], "main", None, "main", 1, id="branch-only"),
        pytest.param([], "", None, None, 2, id="detached-head"),
        pytest.param(["--issue", "42"], "", "42", None, 1, id="override-detached"),
        pytest.param(["--issue", "42"], "QS_99", "42", "QS_99", 1, id="override-other-branch"),
    ],
)
def test_guard_combinations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    branch: str,
    want_issue_cmd: str | None,
    want_pr_head: str | None,
    want_branch_calls: int,
) -> None:
    """Each guard combination issues exactly the ``gh`` calls it should.

    The detached-HEAD row records **two** ``git branch --show-current``:
    ``get_issue_from_branch("")`` re-invokes ``get_current_branch()``. The
    override rows record one — ``issue_override or ...`` short-circuits.
    The last row is the leak check: the ``--issue`` override must not
    reach the PR lookup.
    """
    import utils  # type: ignore[import-not-found]

    fake_run, recorder, _lock = _make_fake_run(branch=branch, repo_root=tmp_path)
    monkeypatch.setattr(utils, "run", fake_run)

    _run_main(monkeypatch, argv)
    ctx = json.loads(capsys.readouterr().out)

    issue_cmds = _select(recorder, GH_ISSUE)
    if want_issue_cmd is None:
        assert issue_cmds == []
        assert ctx["title"] == ""
    else:
        assert len(issue_cmds) == 1
        assert want_issue_cmd in issue_cmds[0]
        assert ctx["title"] == "Fake title"

    pr_cmds = _select(recorder, GH_PR)
    if want_pr_head is None:
        assert pr_cmds == []
        assert ctx["pr_number"] is None
        assert ctx["pr_url"] == ""
    else:
        assert len(pr_cmds) == 1
        assert pr_cmds[0][pr_cmds[0].index("--head") + 1] == want_pr_head
        assert ctx["pr_number"] == 7

    assert len(_select(recorder, GIT_BRANCH)) == want_branch_calls


# ---------------------------------------------------------------------------
# AC4 — failure paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("gh_rc", "raise_on"),
    [
        pytest.param({"issue": 1}, None, id="title-fails"),
        pytest.param({"pr": 1}, None, id="pr-fails"),
        pytest.param({"issue": 1, "pr": 1}, None, id="both-fail"),
        pytest.param(None, "issue", id="title-raises"),
    ],
)
def test_gh_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    gh_rc: dict[str, int] | None,
    raise_on: str | None,
) -> None:
    """A failing ``gh`` degrades its own field; a raising one still issues both calls.

    ``run_gh`` passes ``check=False`` and ``_issue_title`` only inspects
    ``returncode``, so a non-zero exit never raises — exit stays 0 and the
    affected field degrades. The raising case is the only construction
    that pins behavioural delta 1 (both calls always execute): the
    ``returncode=1`` cases cannot, because serial code records both
    commands too.
    """
    import context  # type: ignore[import-not-found]
    import utils  # type: ignore[import-not-found]

    fake_run, recorder, _lock = _make_fake_run(
        branch="QS_42", repo_root=tmp_path, gh_rc=gh_rc, raise_on=raise_on
    )
    monkeypatch.setattr(utils, "run", fake_run)
    monkeypatch.setattr("sys.argv", ["context.py"])

    if raise_on is not None:
        with pytest.raises(RuntimeError, match="boom: gh issue view"):
            context.main()
        assert len(_select(recorder, GH_PR)) == 1
        return

    _run_main(monkeypatch, [])
    ctx = json.loads(capsys.readouterr().out)

    codes = gh_rc or {}
    assert ctx["title"] == ("" if codes.get("issue") else "Fake title")
    if codes.get("pr"):
        assert ctx["pr_number"] is None
        assert ctx["pr_url"] == ""
    else:
        assert ctx["pr_number"] == 7
        assert ctx["pr_url"] == PR_URL
