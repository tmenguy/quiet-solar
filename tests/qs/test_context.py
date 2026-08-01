"""Tests for ``scripts/qs/context.py`` — stdout contract and gh overlap.

Most cases here are *characterization* tests: they pin ``context.py``'s
pre-parallelization behaviour (byte-identical stdout, guard
combinations, failure degradation) so that overlapping the two ``gh``
calls cannot change what agents parse. Deliberately **not** in that set —
these only pass once the calls actually overlap, and they are the
behavioural contract of the overlap itself:

- :func:`test_gh_calls_run_concurrently` (the barrier proof)
- the ``title-raises`` case of :func:`test_gh_failure_paths` (delta 1)
- :func:`test_local_git_failure_surfaces_sibling_gh_error` and
  :func:`test_both_gh_calls_raising_surfaces_both`
- :func:`test_parallel_gh_calls_get_devnull_stdin`
- :func:`test_thread_exhaustion_falls_back_to_sequential`

No case count is stated on purpose — it rots (review fix #01 N5).

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
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, NamedTuple

import pytest

PR_URL = "https://github.com/tmenguy/quiet-solar/pull/7"
PR_JSON = f'[{{"number": 7, "url": "{PR_URL}", "state": "OPEN"}}]'

GIT_BRANCH = ["git", "branch", "--show-current"]
GIT_ROOT = ["git", "rev-parse", "--show-toplevel"]
GH_ISSUE = ["gh", "issue", "view"]
GH_PR = ["gh", "pr", "list"]

GH_ISSUE_BOOM = "boom: gh issue view"
GH_PR_BOOM = "boom: gh pr list"

RaiseTarget = Literal["issue", "pr"]


class _Call(NamedTuple):
    """One recorded ``utils.run`` invocation."""

    cmd: list[str]
    stdin: int | None


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
    raise_on: Sequence[RaiseTarget] = (),
    git_root_fails: bool = False,
) -> tuple[Callable[..., subprocess.CompletedProcess[str]], list[_Call]]:
    """Return ``(fake_run, recorder)`` matching ``utils.run``'s signature.

    Dispatch is on ``cmd[:3]``: the issue number sits at index 3 and the
    branch at index 4, so the first three tokens discriminate all four
    command shapes. ``gh_rc`` maps ``"issue"`` / ``"pr"`` to a return code
    (non-zero only ever for ``gh`` — the fake ignores ``check``, so a
    non-zero ``git`` response would silently return fake stdout instead of
    raising). ``barrier`` gates **only** ``gh`` commands: the pool block
    issues ``git`` concurrently too, so an ungated barrier would be
    satisfied by git+gh and stop discriminating serial from parallel.

    ``raise_on`` is a *sequence* of targets rather than a single value
    (review fix #01 N4): both targets are honoured — a silent no-op for
    ``"pr"`` is what blocked the "both raise" case — and a sequence is
    what lets one case raise from both. Unsupported values assert rather
    than no-op. ``git_root_fails`` makes ``git rev-parse --show-toplevel``
    raise ``CalledProcessError`` the way ``check=True`` would, which is
    how the local git work is failed while both ``gh`` futures are in
    flight.

    The recorder is written from both pool workers and the main thread, so
    it is guarded by an internal lock; all assertions on it must be
    order-independent.
    """
    assert not isinstance(raise_on, str), "raise_on takes a sequence, e.g. ('issue',)"
    unsupported = set(raise_on) - {"issue", "pr"}
    assert not unsupported, f"unsupported raise_on target(s): {sorted(unsupported)}"
    codes = gh_rc or {}
    recorder: list[_Call] = []
    lock = threading.Lock()

    def fake_run(
        cmd: list[str],
        *,
        check: bool = True,
        capture: bool = True,
        cwd: str | None = None,
        stdin: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        with lock:
            recorder.append(_Call(list(cmd), stdin))
        head = cmd[:3]
        if head == GIT_BRANCH:
            return _completed(cmd, f"{branch}\n" if branch else "")
        if head == GIT_ROOT:
            if git_root_fails:
                raise subprocess.CalledProcessError(128, cmd)
            return _completed(cmd, f"{repo_root}\n")
        if barrier is not None and cmd[0] == "gh":
            barrier.wait()
        if head == GH_ISSUE:
            if "issue" in raise_on:
                raise RuntimeError(GH_ISSUE_BOOM)
            return _completed(cmd, f"{title}\n", returncode=codes.get("issue", 0))
        if head == GH_PR:
            if "pr" in raise_on:
                raise RuntimeError(GH_PR_BOOM)
            return _completed(cmd, PR_JSON, returncode=codes.get("pr", 0))
        raise AssertionError(f"unexpected command: {cmd}")

    return fake_run, recorder


def _select(recorder: list[_Call], prefix: list[str]) -> list[_Call]:
    """Return every recorded call whose first three tokens match ``prefix``."""
    return [call for call in recorder if call.cmd[:3] == prefix]


def _notes(exc: BaseException) -> list[str]:
    """Return an exception's ``__notes__`` (absent until one is added)."""
    return list(getattr(exc, "__notes__", []))


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
    """stdout equals ``json.dumps(expected, indent=2) + "\\n"``, key order included.

    The ``with-story`` case also writes **two** review-fix files so the
    truthy branch of ``latest_review_fix`` and ``find_latest_review_fix``'s
    ``sorted(...)[-1]`` newest-wins selection are both reached through
    ``main()`` — without it, that key is only ever asserted as ``""``
    (review fix #01 S3).
    """
    import utils  # type: ignore[import-not-found]

    stories = tmp_path / "docs" / "stories"
    story = stories / "QS-42.story.md"
    review_fix = stories / "QS-42.story_review_fix_#02.md"
    if with_story:
        stories.mkdir(parents=True)
        story.write_text("# story\n")
        (stories / "QS-42.story_review_fix_#01.md").write_text("# fix 1\n")
        review_fix.write_text("# fix 2\n")

    fake_run, _recorder = _make_fake_run(branch="QS_42", repo_root=tmp_path)
    monkeypatch.setattr(utils, "run", fake_run)

    _run_main(monkeypatch, [])

    expected = {
        "harness": "claude-code",
        "branch": "QS_42",
        "issue": 42,
        "title": "Fake title",
        "story_file": str(story) if with_story else "",
        "story_exists": with_story,
        "latest_review_fix": str(review_fix) if with_story else "",
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
    fake_run, recorder = _make_fake_run(branch="QS_42", repo_root=tmp_path, barrier=barrier)
    monkeypatch.setattr(utils, "run", fake_run)

    _run_main(monkeypatch, [])

    assert not barrier.broken
    assert len(_select(recorder, GH_ISSUE)) == 1
    assert len(_select(recorder, GH_PR)) == 1


def test_parallel_gh_calls_get_devnull_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Concurrent ``gh`` children must not share the caller's stdin.

    Serialized, at most one ``gh`` could ever prompt for auth; overlapped,
    two children inheriting the same TTY can prompt at once and garble
    input. Both submissions pass ``subprocess.DEVNULL`` — scoped to this
    call site, so ``utils.run``'s default (inherit) is unchanged for every
    other caller (review fix #01 S2).
    """
    import utils  # type: ignore[import-not-found]

    fake_run, recorder = _make_fake_run(branch="QS_42", repo_root=tmp_path)
    monkeypatch.setattr(utils, "run", fake_run)

    _run_main(monkeypatch, [])

    gh_calls = _select(recorder, GH_ISSUE) + _select(recorder, GH_PR)
    assert len(gh_calls) == 2
    assert [call.stdin for call in gh_calls] == [subprocess.DEVNULL] * 2
    # The local git work is not concurrent with itself; leave it alone.
    assert all(call.stdin is None for call in _select(recorder, GIT_BRANCH))


# ---------------------------------------------------------------------------
# AC3 — guard combinations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "branch", "want_issue", "want_issue_cmd", "want_pr_head", "want_branch_calls"),
    [
        pytest.param([], "QS_42", 42, "42", "QS_42", 1, id="issue-and-branch"),
        pytest.param([], "main", None, None, "main", 1, id="branch-only"),
        pytest.param([], "", None, None, None, 2, id="detached-head"),
        pytest.param(["--issue", "42"], "", 42, "42", None, 1, id="override-detached"),
        pytest.param(["--issue", "42"], "QS_99", 42, "42", "QS_99", 1, id="override-other-branch"),
        pytest.param([], "QS_0", 0, None, "QS_0", 1, id="issue-zero-branch"),
    ],
)
def test_guard_combinations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    branch: str,
    want_issue: int | None,
    want_issue_cmd: str | None,
    want_pr_head: str | None,
    want_branch_calls: int,
) -> None:
    """Each guard combination issues exactly the ``gh`` calls it should.

    The detached-HEAD row records **two** ``git branch --show-current``:
    ``get_issue_from_branch("")`` re-invokes ``get_current_branch()``. The
    override rows record one — the override short-circuits. The
    ``override-other-branch`` row is the leak check: the ``--issue``
    override must not reach the PR lookup.

    The sixth row pins the ``issue == 0`` semantics (review fix #01 N1):
    ``QS_0`` reports ``"issue": 0`` faithfully but is **not** a usable
    issue number, so no ``gh issue view`` and no story lookup happen. The
    source story said "do not add a sixth row"; N1 supersedes that.
    """
    import utils  # type: ignore[import-not-found]

    fake_run, recorder = _make_fake_run(branch=branch, repo_root=tmp_path)
    monkeypatch.setattr(utils, "run", fake_run)

    _run_main(monkeypatch, argv)
    ctx = json.loads(capsys.readouterr().out)

    assert ctx["issue"] == want_issue

    issue_calls = _select(recorder, GH_ISSUE)
    if want_issue_cmd is None:
        assert issue_calls == []
        assert ctx["title"] == ""
    else:
        assert len(issue_calls) == 1
        assert want_issue_cmd in issue_calls[0].cmd
        assert ctx["title"] == "Fake title"

    pr_calls = _select(recorder, GH_PR)
    if want_pr_head is None:
        assert pr_calls == []
        assert ctx["pr_number"] is None
        assert ctx["pr_url"] == ""
    else:
        assert len(pr_calls) == 1
        pr_cmd = pr_calls[0].cmd
        assert pr_cmd[pr_cmd.index("--head") + 1] == want_pr_head
        assert ctx["pr_number"] == 7

    assert len(_select(recorder, GIT_BRANCH)) == want_branch_calls


@pytest.mark.parametrize("raw", ["0", "-1"], ids=["zero", "negative"])
def test_non_positive_issue_override_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: str
) -> None:
    """``--issue 0`` / ``--issue -1`` fail at the argparse boundary.

    Previously ``issue_override or ...`` silently discarded ``0`` and fell
    back to the branch-derived issue, while ``-1`` was truthy enough to
    issue ``gh issue view -1`` and degrade to an empty title with exit 0.
    Both now exit 2 instead of producing a useless context
    (review fix #01 N1).
    """
    import context  # type: ignore[import-not-found]
    import utils  # type: ignore[import-not-found]

    fake_run, recorder = _make_fake_run(branch="QS_42", repo_root=tmp_path)
    monkeypatch.setattr(utils, "run", fake_run)
    monkeypatch.setattr("sys.argv", ["context.py", "--issue", raw])

    with pytest.raises(SystemExit) as exc:
        context.main()

    assert exc.value.code == 2
    assert recorder == []


# ---------------------------------------------------------------------------
# AC4 — failure paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("gh_rc", "raise_on"),
    [
        pytest.param({"issue": 1}, (), id="title-fails"),
        pytest.param({"pr": 1}, (), id="pr-fails"),
        pytest.param({"issue": 1, "pr": 1}, (), id="both-fail"),
        pytest.param(None, ("issue",), id="title-raises"),
    ],
)
def test_gh_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    gh_rc: dict[str, int] | None,
    raise_on: Sequence[RaiseTarget],
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

    fake_run, recorder = _make_fake_run(
        branch="QS_42", repo_root=tmp_path, gh_rc=gh_rc, raise_on=raise_on
    )
    monkeypatch.setattr(utils, "run", fake_run)
    monkeypatch.setattr("sys.argv", ["context.py"])

    if raise_on:
        with pytest.raises(RuntimeError, match=GH_ISSUE_BOOM):
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


def test_both_gh_calls_raising_surfaces_both(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When both ``gh`` calls raise, neither exception is silently discarded.

    ``concurrent.futures`` never logs an unretrieved future exception, so
    without draining both futures the PR failure would vanish without a
    trace. The title error propagates (deterministic ordering) carrying
    the sibling failure as a note (review fix #01 S1).
    """
    import context  # type: ignore[import-not-found]
    import utils  # type: ignore[import-not-found]

    fake_run, recorder = _make_fake_run(
        branch="QS_42", repo_root=tmp_path, raise_on=("issue", "pr")
    )
    monkeypatch.setattr(utils, "run", fake_run)
    monkeypatch.setattr("sys.argv", ["context.py"])

    with pytest.raises(RuntimeError, match=GH_ISSUE_BOOM) as exc:
        context.main()

    assert len(_select(recorder, GH_ISSUE)) == 1
    assert len(_select(recorder, GH_PR)) == 1
    assert any(GH_PR_BOOM in note for note in _notes(exc.value))


def test_local_git_failure_surfaces_sibling_gh_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failing local git lookup propagates without swallowing the ``gh`` error.

    Reproduces the real shape: ``git branch --show-current`` succeeds while
    ``git rev-parse --show-toplevel`` fails (rc=128 from inside a ``.git/``
    directory, or the ``git worktree remove`` window), so ``get_repo_root()``
    raises inside the pool block *after* both futures were submitted. The
    git error is what surfaces; the concurrent ``gh`` failure rides along
    as a note instead of disappearing (review fix #01 S1).
    """
    import context  # type: ignore[import-not-found]
    import utils  # type: ignore[import-not-found]

    fake_run, recorder = _make_fake_run(
        branch="QS_42", repo_root=tmp_path, raise_on=("pr",), git_root_fails=True
    )
    monkeypatch.setattr(utils, "run", fake_run)
    monkeypatch.setattr("sys.argv", ["context.py"])

    with pytest.raises(subprocess.CalledProcessError) as exc:
        context.main()

    assert exc.value.returncode == 128
    assert any(GH_PR_BOOM in note for note in _notes(exc.value))
    # Both futures were in flight when the local work blew up.
    assert len(_select(recorder, GH_ISSUE)) == 1
    assert len(_select(recorder, GH_PR)) == 1


def test_thread_exhaustion_falls_back_to_sequential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``pool.submit`` raising ``RuntimeError`` degrades to inline calls.

    Thread-pool creation is a failure mode the serial code never had: under
    a low ``ulimit -u`` / pids cgroup, ``submit`` raises
    ``RuntimeError: can't start new thread`` and every agent startup would
    traceback. The fallback runs the same two helpers inline — slower, but
    a full context and exit 0 (review fix #01 N2).
    """
    import context  # type: ignore[import-not-found]
    import utils  # type: ignore[import-not-found]

    def no_threads(self: object, fn: Callable, /, *args: object, **kwargs: object) -> object:
        raise RuntimeError("can't start new thread")

    fake_run, recorder = _make_fake_run(branch="QS_42", repo_root=tmp_path)
    monkeypatch.setattr(utils, "run", fake_run)
    monkeypatch.setattr(context.ThreadPoolExecutor, "submit", no_threads)

    _run_main(monkeypatch, [])
    ctx = json.loads(capsys.readouterr().out)

    assert ctx["title"] == "Fake title"
    assert ctx["pr_number"] == 7
    assert len(_select(recorder, GH_ISSUE)) == 1
    assert len(_select(recorder, GH_PR)) == 1


def test_thread_exhaustion_fallback_still_propagates_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The inline fallback keeps the raising contract identical to the pooled path."""
    import context  # type: ignore[import-not-found]
    import utils  # type: ignore[import-not-found]

    def no_threads(self: object, fn: Callable, /, *args: object, **kwargs: object) -> object:
        raise RuntimeError("can't start new thread")

    fake_run, recorder = _make_fake_run(
        branch="QS_42", repo_root=tmp_path, raise_on=("issue",)
    )
    monkeypatch.setattr(utils, "run", fake_run)
    monkeypatch.setattr(context.ThreadPoolExecutor, "submit", no_threads)
    monkeypatch.setattr("sys.argv", ["context.py"])

    with pytest.raises(RuntimeError, match=GH_ISSUE_BOOM):
        context.main()

    assert len(_select(recorder, GH_PR)) == 1
