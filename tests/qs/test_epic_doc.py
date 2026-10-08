"""Tests for ``scripts/qs/epic_doc.py`` — the epic × factory lane's scripts (QS-340).

Three layers:

- **Pure functions** — ``parse_decomposition`` and ``sync_body`` on
  inline bodies (the #369 shape, legacy hand lists, CRLF, …).
- **Real git** — a throwaway bare ``origin`` plus two clones (``seed``
  plays "someone else pushing to main", ``work`` is the epic worktree),
  no network. ``utils.run`` is patched with a :class:`Runner` that answers
  every ``gh`` call and can inject a git failure or run a side effect
  just before a given git command — so every refusal branch is reached
  through the real commands.
- **Refusals leave the tree byte-identical** — each refusal test
  snapshots the worktree (every file's bytes + HEAD + status) and
  compares after.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "qs"))

import epic_doc  # type: ignore[import-not-found]  # noqa: E402
import mermaid_svg  # type: ignore[import-not-found]  # noqa: E402

import utils  # type: ignore[import-not-found]  # noqa: E402

ISSUE = 900
DOC = f"docs/epics/QS-{ISSUE}.md"
EPIC_LABELS = ["target:factory", "scale:epic", "area:dev-pipeline"]
REPO_URL = "https://github.com/acme/widgets"

_REAL_RUN = utils.run


# ---------------------------------------------------------------------------
# Real-git fixture
# ---------------------------------------------------------------------------


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return result.stdout


@dataclass
class Repos:
    origin: Path
    seed: Path
    work: Path

    def push_from_seed(self, path: str, text: str | None, message: str = "seed change") -> str:
        """Commit ``path`` (``None`` deletes it) on ``main`` from another clone."""
        _git(self.seed, "pull", "--ff-only", "-q")
        target = self.seed / path
        if text is None:
            _git(self.seed, "rm", "-q", path)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
            _git(self.seed, "add", path)
        _git(self.seed, "commit", "-q", "-m", message)
        _git(self.seed, "push", "-q", "origin", "main")
        return _git(self.seed, "rev-parse", "HEAD").strip()

    def main_file(self, path: str) -> str | None:
        result = subprocess.run(
            ["git", "show", f"main:{path}"], cwd=self.origin, capture_output=True, text=True
        )
        return result.stdout if result.returncode == 0 else None

    def main_sha(self) -> str:
        return _git(self.origin, "rev-parse", "main").strip()

    def write(self, path: str, text: str) -> None:
        target = self.work / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def snapshot(self) -> dict[str, Any]:
        files = {
            str(p.relative_to(self.work)): p.read_bytes()
            for p in sorted(self.work.rglob("*"))
            if p.is_file() and ".git" not in p.relative_to(self.work).parts
        }
        return {
            "files": files,
            "head": _git(self.work, "rev-parse", "HEAD"),
            "status": _git(self.work, "status", "--porcelain", "--untracked-files=all"),
        }


_TEMPLATE: dict[str, Path] = {}


def _template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the three repos once; each test gets a copy."""
    if "root" not in _TEMPLATE:
        root = tmp_path_factory.mktemp("epic_doc_template")
        env_cfg = root / "gitconfig"
        env_cfg.write_text(
            "[user]\n\tname = QS Test\n\temail = qs@example.invalid\n"
            "[init]\n\tdefaultBranch = main\n[commit]\n\tgpgsign = false\n"
        )
        _TEMPLATE["cfg"] = env_cfg
        _TEMPLATE["root"] = root
    return _TEMPLATE["root"]


@pytest.fixture
def repos(tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch) -> Repos:
    template = _template(tmp_path_factory)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(_TEMPLATE["cfg"]))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    if not (template / "origin.git").exists():
        _git(template, "init", "-q", "--bare", "-b", "main", "origin.git")
        _git(template, "clone", "-q", str(template / "origin.git"), "seed")
        seed = template / "seed"
        (seed / "README.md").write_text("readme\n")
        (seed / "scripts").mkdir()
        (seed / "scripts" / "tool.py").write_text("print('x')\n")
        (seed / "docs" / "epics").mkdir(parents=True)
        (seed / "docs" / "epics" / "QS-1.md").write_text("# Epic QS-1\n")
        _git(seed, "add", ".")
        _git(seed, "commit", "-q", "-m", "initial")
        _git(seed, "push", "-q", "origin", "main")
        _git(template, "clone", "-q", str(template / "origin.git"), "work")
        _git(template / "work", "checkout", "-q", "-b", f"QS_{ISSUE}")
    for name in ("origin.git", "seed", "work"):
        shutil.copytree(template / name, tmp_path / name, symlinks=True)
    origin = tmp_path / "origin.git"
    for clone in ("seed", "work"):
        _git(tmp_path / clone, "remote", "set-url", "origin", str(origin))
    monkeypatch.chdir(tmp_path / "work")
    return Repos(origin=origin, seed=tmp_path / "seed", work=tmp_path / "work")


# ---------------------------------------------------------------------------
# utils.run stand-in
# ---------------------------------------------------------------------------


def _done(cmd: list[str], rc: int = 0, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr=stderr)


@dataclass
class Runner:
    """Answers ``gh``; delegates ``git`` to the real ``utils.run`` unless hooked."""

    labels: list[str] = field(default_factory=lambda: list(EPIC_LABELS))
    state: str = "OPEN"
    body: str = ""
    issue_rc: int = 0
    issue_stdout: str | None = None
    repo_rc: int = 0
    repo_stdout: str | None = None
    edit_rc: int = 0
    calls: list[list[str]] = field(default_factory=list)
    edits: list[dict[str, Any]] = field(default_factory=list)
    hooks: list[tuple[Callable[[list[str]], bool], Callable[[list[str]], Any]]] = field(
        default_factory=list
    )

    def on(self, predicate: Callable[[list[str]], bool], action: Callable[[list[str]], Any]):
        self.hooks.append((predicate, action))

    def __call__(self, cmd: list[str], **kwargs: Any):
        self.calls.append(list(cmd))
        for predicate, action in list(self.hooks):
            if predicate(cmd):
                result = action(cmd)
                if result is not None:
                    return result
        if cmd[0] == "gh":
            return self._gh(cmd)
        return _REAL_RUN(cmd, **kwargs)

    def _gh(self, cmd: list[str]):
        if cmd[1:3] == ["issue", "view"]:
            stdout = self.issue_stdout
            if stdout is None:
                stdout = json.dumps({
                    "labels": [{"name": n} for n in self.labels],
                    "state": self.state,
                    "body": self.body,
                })
            return _done(cmd, self.issue_rc, stdout, "boom" if self.issue_rc else "")
        if cmd[1:3] == ["repo", "view"]:
            stdout = self.repo_stdout if self.repo_stdout is not None else json.dumps(
                {"url": REPO_URL + "/"}
            )
            return _done(cmd, self.repo_rc, stdout, "nope" if self.repo_rc else "")
        if cmd[1:3] == ["issue", "edit"]:
            path = Path(cmd[cmd.index("--body-file") + 1])
            self.edits.append({"path": path, "body": path.read_text(encoding="utf-8")})
            return _done(cmd, self.edit_rc, "", "denied" if self.edit_rc else "")
        raise AssertionError(f"unexpected gh command: {cmd}")

    def git_calls(self, *tokens: str) -> list[list[str]]:
        return [c for c in self.calls if c[0] == "git" and all(t in c for t in tokens)]


def _is(*tokens: str) -> Callable[[list[str]], bool]:
    return lambda cmd: cmd[0] == "git" and all(t in cmd for t in tokens)


@pytest.fixture
def runner(monkeypatch) -> Runner:
    fake = Runner()
    monkeypatch.setattr(utils, "run", fake)
    return fake


@pytest.fixture
def clean_drift(monkeypatch) -> list[list[str]]:
    """Patch the drift checker to a clean report; record its argv."""
    seen: list[list[str]] = []

    def fake_main(argv: list[str]) -> int:
        seen.append(list(argv))
        print(json.dumps({"stale_docs": [], "missing_covers": [], "malformed_frontmatter": []}))
        return 0

    monkeypatch.setattr(epic_doc.check_doc_drift, "main", fake_main)
    return seen


def _run(argv: list[str], capsys) -> tuple[int, dict]:
    rc = epic_doc.main(argv)
    return rc, json.loads(capsys.readouterr().out)


# S2: land now requires a parseable ``## Decomposition`` on the issue's own
# doc, so every doc that reaches a real land carries this minimal valid table.
_DECOMP = "\n## Decomposition\n\n| child | issue |\n|---|---|\n| a child | not filed |\n"


def _epic(intro: str) -> str:
    """An epic doc body with a valid Decomposition table appended (S2)."""
    return intro + _DECOMP


def _land(capsys, *extra: str, message: str = "QS-900: land the epic doc") -> tuple[int, dict]:
    return _run(["land", "--issue", str(ISSUE), "--message", message, *extra], capsys)


def _status(capsys, *extra: str) -> tuple[int, dict]:
    return _run(["status", "--issue", str(ISSUE), *extra], capsys)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_status_fresh_worktree_is_a_new_draft(repos, runner, capsys) -> None:
    rc, out = _status(capsys)
    assert rc == 0
    assert out["mode"] == "DECOMPOSE"
    assert (out["local"], out["on_main"], out["local_modified"], out["behind"]) == (
        False, False, False, False,
    )
    assert out["safe_to_discard"] is True
    assert out["diff"] == ""
    assert runner.git_calls("fetch", "origin", "main")


def test_status_local_draft_continues_decompose(repos, runner, capsys) -> None:
    repos.write(DOC, "# Epic QS-900\n\ndraft\n")
    rc, out = _status(capsys)
    assert rc == 0
    assert out["mode"] == "DECOMPOSE"
    assert out["local"] is True and out["on_main"] is False
    assert out["local_modified"] is True
    assert out["safe_to_discard"] is False
    assert "+draft" in out["diff"]


def test_status_doc_on_main_is_resume_and_sync_fast_forwards(repos, runner, capsys) -> None:
    repos.push_from_seed(DOC, "# Epic QS-900\n\nlanded\n")
    rc, out = _status(capsys)
    assert rc == 0
    assert out["mode"] == "RESUME"
    assert out["on_main"] is True and out["local"] is False
    assert out["behind"] is True and out["local_modified"] is False
    assert out["safe_to_discard"] is True and out["synced"] is False

    rc, out = _status(capsys, "--sync")
    assert rc == 0
    assert out["synced"] is True and out["behind"] is False and out["local"] is True
    assert (repos.work / DOC).read_text() == "# Epic QS-900\n\nlanded\n"


def test_status_sync_never_touches_unsafe_worktree(repos, runner, capsys) -> None:
    repos.push_from_seed(DOC, "# on main\n")
    repos.write("docs/epics/QS-7.md", "unrelated local work\n")
    before = repos.snapshot()
    rc, out = _status(capsys, "--sync")
    assert rc == 0
    assert out["behind"] is True and out["safe_to_discard"] is False
    assert out["synced"] is False
    assert not runner.git_calls("merge", "--ff-only")
    assert repos.snapshot() == before


def test_status_sync_fast_forwards_when_stale_only_on_other_paths(repos, runner, capsys) -> None:
    """S4: --sync fast-forwards a clean worktree that is behind on non-doc paths."""
    repos.push_from_seed("README.md", "updated readme\n")
    rc, out = _status(capsys, "--sync")
    assert rc == 0
    assert out["doc_differs"] is False
    assert out["synced"] is True and out["behind"] is False
    assert runner.git_calls("merge", "--ff-only")
    assert (repos.work / "README.md").read_text() == "updated readme\n"


def test_status_landed_but_unreset_is_safe_to_discard(repos, runner, capsys) -> None:
    """S4: after a land whose reset never ran, every changed byte is already on main."""
    repos.push_from_seed(DOC, "# landed\n")
    repos.write(DOC, "# landed\n")  # identical bytes, never pulled → in the change set
    rc, out = _status(capsys)
    assert rc == 0
    assert out["safe_to_discard"] is True
    assert out["landed_not_reset"] is True
    assert out["local_modified"] is True


def test_status_sync_resets_a_landed_but_unreset_worktree(repos, runner, capsys) -> None:
    """S4: --sync in a landed-not-reset state resets instead of a clobbering ff-merge."""
    repos.push_from_seed(DOC, "# landed\n")
    repos.write(DOC, "# landed\n")
    rc, out = _status(capsys, "--sync")
    assert rc == 0
    assert out["synced"] is True and out["behind"] is False
    assert not runner.git_calls("merge", "--ff-only")
    assert runner.git_calls("reset", "--hard")
    assert _git(repos.work, "status", "--porcelain") == ""


def test_status_sync_reports_post_sync_fields(repos, runner, capsys) -> None:
    """N1: --sync mutates the tree, so the reported fields are post-sync, not stale."""
    repos.push_from_seed(DOC, "# landed\n")
    repos.write(DOC, "# landed\n")  # landed-not-reset: local_modified is True pre-sync
    rc, out = _status(capsys, "--sync")
    assert rc == 0 and out["synced"] is True
    assert out["local_modified"] is False
    assert out["landed_not_reset"] is False
    assert out["doc_differs"] is False
    assert out["safe_to_discard"] is True
    assert out["diff"] == ""


def test_status_committed_then_deleted_doc_is_not_safe_to_discard(repos, runner, capsys) -> None:
    """N2: a doc committed on the branch and then deleted from the worktree is not
    'already landed' — dropping the commit would lose the only copy of the doc."""
    repos.write(DOC, "# draft only on the branch\n")
    _git(repos.work, "add", DOC)
    _git(repos.work, "commit", "-q", "-m", "wip doc")
    (repos.work / DOC).unlink()  # absent locally AND absent on main
    rc, out = _status(capsys)
    assert rc == 0
    assert out["landed_not_reset"] is False
    assert out["safe_to_discard"] is False


def test_status_doc_differs_when_head_diverges_from_main(repos, runner, capsys) -> None:
    """S4: doc_differs is the HEAD-vs-origin/main doc delta, separate from behind."""
    repos.write(DOC, "# committed draft\n")
    _git(repos.work, "add", DOC)
    _git(repos.work, "commit", "-q", "-m", "wip doc")
    rc, out = _status(capsys)
    assert rc == 0
    assert out["doc_differs"] is True
    assert out["behind"] is False  # HEAD is ahead of origin/main, not behind


def test_status_local_edit_of_landed_doc_shows_diff(repos, runner, capsys) -> None:
    repos.push_from_seed(DOC, "line one\n")
    _status(capsys, "--sync")
    repos.write(DOC, "line one\nline two\n")
    rc, out = _status(capsys)
    assert rc == 0
    assert out["mode"] == "RESUME"
    # ``behind`` compares HEAD with origin/main — the edit is not a commit.
    assert out["local_modified"] is True and out["behind"] is False
    assert out["safe_to_discard"] is False
    assert "+line two" in out["diff"] and f"origin/main:{DOC}" in out["diff"]


def test_status_locally_deleted_doc_diff(repos, runner, capsys) -> None:
    repos.push_from_seed(DOC, "gone soon\n")
    _status(capsys, "--sync")
    (repos.work / DOC).unlink()
    rc, out = _status(capsys)
    assert rc == 0
    assert out["local"] is False and out["local_modified"] is True
    assert "-gone soon" in out["diff"]


def test_status_leftover_commit_is_not_safe_to_discard(repos, runner, capsys) -> None:
    """An empty leftover commit changes no path but is not on main."""
    _git(repos.work, "commit", "-q", "--allow-empty", "-m", "leftover")
    rc, out = _status(capsys)
    assert rc == 0
    assert out["local_modified"] is False
    assert out["safe_to_discard"] is False
    # S3: an empty leftover commit shows in unpushed_commits (nothing in changed).
    assert out["unpushed_commits"] == 1
    assert out["changed"] == []


def test_status_reports_changed_paths_and_unpushed_commits(repos, runner, capsys) -> None:
    """S3: a local edit AND a committed non-doc change both surface, so a false
    safe_to_discard can be explained even when the delta is not the epic doc."""
    repos.write(DOC, "# Epic QS-900\n\nlocal edit\n")  # uncommitted doc edit
    repos.write("docs/workflow/note.md", "an uncommitted lane note\n")
    rc, out = _status(capsys)
    assert rc == 0
    assert out["safe_to_discard"] is False
    assert DOC in out["changed"]
    assert "docs/workflow/note.md" in out["changed"]
    assert out["unpushed_commits"] == 0  # both edits are uncommitted


def test_status_post_sync_clears_changed_and_unpushed(repos, runner, capsys) -> None:
    """S3/N1: after --sync the reported changed/unpushed fields are post-sync."""
    repos.push_from_seed(DOC, "# landed\n")
    repos.write(DOC, "# landed\n")  # landed-not-reset
    rc, out = _status(capsys, "--sync")
    assert rc == 0 and out["synced"] is True
    assert out["changed"] == []
    assert out["unpushed_commits"] == 0


def test_status_git_error_on_bad_predicate(repos, runner, capsys) -> None:
    runner.on(_is("diff", "--quiet"), lambda cmd: _done(cmd, 128, "", "fatal: bad"))
    rc, out = _status(capsys)
    assert rc == 1
    assert out["status"] == "git-error"
    assert "fatal: bad" in out["detail"]


def test_status_merge_failure_is_git_error(repos, runner, capsys) -> None:
    repos.push_from_seed(DOC, "x\n")
    runner.on(_is("merge", "--ff-only"), lambda cmd: _done(cmd, 1, "", "not possible"))
    rc, out = _status(capsys, "--sync")
    assert rc == 1 and out["status"] == "git-error"


def test_toplevel_failure_is_git_error(monkeypatch, capsys) -> None:
    fake = Runner()
    fake.on(_is("rev-parse", "--show-toplevel"), lambda cmd: _done(cmd, 128, "", "not a repo"))
    monkeypatch.setattr(utils, "run", fake)
    rc, out = _status(capsys)
    assert rc == 1
    assert out == {
        "status": "git-error",
        "command": "git rev-parse --show-toplevel",
        "detail": "not a repo",
    }


def test_fetch_failure_is_git_error(repos, runner, capsys) -> None:
    runner.on(_is("fetch"), lambda cmd: _done(cmd, 1, "", "offline"))
    rc, out = _status(capsys)
    assert rc == 1 and out["status"] == "git-error" and out["detail"] == "offline"


# ---------------------------------------------------------------------------
# land — success paths
# ---------------------------------------------------------------------------


def test_first_landing_builds_on_main_pushes_verifies_then_resets(
    repos, runner, clean_drift, capsys
) -> None:
    base = repos.main_sha()
    repos.write(DOC, _epic("# Epic QS-900\n"))
    rc, out = _land(capsys, message="QS-900: land\n\nCo-Authored-By: x")
    assert rc == 0, out
    assert out["status"] == "landed"
    assert out["paths"] == [DOC]
    assert repos.main_file(DOC) == _epic("# Epic QS-900\n")
    landed = repos.main_sha()
    assert landed == out["sha"]
    assert _git(repos.origin, "rev-parse", f"{landed}^").strip() == base
    assert _git(repos.origin, "log", "-1", "--format=%B", landed).strip() == (
        "QS-900: land\n\nCo-Authored-By: x"
    )
    # The worktree now matches main and is clean.
    assert _git(repos.work, "rev-parse", "HEAD").strip() == landed
    assert _git(repos.work, "status", "--porcelain") == ""
    # The drift checker ran in-process on exactly the changed paths.
    assert clean_drift == [["--repo-root", str(repos.work), "--json", "--paths", DOC]]
    # The real index was never used for the landing commit — built on the
    # origin/main pinned right after the fetch (review fix #01 F4).
    assert runner.git_calls("read-tree", base)


def test_landing_a_committed_doc(repos, runner, clean_drift, capsys) -> None:
    """Committed changes to the issue's own doc count as landable (N6)."""
    repos.write(DOC, _epic("# committed\n"))
    _git(repos.work, "add", DOC)
    _git(repos.work, "commit", "-q", "-m", "wip")
    rc, out = _land(capsys)
    assert rc == 0, out
    assert out["paths"] == [DOC]
    assert repos.main_file(DOC) == _epic("# committed\n")
    assert repos.main_file("README.md") == "readme\n"


def test_build_commit_removes_a_deleted_path_from_the_tree(repos, runner) -> None:
    """_build_commit's force-remove branch: a path absent from the worktree is
    dropped from the landing tree (exercised directly since N6 keeps cmd_land
    to a single always-present doc)."""
    root = str(repos.work)
    _git(repos.work, "fetch", "-q", "origin", "main")
    (repos.work / "docs/epics/QS-1.md").unlink()  # absent locally, present on main
    sha = epic_doc._build_commit(root, ["docs/epics/QS-1.md"], "drop QS-1")
    assert sha is not None
    tree = _git(repos.work, "ls-tree", "-r", "--name-only", sha)
    assert "docs/epics/QS-1.md" not in tree.split()


def test_dry_run_stops_before_plumbing(repos, runner, clean_drift, capsys) -> None:
    repos.write(DOC, _epic("# draft\n"))
    before = repos.snapshot()
    main_before = repos.main_sha()
    rc, out = _land(capsys, "--dry-run")
    assert rc == 0
    assert out["status"] == "ok-dry-run" and out["paths"] == [DOC]
    assert out["warnings"] == []
    for token in ("read-tree", "commit-tree", "push", "reset"):
        assert not runner.git_calls(token), token
    assert repos.snapshot() == before
    assert repos.main_sha() == main_before


def test_dry_run_does_not_reset_a_landed_but_unreset_worktree(repos, runner, capsys) -> None:
    """S1: --dry-run in a landed-but-not-reset state leaves HEAD and the tree unchanged."""
    repos.push_from_seed(DOC, "# landed\n")
    repos.write(DOC, "# landed\n")  # identical bytes, never pulled → in the change set
    before = repos.snapshot()
    main_before = repos.main_sha()
    rc, out = _land(capsys, "--dry-run")
    assert rc == 0, out
    assert out["status"] == "ok-dry-run"
    assert out["already_landed"] is True and out["reset"] is False
    assert not runner.git_calls("reset")
    assert repos.snapshot() == before
    assert repos.main_sha() == main_before


def test_malformed_drift_entries_are_warnings(repos, runner, monkeypatch, capsys) -> None:
    def fake_main(argv: list[str]) -> int:
        print(json.dumps({
            "stale_docs": [],
            "missing_covers": ["docs/agents/a.md::custom_components/quiet_solar/x.py"],
            "malformed_frontmatter": ["docs/agents/b.md"],
        }))
        return 2

    monkeypatch.setattr(epic_doc.check_doc_drift, "main", fake_main)
    repos.write(DOC, _epic("# draft\n"))
    rc, out = _land(capsys, "--dry-run")
    assert rc == 0, out
    assert out["warnings"] == [
        "malformed frontmatter: docs/agents/b.md",
        "missing covers path: docs/agents/a.md::custom_components/quiet_solar/x.py",
    ]


def test_empty_change_set_with_doc_on_main_is_already_landed(repos, runner, capsys) -> None:
    repos.push_from_seed(DOC, "# landed\n")
    _git(repos.work, "pull", "-q", "--ff-only", "origin", "main")
    rc, out = _land(capsys)
    assert rc == 0
    assert out == {"status": "already-landed", "paths": [], "reset": False}


def test_push_accepted_but_reset_failed_then_rerun(repos, runner, clean_drift, capsys) -> None:
    """The push reached main, the local reset never ran: a re-run resets and
    reports ``already-landed``; ``status`` then reports safe to discard."""
    repos.write(DOC, _epic("# Epic\n"))
    fired: list[bool] = []

    def fail_first_reset(cmd: list[str]):
        if not fired:
            fired.append(True)
            return _done(cmd, 1, "", "index.lock exists")
        return None

    runner.on(_is("reset", "--hard"), fail_first_reset)
    rc, out = _land(capsys)
    assert rc == 1 and out["status"] == "git-error"
    assert repos.main_file(DOC) == _epic("# Epic\n")

    rc, out = _land(capsys)
    assert rc == 0
    assert out == {"status": "already-landed", "paths": [DOC], "reset": True}
    assert _git(repos.work, "rev-parse", "HEAD").strip() == repos.main_sha()

    rc, out = _status(capsys)
    assert out["safe_to_discard"] is True and out["local_modified"] is False


def test_land_refuses_loudly_when_build_commit_returns_none(
    repos, runner, clean_drift, monkeypatch, capsys
) -> None:
    """N6: past the already-landed shortcut ``_build_commit`` can only be ``None``
    on a broken internal invariant — land must refuse loudly, never push ``None``."""
    repos.write(DOC, _epic("# Epic\n"))
    monkeypatch.setattr(epic_doc, "_build_commit", lambda *a, **k: None)
    before = repos.snapshot()
    rc, out = _land(capsys)
    assert rc == 1 and out["status"] == "git-error"
    assert "internal" in out["detail"]
    assert not runner.git_calls("push", "origin")
    assert repos.snapshot() == before


def test_build_commit_returns_none_for_a_tree_equal_to_main(repos, runner) -> None:
    root = str(repos.work)
    _git(repos.work, "fetch", "-q", "origin", "main")
    assert epic_doc._build_commit(root, ["docs/epics/QS-1.md"], "m") is None


def test_verify_accepts_the_landed_commit_as_an_ancestor(
    repos, runner, clean_drift, capsys
) -> None:
    """Someone pushed on top of the landing commit before the verify fetch."""
    repos.write(DOC, _epic("# Epic\n"))

    def push_on_top(cmd: list[str]):
        if cmd[1:2] == ["fetch"] and runner.git_calls("push"):
            repos.push_from_seed("README.md", "later\n", "on top")
        return None

    runner.on(lambda cmd: cmd[0] == "git", push_on_top)
    rc, out = _land(capsys)
    assert rc == 0 and out["status"] == "landed"
    assert repos.main_file("README.md") == "later\n"
    assert repos.main_file(DOC) == _epic("# Epic\n")


# ---------------------------------------------------------------------------
# land — refusals leave the working tree byte-identical
# ---------------------------------------------------------------------------


def _refused(repos: Repos, capsys, status: str, *extra: str) -> dict:
    before = repos.snapshot()
    main_before = repos.main_sha()
    rc, out = _land(capsys, *extra)
    assert rc == 1, out
    assert out["status"] == status, out
    assert repos.snapshot() == before
    assert repos.main_sha() == main_before
    return out


def test_lookup_failure_is_not_not_an_epic(repos, runner, capsys) -> None:
    runner.issue_rc = 1
    repos.write(DOC, "x\n")
    out = _refused(repos, capsys, "lookup-failed")
    assert out["detail"] == "boom"


@pytest.mark.parametrize("raw", ["not json", "null", "[]"])
def test_lookup_invalid_json(repos, runner, capsys, raw: str) -> None:
    runner.issue_stdout = raw
    _refused(repos, capsys, "lookup-failed")


def test_non_epic_issue_is_refused(repos, runner, capsys) -> None:
    runner.labels = ["kind:feature", "target:factory", "scale:task"]
    repos.write(DOC, "x\n")
    out = _refused(repos, capsys, "not-an-epic")
    assert "scale:epic" in out["detail"]
    assert not runner.git_calls("fetch")


def test_out_of_scope_untracked(repos, runner, capsys) -> None:
    repos.write(DOC, "x\n")
    repos.write("notes.txt", "stray\n")
    assert _refused(repos, capsys, "out-of-scope")["offenders"] == ["notes.txt"]


def test_out_of_scope_staged(repos, runner, capsys) -> None:
    repos.write(DOC, "x\n")
    repos.write("scripts/new.py", "x\n")
    _git(repos.work, "add", "scripts/new.py")
    assert _refused(repos, capsys, "out-of-scope")["offenders"] == ["scripts/new.py"]


def test_out_of_scope_unstaged_modification(repos, runner, capsys) -> None:
    repos.write(DOC, "x\n")
    repos.write("README.md", "edited\n")
    assert _refused(repos, capsys, "out-of-scope")["offenders"] == ["README.md"]


def test_out_of_scope_porcelain_rename_contributes_both_paths(repos, runner, capsys) -> None:
    repos.write(DOC, "x\n")
    _git(repos.work, "mv", "scripts/tool.py", "docs/epics/tool.py")
    # M3: docs/epics/tool.py is not a QS-<N>.md, so both paths are offenders.
    assert _refused(repos, capsys, "out-of-scope")["offenders"] == [
        "docs/epics/tool.py", "scripts/tool.py",
    ]


def test_out_of_scope_committed_rename_into_docs_epics(repos, runner, capsys) -> None:
    repos.write(DOC, "x\n")
    _git(repos.work, "mv", "scripts/tool.py", "docs/epics/tool.py")
    _git(repos.work, "commit", "-q", "-m", "sneaky rename")
    # M3: the renamed-in file is not a valid epic doc, so it is refused too.
    assert _refused(repos, capsys, "out-of-scope")["offenders"] == [
        "docs/epics/tool.py", "scripts/tool.py",
    ]


def test_out_of_scope_untracked_swap_file_under_docs_epics(repos, runner, capsys) -> None:
    """M3: an editor swap file under docs/epics/ is refused, not landed."""
    repos.write(DOC, "x\n")
    repos.write("docs/epics/.QS-900.md.swp", "vim junk\n")
    assert _refused(repos, capsys, "out-of-scope")["offenders"] == ["docs/epics/.QS-900.md.swp"]


def test_out_of_scope_merge_orig_file_under_docs_epics(repos, runner, capsys) -> None:
    """M3: a merge-tool ``*.orig`` under docs/epics/ is refused, not landed."""
    repos.write(DOC, "x\n")
    repos.write("docs/epics/QS-900.md.orig", "conflict junk\n")
    assert _refused(repos, capsys, "out-of-scope")["offenders"] == ["docs/epics/QS-900.md.orig"]


_BAD_DECOMP_DOC = (
    "# Epic QS-900\n\n## Decomposition\n\n"
    "| # | child | issue |\n|---|---|---|\n| 1 | a child | TBD |\n"
)


def test_land_refuses_an_unparseable_decomposition_before_plumbing(repos, runner, capsys) -> None:
    """S2: a malformed Decomposition cell is caught at land, not later at sync-issue."""
    repos.write(DOC, _BAD_DECOMP_DOC)
    out = _refused(repos, capsys, "unparseable-decomposition")
    assert out["doc"] == DOC


def test_land_dry_run_also_validates_the_decomposition(repos, runner, capsys) -> None:
    """S2: dry-run validates the table too, and touches nothing."""
    repos.write(DOC, _BAD_DECOMP_DOC)
    before = repos.snapshot()
    rc, out = _land(capsys, "--dry-run")
    assert rc == 1 and out["status"] == "unparseable-decomposition"
    assert not runner.git_calls("read-tree")
    assert repos.snapshot() == before


def test_land_refuses_a_numbered_decomposition_heading(repos, runner, capsys) -> None:
    """S2: '## 5. Decomposition' is not the exact heading sync-issue reads."""
    repos.write(
        DOC,
        "# Epic QS-900\n\n## 5. Decomposition\n\n| child | issue |\n|---|---|\n| a | not filed |\n",
    )
    out = _refused(repos, capsys, "unparseable-decomposition")
    assert out["doc"] == DOC


def test_land_refuses_a_doc_without_a_decomposition_section(repos, runner, capsys) -> None:
    """S2: a doc with no ## Decomposition section is refused, not landed
    half-way (sync-issue would refuse it after the push)."""
    repos.write(DOC, "# Epic QS-900\n\njust prose, no table yet\n")
    out = _refused(repos, capsys, "unparseable-decomposition")
    assert out["doc"] == DOC


def test_land_skips_revalidation_when_already_landed(repos, runner, capsys) -> None:
    """S2: an already-landed worktree (bytes on main) is cleaned up without
    re-parsing the doc — even a doc that predates the Decomposition rule."""
    repos.push_from_seed(DOC, "# landed long ago, no table\n")
    repos.write(DOC, "# landed long ago, no table\n")  # identical bytes → already-landed
    rc, out = _land(capsys)
    assert rc == 0, out
    assert out["status"] == "already-landed" and out["reset"] is True


def test_missing_doc_with_empty_change_set_not_on_main(repos, runner, capsys) -> None:
    out = _refused(repos, capsys, "missing-doc")
    assert out["doc"] == DOC


def test_land_refuses_a_locally_deleted_doc(repos, runner, capsys) -> None:
    """The doc is in the change set (a deletion) but absent from the worktree."""
    repos.push_from_seed(DOC, "# on main\n")
    _git(repos.work, "pull", "-q", "--ff-only", "origin", "main")
    (repos.work / DOC).unlink()
    out = _refused(repos, capsys, "missing-doc")
    assert out["doc"] == DOC
    assert "does not exist" in out["detail"]


def test_only_another_epic_doc_changed_is_out_of_scope(repos, runner, capsys) -> None:
    """N6: editing only another epic's doc is out-of-scope (was missing-doc)."""
    repos.write("docs/epics/QS-1.md", "# edited\n")
    out = _refused(repos, capsys, "out-of-scope")
    assert out["offenders"] == ["docs/epics/QS-1.md"]


def test_stale_docs_refuse_with_the_report(repos, runner, monkeypatch, capsys) -> None:
    report = {
        "stale_docs": [{"doc": "docs/agents/x.md", "stale_sources": ["a"], "last_verified": ""}],
        "missing_covers": [],
        "malformed_frontmatter": [],
    }

    def fake_main(argv: list[str]) -> int:
        print(json.dumps(report))
        return 1

    monkeypatch.setattr(epic_doc.check_doc_drift, "main", fake_main)
    repos.write(DOC, _epic("x\n"))
    out = _refused(repos, capsys, "drift")
    assert out["drift"] == report


def test_drift_without_json_refuses(repos, runner, monkeypatch, capsys) -> None:
    monkeypatch.setattr(epic_doc.check_doc_drift, "main", lambda argv: 2)
    repos.write(DOC, _epic("x\n"))
    _refused(repos, capsys, "drift")


def test_drift_systemexit_is_a_refusal_not_a_traceback(repos, runner, monkeypatch, capsys) -> None:
    """S5: argparse-style SystemExit from the drift checker becomes a JSON refusal."""
    def boom(argv: list[str]) -> int:
        raise SystemExit(2)

    monkeypatch.setattr(epic_doc.check_doc_drift, "main", boom)
    repos.write(DOC, _epic("x\n"))
    _refused(repos, capsys, "drift")


def test_drift_unexpected_exception_is_a_refusal(repos, runner, monkeypatch, capsys) -> None:
    """S5: any other exception inside the drift checker becomes a JSON refusal."""
    def boom(argv: list[str]) -> int:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(epic_doc.check_doc_drift, "main", boom)
    repos.write(DOC, _epic("x\n"))
    out = _refused(repos, capsys, "drift")
    assert "kaboom" in out["detail"]


def test_conflict_carries_fresh_main_content(repos, runner, clean_drift, capsys) -> None:
    repos.push_from_seed(DOC, "v1\n")
    _git(repos.work, "pull", "-q", "--ff-only", "origin", "main")
    repos.write(DOC, _epic("v1\nlocal edit\n"))
    main_blob = repos.push_from_seed(DOC, "v1\nmain edit\n", "child PR amends the doc")
    del main_blob
    out = _refused(repos, capsys, "conflict")
    (conflict,) = out["conflicts"]
    assert conflict["path"] == DOC
    assert conflict["main_content"] == "v1\nmain edit\n"
    blob = conflict["main_blob"]
    assert _git(repos.origin, "rev-parse", f"main:{DOC}").strip() == blob

    # The agent merges main's content into the local file and re-runs.
    repos.write(DOC, _epic("v1\nmain edit\nlocal edit\n"))
    rc, out = _land(capsys, "--merged", f"{DOC}={blob}")
    assert rc == 0, out
    assert out["status"] == "landed"
    assert repos.main_file(DOC) == _epic("v1\nmain edit\nlocal edit\n")


def test_conflict_refused_again_when_main_moves_after_the_refusal(
    repos, runner, clean_drift, capsys
) -> None:
    repos.push_from_seed(DOC, "v1\n")
    _git(repos.work, "pull", "-q", "--ff-only", "origin", "main")
    repos.push_from_seed(DOC, "v2\n")
    repos.write(DOC, _epic("v1\nlocal\n"))
    stale_blob = _refused(repos, capsys, "conflict")["conflicts"][0]["main_blob"]
    repos.write(DOC, _epic("v2\nlocal\n"))
    repos.push_from_seed(DOC, "v3\n")
    out = _refused(repos, capsys, "conflict", "--merged", f"{DOC}={stale_blob}")
    assert out["conflicts"][0]["main_content"] == "v3\n"
    assert out["conflicts"][0]["main_blob"] != stale_blob


def test_land_refuses_another_epics_doc(repos, runner, capsys) -> None:
    """N6: land pushes only the issue's own doc — another epic's doc is out-of-scope."""
    repos.write(DOC, _epic("# Epic QS-900\n"))
    repos.write("docs/epics/QS-1.md", "# edited other epic\n")
    out = _refused(repos, capsys, "out-of-scope")
    assert out["offenders"] == ["docs/epics/QS-1.md"]


def test_conflict_check_git_error(repos, runner, clean_drift, capsys) -> None:
    repos.write(DOC, _epic("x\n"))
    runner.on(_is("diff", "--quiet"), lambda cmd: _done(cmd, 128, "", "fatal"))
    _refused(repos, capsys, "git-error")


def test_bad_merged_argument(repos, runner, capsys) -> None:
    out = _refused(repos, capsys, "bad-arguments", "--merged", "no-equals-sign")
    assert "PATH=<blob>" in out["detail"]


@pytest.mark.parametrize("message", ["", "   ", "\n\t "])
def test_blank_message_is_refused_before_plumbing(repos, runner, capsys, message: str) -> None:
    """S8: an empty / whitespace-only --message never reaches commit-tree."""
    repos.write(DOC, "# draft\n")
    before = repos.snapshot()
    main_before = repos.main_sha()
    rc, out = _land(capsys, message=message)
    assert rc == 1 and out["status"] == "bad-arguments"
    assert not runner.git_calls("commit-tree")
    assert repos.snapshot() == before
    assert repos.main_sha() == main_before


def test_land_refuses_an_undecodable_doc(repos, runner, capsys) -> None:
    """N2: a non-UTF-8 landing doc is a refusal, not a traceback."""
    (repos.work / DOC).parent.mkdir(parents=True, exist_ok=True)
    (repos.work / DOC).write_bytes(b"\xff\xfe not utf-8\n")
    out = _refused(repos, capsys, "undecodable")
    assert out["detail"] == DOC


def test_land_conflict_with_an_undecodable_main_blob(repos, runner, clean_drift, capsys) -> None:
    """S8: reading a non-UTF-8 main blob during a conflict is a refusal, not a crash."""
    # main gets a non-UTF-8 version of the doc, pushed from the seed clone.
    _git(repos.seed, "pull", "--ff-only", "-q")
    (repos.seed / DOC).parent.mkdir(parents=True, exist_ok=True)
    (repos.seed / DOC).write_bytes(b"\xff\xfe not utf-8 on main\n")
    _git(repos.seed, "add", DOC)
    _git(repos.seed, "commit", "-q", "-m", "binary doc on main")
    _git(repos.seed, "push", "-q", "origin", "main")
    # local: a valid UTF-8 doc that differs → conflict → read of the main blob.
    repos.write(DOC, _epic("# local text\n"))
    before = repos.snapshot()
    main_before = repos.main_sha()
    rc, out = _land(capsys)
    assert rc == 1 and out["status"] == "undecodable"
    assert repos.snapshot() == before
    assert repos.main_sha() == main_before


def test_push_rejected_by_protected_branch_points_to_hand_opened_pr(
    repos, runner, clean_drift, capsys
) -> None:
    """N3: a protected-branch (GH006) push points at the hand-opened-PR fallback."""
    repos.write(DOC, _epic("# Epic\n"))
    runner.on(
        _is("push", "origin"),
        lambda cmd: _done(cmd, 1, "", "remote: error: GH006: Protected branch update failed"),
    )
    rc, out = _land(capsys)
    assert rc == 1 and out["status"] == "push-rejected"
    # N3: the hint interpolates the real issue number, not a literal `#<N>`.
    assert "PR" in out["hint"] and f"Refs #{ISSUE}" in out["hint"]
    assert "#<N>" not in out["hint"] and f"Fixes #{ISSUE}" in out["hint"]
    assert "re-run" not in out["hint"]


@pytest.mark.parametrize(
    "stderr",
    [
        "remote: error: GH013: Repository rule violations found for refs/heads/main",
        "remote: error: rule violations found",
    ],
    ids=["GH013", "rule-violation"],
)
def test_push_rejected_by_a_ruleset_points_to_hand_opened_pr(
    repos, runner, clean_drift, capsys, stderr: str
) -> None:
    """N3: a repository ruleset (GH013 / rule violation) also routes to the PR fallback."""
    repos.write(DOC, _epic("# Epic\n"))
    runner.on(_is("push", "origin"), lambda cmd: _done(cmd, 1, "", stderr))
    rc, out = _land(capsys)
    assert rc == 1 and out["status"] == "push-rejected"
    assert "PR" in out["hint"] and f"Refs #{ISSUE}" in out["hint"]
    assert "#<N>" not in out["hint"]
    assert "re-run" not in out["hint"]


def test_push_rejected_by_an_unrelated_failure_gives_a_neutral_hint(
    repos, runner, clean_drift, capsys
) -> None:
    """N3: a non-fast-forward-unrelated failure is not 'main moved' — no re-run loop."""
    repos.write(DOC, _epic("# Epic\n"))
    runner.on(_is("push", "origin"), lambda cmd: _done(cmd, 1, "", "fatal: Authentication failed"))
    rc, out = _land(capsys)
    assert rc == 1 and out["status"] == "push-rejected"
    assert "re-run" not in out["hint"]
    assert "PR" not in out["hint"]
    # N7: the neutral hint points at the JSON ``detail`` field, not "detail above".
    assert "inspect `detail`" in out["hint"]
    assert "detail above" not in out["hint"]


def test_push_rejected_by_a_moved_main_then_rerun(repos, runner, clean_drift, capsys) -> None:
    repos.write(DOC, _epic("# Epic\n"))
    moved: list[bool] = []

    def move_main(cmd: list[str]):
        if not moved:
            moved.append(True)
            repos.push_from_seed("README.md", "moved\n", "race")
        return None

    runner.on(_is("push", "origin"), move_main)
    before = repos.snapshot()
    rc, out = _land(capsys)
    assert rc == 1 and out["status"] == "push-rejected" and out["sha"]
    assert repos.snapshot() == before
    assert repos.main_file(DOC) is None

    rc, out = _land(capsys)
    assert rc == 0, out
    assert out["status"] == "landed"
    assert repos.main_file(DOC) == _epic("# Epic\n")
    assert repos.main_file("README.md") == "moved\n"


def test_verify_failed_when_the_commit_is_not_on_main(repos, runner, clean_drift, capsys) -> None:
    repos.write(DOC, _epic("# Epic\n"))
    runner.on(
        lambda cmd: cmd[0] == "git" and "--is-ancestor" in cmd and "HEAD" not in cmd,
        lambda cmd: _done(cmd, 1),
    )
    before = repos.snapshot()
    rc, out = _land(capsys)
    assert rc == 1 and out["status"] == "verify-failed"
    assert repos.snapshot() == before


def test_plumbing_failure_is_git_error_and_cleans_the_temp_index(
    repos, runner, clean_drift, monkeypatch, capsys
) -> None:
    repos.write(DOC, _epic("# Epic\n"))
    seen_index: list[str] = []

    def fail(cmd: list[str]):
        return _done(cmd, 128, "", "fatal: write-tree")

    runner.on(_is("write-tree"), fail)
    original = epic_doc.utils.run_git

    def spy(args: list[str], **kwargs: Any):
        env = kwargs.get("env")
        if env:
            seen_index.append(env["GIT_INDEX_FILE"])
        return original(args, **kwargs)

    monkeypatch.setattr(epic_doc.utils, "run_git", spy)
    _refused(repos, capsys, "git-error")
    assert seen_index and not Path(seen_index[0]).parent.exists()


# ---------------------------------------------------------------------------
# land — the declared SVGs (QS-404)
# ---------------------------------------------------------------------------

SVG = f"docs/epics/img/QS-{ISSUE}-x.svg"
RENDERER = "scripts/qs/mermaid_svg.py"
_RENDERER_SOURCE = (REPO_ROOT / RENDERER).read_text(encoding="utf-8")
_RENDERER_KEY_PREFIX = "qs_mermaid_svg_main_"


def _block(label: str = "A", node: str = "at=10,10 size=100,50", out: str = f"img/QS-{ISSUE}-x.svg") -> str:
    """The inner text of the fixture Mermaid block (between the fences)."""
    return (
        "flowchart TB\n"
        f'    a["{label}"]\n'
        f"    %% @out {out}\n"
        "    %% @canvas 200 100\n"
        f"    %% @node a {node}\n"
    )


_DIAGRAM = f"```mermaid\n{_block()}```\n"


def _diagram_doc(intro: str = "# Epic QS-900\n", *blocks: str) -> str:
    """An epic doc with a valid Decomposition and the given Mermaid blocks (``_block()`` by default)."""
    fences = "".join(f"\n```mermaid\n{b}```\n" for b in (blocks or (_block(),)))
    return _epic(intro + fences)


def _svg(block: str | None = None) -> str:
    return mermaid_svg.render_block(block or _block())[1]


def _seed_renderer(repos: Repos) -> str:
    """Put this worktree's renderer on ``origin/main`` — not a changed path of ``work``."""
    return repos.push_from_seed(RENDERER, _RENDERER_SOURCE, "seed the renderer")


def _pull(repos: Repos) -> None:
    _git(repos.work, "pull", "-q", "--ff-only", "origin", "main")


def _renderer_loaded(runner: Runner) -> bool:
    """Whether ``land`` read main's renderer (``git show <pinned main>:scripts/qs/mermaid_svg.py``)."""
    return any(
        c[0] == "git" and "show" in c and any(arg.endswith(f":{RENDERER}") for arg in c) for c in runner.calls
    )


def _renderer_keys() -> list[str]:
    return [k for k in sys.modules if k.startswith(_RENDERER_KEY_PREFIX)]


def _main_blob_sha(repos: Repos, path: str) -> str:
    return _git(repos.origin, "rev-parse", f"main:{path}").strip()


def test_land_lands_the_doc_with_its_rendered_svg(repos, runner, clean_drift, capsys, tmp_path) -> None:
    """AC 1: the doc changed, the SVG is absent locally and on main → both land."""
    _seed_renderer(repos)
    repos.write(DOC, _diagram_doc())
    rc, out = _land(capsys)
    assert rc == 0, out
    assert out["status"] == "landed"
    assert out["paths"] == [DOC, SVG]
    assert repos.main_file(SVG) == _svg()
    # The landed blob is byte-for-byte what ``mermaid_svg.py render`` writes.
    scratch = tmp_path / "scratch"
    (scratch / "docs" / "epics").mkdir(parents=True)
    (scratch / DOC).write_text(_diagram_doc(), encoding="utf-8")
    assert mermaid_svg.run([scratch / DOC], check=False)[0] == 0
    assert _git(repos.work, "hash-object", str(scratch / SVG)).strip() == _main_blob_sha(repos, SVG)
    assert _git(repos.work, "rev-parse", "HEAD").strip() == repos.main_sha()
    assert _git(repos.work, "status", "--porcelain") == ""
    assert (repos.work / SVG).read_text() == _svg()
    assert _renderer_loaded(runner)
    assert clean_drift == [["--repo-root", str(repos.work), "--json", "--paths", DOC, SVG]]
    assert _renderer_keys() == []


def test_land_the_rendered_bytes_win_over_a_garbage_local_svg(repos, runner, clean_drift, capsys) -> None:
    """AC 2: the landing commit gets the rendered SVG, whatever the working tree holds."""
    _seed_renderer(repos)
    repos.write(DOC, _diagram_doc())
    repos.write(SVG, "<svg>garbage</svg>\n")
    rc, out = _land(capsys)
    assert rc == 0, out
    assert out["paths"] == [DOC, SVG]
    assert repos.main_file(SVG) == _svg()


def test_land_push_rejected_leaves_the_doc_and_a_garbage_svg_untouched(
    repos, runner, clean_drift, capsys
) -> None:
    """AC 2: a failed push never writes the rendered SVG into the working tree."""
    _seed_renderer(repos)
    repos.write(DOC, _diagram_doc())
    repos.write(SVG, "<svg>garbage</svg>\n")
    runner.on(
        _is("push", "origin"),
        lambda cmd: _done(cmd, 1, "", "! [rejected] main -> main (non-fast-forward)"),
    )
    out = _refused(repos, capsys, "push-rejected")
    assert "re-run" in out["hint"]
    assert (repos.work / SVG).read_text() == "<svg>garbage</svg>\n"
    assert (repos.work / DOC).read_text() == _diagram_doc()
    assert repos.main_file(SVG) is None


def test_land_a_locally_deleted_declared_svg_lands_its_rendered_blob(
    repos, runner, clean_drift, capsys
) -> None:
    """AC 3: deleting a declared SVG is not a deletion — its render lands."""
    _seed_renderer(repos)
    repos.push_from_seed(DOC, _diagram_doc())
    repos.push_from_seed(SVG, _svg())
    _pull(repos)
    repos.write(DOC, _diagram_doc("# Epic QS-900\n", _block("B")))
    (repos.work / SVG).unlink()
    rc, out = _land(capsys)
    assert rc == 0, out
    assert out["paths"] == [DOC, SVG]
    assert repos.main_file(SVG) == _svg(_block("B"))


def test_land_a_locally_deleted_declared_svg_alone_is_already_landed(repos, runner, capsys) -> None:
    """AC 3: with the doc unchanged, the render equals main — the reset restores the SVG."""
    _seed_renderer(repos)
    repos.push_from_seed(DOC, _diagram_doc())
    repos.push_from_seed(SVG, _svg())
    _pull(repos)
    (repos.work / SVG).unlink()
    rc, out = _land(capsys)
    assert rc == 0, out
    assert out == {"status": "already-landed", "paths": [SVG], "reset": True}
    assert _renderer_loaded(runner)
    assert (repos.work / SVG).read_text() == _svg()
    assert repos.main_file(SVG) == _svg()


def test_land_refuses_an_undeclared_untracked_svg(repos, runner, capsys) -> None:
    """AC 4: a pattern SVG no hint declares is a leftover render (step 8)."""
    _seed_renderer(repos)
    repos.write(DOC, _diagram_doc())
    leftover = f"docs/epics/img/QS-{ISSUE}-y.svg"
    repos.write(leftover, "<svg/>\n")
    out = _refused(repos, capsys, "out-of-scope")
    assert out["offenders"] == [leftover]
    assert "leftover render" in out["hint"] and "delete it" in out["hint"]
    # Review fix #02 G1: main lacks it, so there is no copy to restore.
    assert "git checkout" not in out["hint"]
    assert _renderer_loaded(runner)


def test_land_refuses_an_undeclared_svg_main_holds_with_a_restore_hint(repos, runner, capsys) -> None:
    """Review fix #02 G1: an edited undeclared SVG main holds is restored, never deleted."""
    _seed_renderer(repos)
    orphan = f"docs/epics/img/QS-{ISSUE}-orphan.svg"
    repos.push_from_seed(orphan, "<svg/>\n")
    _pull(repos)
    repos.write(orphan, "<svg>edited</svg>\n")
    repos.write(DOC, _diagram_doc())
    out = _refused(repos, capsys, "out-of-scope")
    assert out["offenders"] == [orphan]
    assert f"`git checkout origin/main -- {orphan}`" in out["hint"]
    assert "delete it" not in out["hint"]


def test_land_refuses_deleting_an_undeclared_tracked_svg(repos, runner, capsys) -> None:
    """AC 4: deleting a pattern SVG that no hint declares is refused too."""
    _seed_renderer(repos)
    leftover = f"docs/epics/img/QS-{ISSUE}-y.svg"
    repos.push_from_seed(leftover, "<svg/>\n")
    _pull(repos)
    repos.write(DOC, _diagram_doc())
    (repos.work / leftover).unlink()
    out = _refused(repos, capsys, "out-of-scope")
    assert out["offenders"] == [leftover]
    assert "deleting an epic's SVG is not supported" in out["hint"]
    assert f"git checkout origin/main -- {leftover}" in out["hint"]
    assert "delete it" not in out["hint"]


def test_land_refuses_another_epics_svg_before_rendering(repos, runner, capsys) -> None:
    """AC 4: another epic's SVG fails the pre-render scope (step 4)."""
    _seed_renderer(repos)
    repos.write(DOC, _diagram_doc())
    repos.write("docs/epics/img/QS-901-x.svg", "<svg/>\n")
    out = _refused(repos, capsys, "out-of-scope")
    assert out["offenders"] == ["docs/epics/img/QS-901-x.svg"]
    assert f"docs/epics/img/QS-{ISSUE}-*.svg" in out["detail"]
    assert not _renderer_loaded(runner)


def test_land_refuses_an_out_hint_outside_the_epic_pattern(repos, runner, capsys) -> None:
    """AC 5: an ``@out`` must target ``docs/epics/img/QS-<N>-*.svg``."""
    _seed_renderer(repos)
    repos.write(DOC, _diagram_doc("# Epic QS-900\n", _block(out="img/other.svg")))
    out = _refused(repos, capsys, "out-of-scope")
    assert out["offenders"] == ["docs/epics/img/other.svg"]
    assert f"docs/epics/img/QS-{ISSUE}-*.svg" in out["detail"]
    assert not runner.git_calls("push")
    assert _renderer_keys() == []


def test_land_refuses_an_out_hint_escaping_the_repo(repos, runner, capsys) -> None:
    """AC 5: an escaping ``@out`` is a ``../`` offender, never a traceback."""
    _seed_renderer(repos)
    repos.write(DOC, _diagram_doc("# Epic QS-900\n", _block(out="../../../../x.svg")))
    out = _refused(repos, capsys, "out-of-scope")
    (offender,) = out["offenders"]
    assert offender.startswith("../") and offender.endswith("x.svg")
    assert not runner.git_calls("push")
    assert _renderer_keys() == []


@pytest.mark.parametrize(
    "blocks",
    [
        pytest.param((_block(), _block("B")), id="duplicate-out"),
        pytest.param((_block(node="at=1 size=100,50"),), id="bad-node-hint"),
    ],
)
def test_land_refuses_a_broken_diagram(repos, runner, capsys, blocks) -> None:
    """AC 5: a duplicate ``@out`` or a malformed hint is ``diagram-error``."""
    _seed_renderer(repos)
    repos.write(DOC, _diagram_doc("# Epic QS-900\n", *blocks))
    out = _refused(repos, capsys, "diagram-error")
    assert out["detail"]
    assert not runner.git_calls("push")
    assert _renderer_keys() == []


def test_land_refuses_a_diagram_when_main_has_no_renderer(repos, runner, capsys) -> None:
    """AC 5: ``origin/main`` without ``scripts/qs/mermaid_svg.py`` cannot render."""
    repos.write(DOC, _diagram_doc())
    out = _refused(repos, capsys, "diagram-error")
    assert RENDERER in out["detail"]
    assert _renderer_keys() == []


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        pytest.param('raise RuntimeError("broken renderer")\n', "broken renderer", id="raises-at-import"),
        pytest.param("def render_block(block):\n    return None, ''\n", "outputs_from_text", id="predates-qs404"),
        pytest.param("this is not python\n", "", id="syntax-error"),
        pytest.param("import sys\nsys.exit(3)\n", "exited", id="exits-at-import"),
    ],
)
def test_land_refuses_a_diagram_with_a_broken_main_renderer(repos, runner, capsys, source, fragment) -> None:
    """AC 5: a renderer that fails to load leaves no ``sys.modules`` entry behind."""
    repos.push_from_seed(RENDERER, source, "a broken renderer")
    repos.write(DOC, _diagram_doc())
    out = _refused(repos, capsys, "diagram-error")
    assert fragment in out["detail"]
    assert _renderer_keys() == []


def test_land_dry_run_lists_the_svg_and_writes_nothing(repos, runner, clean_drift, capsys) -> None:
    """AC 6: dry-run reports the SVG and leaves every file's bytes and mtime alone."""
    _seed_renderer(repos)
    repos.write(DOC, _diagram_doc())
    repos.write(SVG, "<svg>garbage</svg>\n")
    mtimes = {p: (repos.work / p).stat().st_mtime_ns for p in (DOC, SVG)}
    before = repos.snapshot()
    main_before = repos.main_sha()
    rc, out = _land(capsys, "--dry-run")
    assert rc == 0, out
    assert out["status"] == "ok-dry-run"
    assert out["paths"] == [DOC, SVG]
    assert repos.snapshot() == before
    assert {p: (repos.work / p).stat().st_mtime_ns for p in (DOC, SVG)} == mtimes
    assert repos.main_sha() == main_before


def test_land_dry_run_never_creates_an_absent_svg(repos, runner, clean_drift, capsys) -> None:
    _seed_renderer(repos)
    repos.write(DOC, _diagram_doc())
    rc, out = _land(capsys, "--dry-run")
    assert rc == 0, out
    assert out["paths"] == [DOC, SVG]
    assert not (repos.work / SVG).exists()


def _landed_not_reset(repos: Repos) -> None:
    """AC 7: main holds the doc and its SVG; the doc is local but never pulled."""
    _seed_renderer(repos)
    repos.push_from_seed(DOC, _diagram_doc())
    repos.push_from_seed(SVG, _svg())
    repos.write(DOC, _diagram_doc())
    repos.write(SVG, "<svg>garbage</svg>\n")


def test_land_already_landed_with_a_garbage_svg_resets(repos, runner, capsys) -> None:
    """AC 7: the working-tree shortcut ignores an SVG that main holds."""
    _landed_not_reset(repos)
    rc, out = _land(capsys)
    assert rc == 0, out
    assert out == {"status": "already-landed", "paths": [DOC, SVG], "reset": True}
    assert not _renderer_loaded(runner)
    assert (repos.work / SVG).read_text() == _svg()


def test_land_already_landed_dry_run_does_not_reset(repos, runner, capsys) -> None:
    _landed_not_reset(repos)
    before = repos.snapshot()
    rc, out = _land(capsys, "--dry-run")
    assert rc == 0, out
    assert out == {"status": "ok-dry-run", "already_landed": True, "paths": [DOC, SVG], "reset": False}
    assert not runner.git_calls("reset")
    assert repos.snapshot() == before


def test_land_already_landed_never_loads_a_broken_renderer(repos, runner, capsys) -> None:
    """AC 7: rendering never blocks the cleanup of a landed worktree."""
    _landed_not_reset(repos)
    repos.push_from_seed(RENDERER, 'raise RuntimeError("broken renderer")\n', "break the renderer")
    rc, out = _land(capsys)
    assert rc == 0, out
    assert out["status"] == "already-landed" and out["reset"] is True
    assert not _renderer_loaded(runner)


def test_land_already_landed_with_a_leftover_svg_main_lacks_is_refused(repos, runner, capsys) -> None:
    """AC 7: a reset would never remove an untracked SVG main lacks — refuse it instead."""
    _landed_not_reset(repos)
    leftover = f"docs/epics/img/QS-{ISSUE}-y.svg"
    repos.write(leftover, "<svg/>\n")
    out = _refused(repos, capsys, "out-of-scope")
    assert out["offenders"] == [leftover]
    assert "leftover render" in out["hint"]
    assert not runner.git_calls("reset")


def test_land_nothing_changed_never_loads_the_renderer(repos, runner, capsys) -> None:
    _seed_renderer(repos)
    repos.push_from_seed(DOC, _diagram_doc())
    repos.push_from_seed(SVG, _svg())
    _pull(repos)
    rc, out = _land(capsys)
    assert rc == 0, out
    assert out == {"status": "already-landed", "paths": [], "reset": False}
    assert not _renderer_loaded(runner)


def test_land_renders_mains_doc_when_the_local_doc_is_unchanged(repos, runner, capsys) -> None:
    """AC 8: main moved the doc and its SVG; a stale local SVG lands nothing."""
    _seed_renderer(repos)
    repos.push_from_seed(DOC, _diagram_doc())
    repos.push_from_seed(SVG, _svg())
    _pull(repos)
    repos.push_from_seed(DOC, _diagram_doc("# Epic QS-900\n", _block("B")), "a child PR amends the doc")
    repos.push_from_seed(SVG, _svg(_block("B")), "and re-renders")
    main_svg = _main_blob_sha(repos, SVG)
    repos.write(SVG, "<svg>garbage</svg>\n")
    rc, out = _land(capsys)
    assert rc == 0, out
    assert out == {"status": "already-landed", "paths": [SVG], "reset": True}
    assert _renderer_loaded(runner)
    assert _main_blob_sha(repos, SVG) == main_svg
    assert (repos.work / SVG).read_text() == _svg(_block("B"))


def test_land_a_rerender_alone_lands_mains_doc_svg_without_revalidating(
    repos, runner, clean_drift, capsys
) -> None:
    """D4/D5 step 10: only the SVG changed — main's doc renders, and is not re-validated."""
    _seed_renderer(repos)
    legacy = "# Epic QS-900\n\nno Decomposition table yet\n\n" + _DIAGRAM
    repos.push_from_seed(DOC, legacy)
    repos.push_from_seed(SVG, "<svg>hand-edited</svg>\n")
    _pull(repos)
    repos.write(SVG, _svg())
    rc, out = _land(capsys)
    assert rc == 0, out
    assert out["status"] == "landed"
    assert out["paths"] == [SVG]
    assert repos.main_file(SVG) == _svg()
    assert repos.main_file(DOC) == legacy


def test_land_svg_only_change_equal_to_main_never_renders(repos, runner, capsys) -> None:
    """Review fix #01 F2: every changed path (an SVG) already equals main — reset, no render."""
    _seed_renderer(repos)
    repos.push_from_seed(DOC, _diagram_doc())
    repos.push_from_seed(SVG, _svg())
    _pull(repos)
    repos.push_from_seed(SVG, "<svg>main's re-render</svg>\n", "main re-renders")
    repos.push_from_seed(RENDERER, 'raise RuntimeError("broken renderer")\n', "break the renderer")
    repos.write(SVG, "<svg>main's re-render</svg>\n")
    rc, out = _land(capsys)
    assert rc == 0, out
    assert out == {"status": "already-landed", "paths": [SVG], "reset": True}
    assert not _renderer_loaded(runner)


def test_land_pins_main_so_a_concurrent_fetch_is_a_rejected_push(repos, runner, clean_drift, capsys) -> None:
    """Review fix #01 F4: origin/main moving mid-land never overwrites main's newer doc."""
    _seed_renderer(repos)
    repos.write(DOC, _diagram_doc())
    moved: list[bool] = []

    def move_main(cmd: list[str]):
        if not moved and any(arg.endswith(f":{RENDERER}") for arg in cmd):
            moved.append(True)
            repos.push_from_seed(DOC, _diagram_doc("# Epic QS-900\nmain edit\n"), "a child PR amends the doc")
            _git(repos.work, "fetch", "-q", "origin", "main")  # another session's fetch
        return None

    runner.on(lambda cmd: cmd[0] == "git" and "show" in cmd, move_main)
    rc, out = _land(capsys)
    assert moved
    assert rc == 1, out
    assert out["status"] == "push-rejected"
    assert "main moved" in out["hint"]
    assert repos.main_file(DOC) == _diagram_doc("# Epic QS-900\nmain edit\n")
    rc, out = _land(capsys)
    assert rc == 1 and out["status"] == "conflict"
    assert [c["path"] for c in out["conflicts"]] == [DOC]


def test_land_a_doc_equal_to_main_is_never_a_conflict(repos, runner, clean_drift, capsys) -> None:
    """Review fix #01 F8: main added the doc but not its SVG; the local doc is main's, byte for byte."""
    _seed_renderer(repos)
    repos.push_from_seed(DOC, _diagram_doc())
    repos.write(DOC, _diagram_doc())
    repos.write(SVG, _svg())
    rc, out = _land(capsys)
    assert rc == 0, out
    assert out["status"] == "landed"
    assert out["paths"] == [DOC, SVG]
    assert repos.main_file(SVG) == _svg()


def test_land_an_undeclared_svg_equal_to_main_is_not_refused(repos, runner, clean_drift, capsys) -> None:
    """Review fix #03 H1: main changed an orphan SVG since the base; restoring main's copy clears it."""
    _seed_renderer(repos)
    orphan = f"docs/epics/img/QS-{ISSUE}-orphan.svg"
    repos.push_from_seed(orphan, "<svg>v1</svg>\n")
    _pull(repos)
    repos.push_from_seed(orphan, "<svg>v2 on main</svg>\n", "main edits the orphan")
    repos.write(orphan, "<svg>edited</svg>\n")
    repos.write(DOC, _diagram_doc())
    out = _refused(repos, capsys, "out-of-scope")
    assert out["offenders"] == [orphan]
    # Follow the hint literally: the path then equals main but still differs from the base.
    _git(repos.work, "fetch", "-q", "origin", "main")
    _git(repos.work, "checkout", "origin/main", "--", orphan)
    assert orphan in _git(repos.work, "status", "--porcelain")
    rc, out = _land(capsys)
    assert rc == 0, out
    assert out["status"] == "landed"
    assert repos.main_file(orphan) == "<svg>v2 on main</svg>\n"
    assert repos.main_file(SVG) == _svg()


def test_land_an_undeclared_svg_deleted_on_both_sides_is_not_refused(repos, runner, clean_drift, capsys) -> None:
    """Review fix #03 H1: main deleted the orphan too — nothing to delete, nothing to restore."""
    _seed_renderer(repos)
    orphan = f"docs/epics/img/QS-{ISSUE}-orphan.svg"
    repos.push_from_seed(orphan, "<svg>v1</svg>\n")
    _pull(repos)
    repos.push_from_seed(orphan, None, "main drops the orphan")
    (repos.work / orphan).unlink()
    repos.write(DOC, _diagram_doc())
    rc, out = _land(capsys)
    assert rc == 0, out
    assert out["status"] == "landed"
    assert repos.main_file(orphan) is None


def test_land_refuses_an_undeclared_svg_with_both_hints(repos, runner, capsys) -> None:
    """Review fix #01 F7: a leftover and a deletion in one refusal get both hints."""
    _seed_renderer(repos)
    deleted = f"docs/epics/img/QS-{ISSUE}-old.svg"
    repos.push_from_seed(deleted, "<svg/>\n")
    _pull(repos)
    (repos.work / deleted).unlink()
    leftover = f"docs/epics/img/QS-{ISSUE}-y.svg"
    repos.write(leftover, "<svg/>\n")
    repos.write(DOC, _diagram_doc())
    out = _refused(repos, capsys, "out-of-scope")
    assert out["offenders"] == [deleted, leftover]
    assert "leftover render" in out["hint"]
    assert "deleting an epic's SVG is not supported" in out["hint"]


def test_land_a_svg_main_rerendered_is_never_a_conflict(repos, runner, clean_drift, capsys) -> None:
    """AC 9 (D13): main changed only the SVG since the base — land re-renders it."""
    _seed_renderer(repos)
    repos.push_from_seed(DOC, _diagram_doc())
    repos.push_from_seed(SVG, _svg())
    _pull(repos)
    repos.push_from_seed(SVG, "<svg>a different render</svg>\n", "main re-renders")
    repos.write(DOC, _diagram_doc("# Epic QS-900\n", _block("B")))
    rc, out = _land(capsys)
    assert rc == 0, out
    assert out["status"] == "landed"
    assert out["paths"] == [DOC, SVG]
    assert repos.main_file(SVG) == _svg(_block("B"))


def test_land_conflicts_list_only_the_doc_and_merged_lands_the_rerender(
    repos, runner, clean_drift, capsys
) -> None:
    """AC 9 (D13): main changed the doc — only the doc conflicts; ``--merged`` lands both."""
    _seed_renderer(repos)
    repos.push_from_seed(DOC, _diagram_doc())
    repos.push_from_seed(SVG, _svg())
    _pull(repos)
    repos.push_from_seed(DOC, _diagram_doc("# Epic QS-900\nmain edit\n"), "a child PR amends the doc")
    repos.push_from_seed(SVG, "<svg>main's render</svg>\n", "and its SVG")
    repos.write(DOC, _diagram_doc("# Epic QS-900\n", _block("B")))
    out = _refused(repos, capsys, "conflict")
    assert [c["path"] for c in out["conflicts"]] == [DOC]
    blob = out["conflicts"][0]["main_blob"]
    repos.write(DOC, _diagram_doc("# Epic QS-900\nmain edit\n", _block("B")))
    rc, out = _land(capsys, "--merged", f"{DOC}={blob}")
    assert rc == 0, out
    assert out["status"] == "landed"
    assert out["paths"] == [DOC, SVG]
    assert repos.main_file(DOC) == _diagram_doc("# Epic QS-900\nmain edit\n", _block("B"))
    assert repos.main_file(SVG) == _svg(_block("B"))


def test_land_a_doc_without_out_never_loads_the_renderer(repos, runner, clean_drift, capsys) -> None:
    """AC 10: a doc without the substring ``@out`` lands exactly as before."""
    repos.write(DOC, _epic("# Epic QS-900\n\n```mermaid\nflowchart TB\n    a[\"A\"]\n```\n"))
    rc, out = _land(capsys)
    assert rc == 0, out
    assert out["paths"] == [DOC]
    assert not _renderer_loaded(runner)


def test_land_a_doc_mentioning_out_in_prose_lands_alone(repos, runner, clean_drift, capsys) -> None:
    """AC 10 / D4: ``@out`` in prose loads the renderer, which declares nothing."""
    _seed_renderer(repos)
    repos.write(DOC, _epic("# Epic QS-900\n\nan `@out` hint names the SVG.\n"))
    rc, out = _land(capsys)
    assert rc == 0, out
    assert out["paths"] == [DOC]
    assert _renderer_loaded(runner)
    assert _renderer_keys() == []


def test_status_counts_a_changed_svg(repos, runner, capsys) -> None:
    """AC 11: an SVG path is a changed path like any other."""
    repos.write(SVG, "<svg/>\n")
    rc, out = _status(capsys)
    assert rc == 0
    assert SVG in out["changed"]
    assert out["safe_to_discard"] is False


def test_status_landed_but_unreset_with_its_svg_is_safe_to_discard(repos, runner, capsys) -> None:
    """AC 11: the doc and its SVG both equal main → landed, not reset, safe to discard."""
    repos.push_from_seed(DOC, _diagram_doc())
    repos.push_from_seed(SVG, _svg())
    repos.write(DOC, _diagram_doc())
    repos.write(SVG, _svg())
    rc, out = _status(capsys)
    assert rc == 0
    assert sorted(out["changed"]) == [DOC, SVG]
    assert out["landed_not_reset"] is True
    assert out["safe_to_discard"] is True


def test_land_refuses_an_undecodable_doc_that_declares_an_svg(repos, runner, capsys) -> None:
    """AC 11: a non-UTF-8 doc is ``undecodable`` before any render."""
    _seed_renderer(repos)
    (repos.work / DOC).parent.mkdir(parents=True, exist_ok=True)
    (repos.work / DOC).write_bytes(_diagram_doc().encode() + b"\xff\xfe\n")
    out = _refused(repos, capsys, "undecodable")
    assert out["detail"] == DOC
    assert not _renderer_loaded(runner)


def test_load_renderer_runs_the_real_source() -> None:
    """AC 11b: the loader execs the real renderer into a fresh, removable module."""
    for _ in range(2):
        module, key = epic_doc._load_renderer(_RENDERER_SOURCE)
        try:
            assert key.startswith(_RENDERER_KEY_PREFIX)
            assert sys.modules[key] is module
            ((path, svg),) = module.outputs_from_text(_DIAGRAM, Path("/x/docs/epics"))
            assert path == Path(f"/x/docs/epics/img/QS-{ISSUE}-x.svg").resolve()
            assert svg == _svg()
        finally:
            sys.modules.pop(key)
        assert key not in sys.modules


def test_load_renderer_removes_its_module_on_failure() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        epic_doc._load_renderer('raise RuntimeError("boom")\n')
    with pytest.raises(epic_doc.Refusal) as info:
        epic_doc._load_renderer("X = 1\n")
    assert info.value.status == "diagram-error"
    assert _renderer_keys() == []


def test_svg_pattern_matches_only_the_epics_own_flat_svgs() -> None:
    pattern = epic_doc._svg_pattern(ISSUE)
    assert pattern.fullmatch(f"docs/epics/img/QS-{ISSUE}-big-picture.svg")
    assert pattern.fullmatch(f"docs/epics/img/QS-{ISSUE}-v1.2_x.svg")
    for path in (
        f"docs/epics/img/QS-{ISSUE}0-x.svg",
        f"docs/epics/img/QS-{ISSUE}-.svg",
        f"docs/epics/img/QS-{ISSUE}-a/b.svg",
        f"docs/epics/QS-{ISSUE}-x.svg",
        f"docs/epics/img/QS-{ISSUE}-x.svg.bak",
        "docs/epics/img/QS-1-x.svg",
    ):
        assert not pattern.fullmatch(path), path


# ---------------------------------------------------------------------------
# parse_decomposition
# ---------------------------------------------------------------------------


# The live doc evolves: every child filed through ``epic_doc.py land``
# (which runs no pytest) rewrites its ``not filed`` cell with the new
# number. So the live-file test asserts only invariants that never change
# — filed numbers stay filed — and the exact shape is frozen on the inline
# fixture below, copied from the current table (M2).
_369_TABLE_FIXTURE = (
    "# Epic\n\n## Decomposition\n\n"
    "### Shared (lane-agnostic)\n\n"
    "| # | child | kind | issue |\n"
    "|---|---|---|---|\n"
    "| 1 | Make `--impacted` match what CI checks | bug | "
    "[#371](https://github.com/tmenguy/quiet-solar/issues/371) |\n"
    "| 2 | Extract the handoff plumbing into a macro | feature | "
    "[#372](https://github.com/tmenguy/quiet-solar/issues/372) |\n"
    "| 3 | The loop engine | feature | not filed |\n"
)


def test_the_real_qs369_doc_parses_as_is() -> None:
    """The live doc parses; assert only invariants (filed numbers never change)."""
    rows = epic_doc.parse_decomposition((REPO_ROOT / "docs/epics/QS-369.md").read_text())
    assert rows, "the live doc's Decomposition section must be non-empty"
    assert all(r.child for r in rows), "every row must carry a non-empty child cell"
    assert {371, 372} <= {r.issue for r in rows if r.issue is not None}


def test_369_table_fixture_parses_to_the_expected_shape() -> None:
    """Exact-shape assertion is frozen on the inline fixture, not the live file."""
    rows = epic_doc.parse_decomposition(_369_TABLE_FIXTURE)
    assert len(rows) == 3
    assert [r.issue for r in rows] == [371, 372, None]
    assert rows[0].child.startswith("Make `--impacted` match")


def test_parse_link_cell_escaped_pipe_and_section_end() -> None:
    text = (
        "# T\n\n## Decomposition\n\n"
        "| # | Child | Kind | Wave | Issue |\n|---|---|---|:-:|---|\n"
        "| 1 | a \\| b | feature | 1 | [#12](https://x/12) |\n"
        "| 2 | c | bug | 2 | Not Filed |\n\n"
        "| issue | what |\n|---|---|\n| #5 | ignored: no child column |\n\n"
        "## Next\n\n| child | issue |\n|---|---|\n| after | #99 |\n"
    )
    rows = epic_doc.parse_decomposition(text)
    assert rows == [epic_doc.Row("a \\| b", 12), epic_doc.Row("c", None)]


@pytest.mark.parametrize(
    "table",
    [
        "| child | issue |\n|---|---|\n| a | #1 | extra |\n",
        "| child | issue |\n|---|---|\n| a | soon |\n",
        "| child | issue |\n| a | #1 |\n",
        "| child | issue |\n",
    ],
    ids=["ragged", "bad-cell", "no-separator", "header-only"],
)
def test_parse_refuses_unreadable_tables(table: str) -> None:
    with pytest.raises(epic_doc.UnparseableDecomposition):
        epic_doc.parse_decomposition(f"## Decomposition\n\n{table}")


def test_parse_refuses_a_missing_section() -> None:
    with pytest.raises(epic_doc.UnparseableDecomposition):
        epic_doc.parse_decomposition("# no decomposition here\n")


@pytest.mark.parametrize(
    "table",
    [
        "| child | issue |\n|---|---|\n|  | #1 |\n",
        "| child | issue |\n|---|---|\n| a | #0 |\n",
        "| child | issue |\n|---|---|\n| a | [#0](https://x/0) |\n",
    ],
    ids=["empty-child", "zero-issue", "zero-issue-link"],
)
def test_parse_refuses_empty_child_and_sub_one_issue(table: str) -> None:
    """N1: an empty child cell or an issue number below 1 is unparseable."""
    with pytest.raises(epic_doc.UnparseableDecomposition):
        epic_doc.parse_decomposition(f"## Decomposition\n\n{table}")


# ---------------------------------------------------------------------------
# sync_body
# ---------------------------------------------------------------------------

R = epic_doc.Row

_369_BODY = (
    "**Rationale document:** [`docs/epics/QS-369.md`](https://x/blob/main/docs/epics/QS-369.md)\n"
    "\n"
    "## Acceptance\n"
    "\n"
    "- [ ] the loop converges\n"
    "- [x] #999 unrelated tracking box\n"
    "\n"
    "## Children\n"
    "\n"
    "Filed so far (the rest are filed when their turn comes):\n"
    "\n"
    "- [x] #371 — child 1: `--impacted` matches CI (PR #373)\n"
    "- [x] #372 — child 2: extract handoff plumbing (PR #374)\n"
    "- [ ] #375 — child 3: loop engine\n"
    "\n"
    "### Shared (lane-agnostic)\n"
    "1. **Make `--impacted` match** prose for unfiled children.\n"
)
_369_ROWS = [
    R("Make `--impacted` match", 371),
    R("Extract the handoff plumbing", 372),
    R("The loop engine", 375),
    R("feature × product build", 376),
    R("bug × product build", None),
]


def test_369_shape_adds_only_the_missing_child_and_no_owned_lines() -> None:
    new, added, owned = epic_doc.sync_body(_369_BODY, _369_ROWS, link=None)
    assert added == [376] and owned == []
    expected = _369_BODY.replace(
        "- [ ] #375 — child 3: loop engine\n",
        "- [ ] #375 — child 3: loop engine\n- [ ] #376 — feature × product build\n",
    )
    assert new == expected
    # Idempotent: the second run changes nothing.
    assert epic_doc.sync_body(new, _369_ROWS, link=None) == (new, [], [])


def test_unrelated_checklist_is_never_the_anchor() -> None:
    body = "- [ ] first acceptance box\n- [ ] second\n\nprose\n"
    new, added, owned = epic_doc.sync_body(body, [R("c", 5), R("d", None)], link=None)
    assert new == body + "\n## Children\n\n- [ ] #5 — c\n- (not filed) d\n"
    assert added == [5] and owned == ["d"]


def test_new_children_anchor_inside_children_not_a_sketch_list() -> None:
    """S7(a): a sketch list mentioning a filed number must not capture new children."""
    body = (
        "## Decomposition sketch\n\n"
        "- [ ] #371 — sketch reference\n"
        "\n"
        "## Children\n\n"
        "- [ ] #371 — child one\n"
    )
    rows = [R("child one", 371), R("child two", 372)]
    new, added, owned = epic_doc.sync_body(body, rows, link=None)
    assert added == [372] and owned == []
    # #372 lands under ## Children, not appended to the sketch block.
    children_idx = new.index("## Children")
    sketch_idx = new.index("## Decomposition sketch")
    assert new.index("#372") > children_idx
    assert new.count("#372") == 1
    # the sketch block keeps its single #371 reference
    assert new[sketch_idx:children_idx].count("#371") == 1
    assert epic_doc.sync_body(new, rows, link=None)[0] == new


def test_owned_block_before_task_block_decides_ownership_for_the_section() -> None:
    """S1: a ``(not filed)`` block first and the task-line anchor second — ownership
    is a property of the whole ## Children section, so a just-filed child is not
    left listed as unfiled too, and a second run is a no-op."""
    body = "Intro\n\n## Children\n\n- (not filed) B\n- (not filed) C\n\n- [ ] #371 — A\n"
    rows = [R("A", 371), R("B", 380), R("C", None)]
    new, added, owned = epic_doc.sync_body(body, rows, link=None)
    assert added == [380] and owned == ["C"]
    assert new.count("(not filed) B") == 0  # B is filed now, not re-listed as unfiled
    assert new.count("#380") == 1
    assert new.count("(not filed) C") == 1
    assert epic_doc.sync_body(new, rows, link=None)[0] == new


def test_s2_owned_lines_under_heading_survive_a_later_anchor() -> None:
    """S2: a ``(not filed)`` block directly under ## Children with the task-line
    anchor under a later "Wave 1:" sub-list. Three runs:

    1. table unchanged → the body must NOT be rewritten (no relocation);
    2. after B is filed as #380 → B moves to a task line, C stays ``(not filed)``
       under the heading, and B is never listed both ways;
    3. re-run → a no-op.
    """
    body = "## Children\n\n- (not filed) B\n- (not filed) C\n\nWave 1:\n\n- [ ] #371 — A\n"

    # Run 1: table unchanged (A filed, B and C not).
    rows1 = [R("A", 371), R("B", None), R("C", None)]
    new1, added1, _owned1 = epic_doc.sync_body(body, rows1, link=None)
    assert new1 == body, "run 1 must not rewrite an unchanged body"
    assert added1 == []

    # Run 2: B is filed as #380.
    rows2 = [R("A", 371), R("B", 380), R("C", None)]
    new2, added2, owned2 = epic_doc.sync_body(new1, rows2, link=None)
    assert added2 == [380] and owned2 == ["C"]
    assert new2.count("(not filed) B") == 0  # B no longer listed as unfiled
    assert new2.count("#380") == 1
    assert new2.count("(not filed) C") == 1
    # C stays under the heading; B's new task line sits under "Wave 1:".
    assert new2.index("(not filed) C") < new2.index("Wave 1:")
    assert new2.index("#380") > new2.index("Wave 1:")

    # Run 3: no-op.
    assert epic_doc.sync_body(new2, rows2, link=None)[0] == new2


_S1_04_BODIES = [
    "## Children\n\nWave 1:\n\n- [ ] #371 — A\n\nWave 2:\n\n- (not filed) B\n",
    "## Children\n\nRemaining:\n\n- (not filed) B\n\n- [ ] #371 — A\n",
    "## Children\n\nWave 1 (filed first):\n\n- (not filed) B\n- [ ] #371 — A\n",
]


@pytest.mark.parametrize("body", _S1_04_BODIES)
def test_s1_prose_label_between_heading_and_blocks_owns_the_section(body: str) -> None:
    """S1 (#04): a wave/prose label between ``## Children`` and its list blocks
    makes ``_owned_by_heading`` False for every block, yet the section still owns
    its ``(not filed)`` lines. Before the fix, filing a child there left it listed
    both as ``- [ ] #N`` and ``- (not filed)`` and never self-corrected."""
    # Run 1: the table is unchanged (A filed #371, B not filed). The canonical
    # form (a task line before its ``(not filed)`` siblings) is stable, so a
    # re-run is a no-op and B keeps exactly one unfiled listing.
    rows1 = [R("A", 371), R("B", None)]
    once, added1, _owned1 = epic_doc.sync_body(body, rows1, link=None)
    assert added1 == []
    assert once.count("(not filed) B") == 1
    assert epic_doc.sync_body(once, rows1, link=None)[0] == once

    # Run 2: B is filed as #380 — the ``(not filed) B`` line is stripped, so B is
    # never listed both ways (the bug this fix closes).
    rows2 = [R("A", 371), R("B", 380)]
    filed, added2, owned2 = epic_doc.sync_body(once, rows2, link=None)
    assert added2 == [380] and owned2 == []
    assert filed.count("(not filed) B") == 0
    assert filed.count("#380") == 1
    assert "- [ ] #380 — B" in filed
    assert "#371 — A" in filed

    # Run 3: a no-op.
    assert epic_doc.sync_body(filed, rows2, link=None)[0] == filed


def test_owned_lines_split_across_blocks_do_not_duplicate() -> None:
    """S7(b): a not-filed list split by a blank line regenerates once, no stale dupes."""
    body = (
        "## Children\n\n"
        "- (not filed) alpha\n"
        "- (not filed) beta\n"
        "\n"
        "- (not filed) gamma\n"
    )
    rows = [R("alpha", None), R("beta", None), R("gamma", None)]
    new, added, owned = epic_doc.sync_body(body, rows, link=None)
    assert added == [] and owned == ["alpha", "beta", "gamma"]
    assert new.count("(not filed) alpha") == 1
    assert new.count("(not filed) gamma") == 1
    assert epic_doc.sync_body(new, rows, link=None)[0] == new


def test_no_block_body_gets_a_children_section_with_owned_lines() -> None:
    body = "Intro.\n"
    new, added, owned = epic_doc.sync_body(body, [R("one", None), R("two", None)], link=None)
    assert new == "Intro.\n\n## Children\n\n- (not filed) one\n- (not filed) two\n"
    assert owned == ["one", "two"] and added == []
    assert epic_doc.sync_body(new, [R("one", None), R("two", None)], link=None)[0] == new


def test_owned_lines_regenerate_on_rename_drop_and_filing() -> None:
    body = "## Children\n\n- (not filed) alpha\n- (not filed) beta\n- (not filed) gamma\n"
    rows = [R("alpha", 10), R("beta renamed", None)]  # gamma dropped, alpha filed
    new, added, owned = epic_doc.sync_body(body, rows, link=None)
    assert new == "## Children\n\n- [ ] #10 — alpha\n- (not filed) beta renamed\n"
    assert added == [10] and owned == ["beta renamed"]


def test_owned_lines_reappear_after_a_fully_filed_wave() -> None:
    body = "## Children\n\n- [ ] #10 — alpha\n- [x] #11 — beta (PR #12)\n"
    rows = [R("alpha", 10), R("beta", 11), R("wave two", None)]
    new, added, owned = epic_doc.sync_body(body, rows, link=None)
    assert new == body + "- (not filed) wave two\n"
    assert owned == ["wave two"] and added == []


def test_fully_filed_owned_block_with_nothing_left_is_stable() -> None:
    body = "## Children\n\n- (not filed) alpha\n\n## After\n"
    rows: list[epic_doc.Row] = []
    new, _added, owned = epic_doc.sync_body(body, rows, link=None)
    assert owned == [] and "(not filed)" not in new
    assert epic_doc.sync_body(new, rows, link=None)[0] == new


def test_task_list_match_is_exact_on_the_number() -> None:
    body = "## Children\n\n- [ ] #371 — something else\n"
    new, added, _owned = epic_doc.sync_body(body, [R("thirty-seven", 37), R("x", 371)], link=None)
    assert added == [37]
    assert "- [ ] #37 — thirty-seven" in new


def test_other_lines_are_byte_preserved_crlf() -> None:
    body = "Top line  \r\n\r\n- [x] #4 — done  (PR #9)\r\ntrailing prose"
    new, added, _owned = epic_doc.sync_body(body, [R("done", 4), R("next", 5)], link="LINK")
    assert new == "LINK\r\n\r\nTop line  \r\n\r\n- [x] #4 — done  (PR #9)\r\n- [ ] #5 — next\r\ntrailing prose"
    assert added == [5]


def test_block_at_end_of_body_without_trailing_newline() -> None:
    body = "## Children\n\n- [ ] #4 — done"
    new, _added, _owned = epic_doc.sync_body(body, [R("done", 4), R("next", 5)], link=None)
    assert new == "## Children\n\n- [ ] #4 — done\n- [ ] #5 — next"


def test_empty_children_section_gets_the_block_directly_under_it() -> None:
    body = "## Children\n\n## Related\n- QS-321\n"
    new, _added, owned = epic_doc.sync_body(body, [R("a", 3), R("b", None)], link=None)
    assert new == "## Children\n\n- [ ] #3 — a\n- (not filed) b\n\n## Related\n- QS-321\n"
    assert owned == ["b"]


def test_prose_children_section_gets_a_block_at_its_end_without_owned_lines() -> None:
    body = "## Children\n\nChildren are filed later.\n\n## Related\n"
    new, _added, owned = epic_doc.sync_body(body, [R("a", 3), R("b", None)], link=None)
    assert new == "## Children\n\nChildren are filed later.\n\n- [ ] #3 — a\n\n## Related\n"
    assert owned == []


def test_prose_children_section_with_only_unfiled_rows_is_untouched() -> None:
    body = "## Children\n\nChildren are filed later.\n"
    assert epic_doc.sync_body(body, [R("b", None)], link=None) == (body, [], [])


def test_empty_body_with_rows() -> None:
    new, _added, owned = epic_doc.sync_body("", [R("a", None)], link=None)
    assert new == "## Children\n\n- (not filed) a\n"
    assert owned == ["a"]


def test_new_section_after_a_body_without_trailing_newline() -> None:
    new, _added, _owned = epic_doc.sync_body("Intro.", [R("a", 3)], link=None)
    assert new == "Intro.\n\n## Children\n\n- [ ] #3 — a"


def test_parse_rows_without_a_trailing_pipe() -> None:
    rows = epic_doc.parse_decomposition(
        "## Decomposition\n\n| child | issue\n|---|---\n| a | #4\n"
    )
    assert rows == [epic_doc.Row("a", 4)]


def test_no_rows_no_link_is_a_no_op() -> None:
    assert epic_doc.sync_body("text\n", [], link=None) == ("text\n", [], [])


def test_duplicate_filed_rows_add_one_line() -> None:
    new, added, _owned = epic_doc.sync_body("x\n", [R("a", 3), R("a again", 3)], link=None)
    assert added == [3]
    assert new.count("#3") == 1


# ---------------------------------------------------------------------------
# sync-issue CLI
# ---------------------------------------------------------------------------

_DOC_TEXT = (
    f"# Epic QS-{ISSUE}\n\n## Decomposition\n\n"
    "| # | child | kind | wave | issue |\n|---|---|---|---|---|\n"
    "| 1 | first child | feature | 1 | #901 |\n"
    "| 2 | second child | bug | 2 | not filed |\n"
)


def _sync(capsys) -> tuple[int, dict]:
    return _run(["sync-issue", "--issue", str(ISSUE)], capsys)


def test_sync_issue_prepends_the_link_once_and_writes_via_a_temp_file(
    repos, runner, capsys
) -> None:
    repos.write(DOC, _DOC_TEXT)
    runner.body = "Epic intro.\n"
    rc, out = _sync(capsys)
    assert rc == 0, out
    assert out == {
        "status": "synced", "state": "OPEN", "added": [901],
        "owned": ["second child"], "link_added": True,
    }
    (edit,) = runner.edits
    assert edit["body"].startswith(
        f"**Rationale document:** [{DOC}]({REPO_URL}/blob/main/{DOC})\n\nEpic intro.\n"
    )
    assert "- [ ] #901 — first child\n- (not filed) second child" in edit["body"]
    # The temp body file lived outside the worktree and is gone.
    assert not edit["path"].is_relative_to(repos.work)
    assert not edit["path"].exists()

    # Re-run over the edited body: no `gh issue edit` call at all.
    runner.body = edit["body"]
    rc, out = _sync(capsys)
    assert rc == 0
    assert out["status"] == "unchanged" and out["link_added"] is False
    assert len(runner.edits) == 1
    assert not [c for c in runner.calls if c[:2] == ["gh", "repo"]][1:]


def test_sync_issue_reports_a_closed_state(repos, runner, capsys) -> None:
    repos.write(DOC, _DOC_TEXT)
    runner.state = "CLOSED"
    runner.body = f"see {DOC}\n\n## Children\n\n- [ ] #901 — first child\n- (not filed) second child\n"
    rc, out = _sync(capsys)
    assert rc == 0 and out["state"] == "CLOSED" and out["status"] == "unchanged"


def test_sync_issue_unparseable_writes_nothing(repos, runner, capsys) -> None:
    repos.write(DOC, "## Decomposition\n\n| child | issue |\n|---|---|\n| a | maybe |\n")
    rc, out = _sync(capsys)
    assert rc == 1 and out["status"] == "unparseable-decomposition"
    assert runner.edits == []


def test_sync_issue_missing_doc(repos, runner, capsys) -> None:
    rc, out = _sync(capsys)
    assert rc == 1 and out["status"] == "missing-doc"


def test_sync_issue_refuses_a_non_epic(repos, runner, capsys) -> None:
    runner.labels = ["kind:bug", "target:factory", "scale:task"]
    rc, out = _sync(capsys)
    assert rc == 1 and out["status"] == "not-an-epic"


def test_sync_issue_edit_failure(repos, runner, capsys) -> None:
    repos.write(DOC, _DOC_TEXT)
    runner.body = f"{DOC}\n"
    runner.edit_rc = 1
    rc, out = _sync(capsys)
    assert rc == 1 and out["status"] == "edit-failed" and out["detail"] == "denied"
    assert not runner.edits[0]["path"].exists()


@pytest.mark.parametrize(("repo_rc", "repo_stdout"), [(1, None), (0, "{}"), (0, "nope")])
def test_sync_issue_repo_url_failure(repos, runner, capsys, repo_rc, repo_stdout) -> None:
    repos.write(DOC, _DOC_TEXT)
    runner.repo_rc = repo_rc
    runner.repo_stdout = repo_stdout
    rc, out = _sync(capsys)
    assert rc == 1 and out["status"] == "lookup-failed"
    assert runner.edits == []


def _rewrite(capsys, path: Path) -> tuple[int, dict]:
    return _run(["sync-issue", "--issue", str(ISSUE), "--rewrite-from", str(path)], capsys)


_LINK = f"**Rationale document:** [{DOC}]({REPO_URL}/blob/main/{DOC})"


def test_sync_issue_rewrite_replaces_the_stale_body(repos, runner, capsys, tmp_path) -> None:
    repos.write(DOC, _DOC_TEXT)
    runner.body = f"{_LINK}\n\nStale design text.\n\n## Children\n\n- [x] #901 — first child\n- (not filed) second child\n"
    body_file = tmp_path / "body.md"
    body_file.write_text(
        f"{_LINK}\n\nNew design.\n\n## Children\n\n- [x] #901 — first child\n- (not filed) second child\n",
        encoding="utf-8",
    )
    rc, out = _rewrite(capsys, body_file)
    assert rc == 0, out
    assert out == {
        "status": "synced", "state": "OPEN", "added": [],
        "owned": ["second child"], "link_added": False, "rewrite_from": str(body_file),
    }
    (edit,) = runner.edits
    assert edit["body"] == body_file.read_text(encoding="utf-8")
    assert "Stale design text." not in edit["body"]
    assert body_file.exists()  # the caller's file is never deleted


def test_sync_issue_rewrite_still_enforces_link_and_children(repos, runner, capsys, tmp_path) -> None:
    repos.write(DOC, _DOC_TEXT)
    runner.body = f"{_LINK}\n\nOld.\n"
    body_file = tmp_path / "body.md"
    body_file.write_text("New design only.\n", encoding="utf-8")
    rc, out = _rewrite(capsys, body_file)
    assert rc == 0, out
    assert out["status"] == "synced" and out["rewrite_from"] == str(body_file)
    assert out["link_added"] is True and out["added"] == [901]
    (edit,) = runner.edits
    assert edit["body"].startswith(f"{_LINK}\n\nNew design only.\n")
    assert "- [ ] #901 — first child\n- (not filed) second child" in edit["body"]
    assert "Old." not in edit["body"]


def test_sync_issue_rewrite_prose_mention_of_the_doc_is_not_the_link(repos, runner, capsys, tmp_path) -> None:
    repos.write(DOC, _DOC_TEXT)
    runner.body = f"{_LINK}\n\nOld.\n"
    body_file = tmp_path / "body.md"
    body_file.write_text(f"Redesign: see {DOC} section 3.\n", encoding="utf-8")
    rc, out = _rewrite(capsys, body_file)
    assert rc == 0 and out["link_added"] is True, out
    (edit,) = runner.edits
    assert edit["body"].startswith(f"{_LINK}\n\nRedesign: see {DOC} section 3.\n")


@pytest.mark.parametrize(
    "link",
    [
        f"**Rationale document:** [`{DOC}`]({REPO_URL}/blob/main/{DOC}) — the #369 form",
        f"See [the rationale]({REPO_URL}/blob/main/{DOC}#decomposition).",
        f"[r](./{DOC} \"titled\")",
        f"Rationale: {REPO_URL}/blob/main/{DOC}",
        f"Rationale: <{REPO_URL}/blob/main/{DOC}>",
        f"[r][1]\n\n[1]: {REPO_URL}/blob/main/{DOC}",
    ],
)
def test_sync_issue_rewrite_any_link_to_the_doc_counts(repos, runner, capsys, tmp_path, link) -> None:
    repos.write(DOC, _DOC_TEXT)
    runner.body = f"{link}\n\n## Children\n\n- [ ] #901 — first child\n- (not filed) second child\n"
    body_file = tmp_path / "body.md"
    body_file.write_text(runner.body, encoding="utf-8")
    rc, out = _rewrite(capsys, body_file)
    assert rc == 0 and out["status"] == "unchanged" and out["link_added"] is False, out
    assert runner.edits == []


def test_sync_issue_rewrite_equal_to_current_body_is_unchanged(repos, runner, capsys, tmp_path) -> None:
    repos.write(DOC, _DOC_TEXT)
    runner.body = f"{_LINK}\n\n## Children\n\n- [ ] #901 — first child\n- (not filed) second child\n"
    body_file = tmp_path / "body.md"
    body_file.write_text(runner.body, encoding="utf-8")
    rc, out = _rewrite(capsys, body_file)
    assert rc == 0 and out["status"] == "unchanged" and out["rewrite_from"] == str(body_file)
    assert runner.edits == []


def test_sync_issue_rewrite_missing_file_writes_nothing(repos, runner, capsys, tmp_path) -> None:
    repos.write(DOC, _DOC_TEXT)
    missing = tmp_path / "nope.md"
    rc, out = _rewrite(capsys, missing)
    assert rc == 1 and out["status"] == "missing-body-file" and out["path"] == str(missing)
    assert runner.edits == []


@pytest.mark.parametrize("kind", ["directory", "undecodable"])
def test_sync_issue_rewrite_unreadable_file_writes_nothing(repos, runner, capsys, tmp_path, kind) -> None:
    repos.write(DOC, _DOC_TEXT)
    body_file = tmp_path / "body.md"
    if kind == "directory":
        body_file.mkdir()
    else:
        body_file.write_bytes(b"\xff\xfe\xfa not utf-8")
    rc, out = _rewrite(capsys, body_file)
    assert rc == 1 and out["status"] == "missing-body-file"
    assert runner.edits == []


def test_sync_issue_rewrite_reads_stdin(repos, runner, capsys, monkeypatch) -> None:
    repos.write(DOC, _DOC_TEXT)
    runner.body = f"{_LINK}\n\nOld.\n"
    stdin = io.TextIOWrapper(io.BytesIO(f"\ufeff{_LINK}\n\nFrom stdin — utf-8.\n".encode()))
    monkeypatch.setattr(sys, "stdin", stdin)
    rc, out = _run(["sync-issue", "--issue", str(ISSUE), "--rewrite-from", "-"], capsys)
    assert rc == 0, out
    assert out["rewrite_from"] == "-" and out["added"] == [901]
    (edit,) = runner.edits
    assert edit["body"].startswith(f"{_LINK}\n\nFrom stdin — utf-8.\n")
    assert "Old." not in edit["body"]


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def _closed_stdin() -> io.StringIO:
    stream = io.StringIO("body\n")
    stream.close()
    return stream


@pytest.mark.parametrize("stdin", [None, _Tty("body\n"), _closed_stdin(), io.StringIO("text-only, no .buffer\n")])
def test_sync_issue_rewrite_refuses_closed_or_interactive_stdin(repos, runner, capsys, monkeypatch, stdin) -> None:
    repos.write(DOC, _DOC_TEXT)
    monkeypatch.setattr(sys, "stdin", stdin)
    rc, out = _run(["sync-issue", "--issue", str(ISSUE), "--rewrite-from", "-"], capsys)
    assert rc == 1 and out["status"] == "missing-body-file" and out["path"] == "-"
    assert runner.edits == []


def test_sync_issue_rewrite_keeps_crlf_and_ignores_a_trailing_newline(repos, runner, capsys, tmp_path) -> None:
    repos.write(DOC, _DOC_TEXT)
    runner.body = f"{_LINK}\r\n\r\n## Children\r\n\r\n- [ ] #901 — first child\r\n- (not filed) second child"
    body_file = tmp_path / "body.md"
    body_file.write_bytes((runner.body + "\r\n").encode("utf-8"))
    rc, out = _rewrite(capsys, body_file)
    assert rc == 0 and out["status"] == "unchanged", out
    assert runner.edits == []


def test_sync_issue_rewrite_lf_text_over_a_crlf_body_is_unchanged(repos, runner, capsys, tmp_path) -> None:
    repos.write(DOC, _DOC_TEXT)
    runner.body = f"{_LINK}\r\n\r\n## Children\r\n\r\n- [ ] #901 — first child\r\n- (not filed) second child\r\n"
    body_file = tmp_path / "body.md"
    body_file.write_text(runner.body.replace("\r\n", "\n"), encoding="utf-8")
    rc, out = _rewrite(capsys, body_file)
    assert rc == 0 and out["status"] == "unchanged", out
    assert runner.edits == []


def test_sync_issue_rewrite_keeps_a_readded_child_ticked(repos, runner, capsys, tmp_path) -> None:
    # The rewrite text drops the ticked #901 line: sync_body re-adds it, still ticked.
    repos.write(DOC, _DOC_TEXT)
    runner.body = f"{_LINK}\n\n## Children\n\n- [x] #901 — first child\n"
    body_file = tmp_path / "body.md"
    body_file.write_text(f"{_LINK}\n\nNew.\n\n## Children\n\n- (not filed) second child\n", encoding="utf-8")
    rc, out = _rewrite(capsys, body_file)
    assert rc == 0 and out["added"] == [901], out
    (edit,) = runner.edits
    assert "- [x] #901 — first child\n" in edit["body"]
    assert "- [ ] #901" not in edit["body"]


def test_sync_issue_rewrite_keeps_ticked_children_ticked(repos, runner, capsys, tmp_path) -> None:
    repos.write(DOC, _DOC_TEXT)
    runner.body = f"{_LINK}\n\n## Children\n\n- [x] #901 — first child\n"
    body_file = tmp_path / "body.md"
    body_file.write_text(f"{_LINK}\n\nNew.\n\n## Children\n\n- [ ] #901 — first child (renamed)\n", encoding="utf-8")
    rc, out = _rewrite(capsys, body_file)
    assert rc == 0, out
    (edit,) = runner.edits
    assert "- [x] #901 — first child (renamed)\n" in edit["body"]


def test_carry_ticks_only_touches_ticked_children() -> None:
    current = "- [X] #1 — a\n- [ ] #2 — b\n* [x] #3 — c\n"
    new = "- [ ] #1 — a\r\n- [ ] #2 — b\r\n  * [ ] #3 — c\r\n- [ ] #4 — d\r\nsee #1\r\n"
    assert epic_doc.carry_ticks(current, new) == (
        "- [x] #1 — a\r\n- [ ] #2 — b\r\n  * [x] #3 — c\r\n- [ ] #4 — d\r\nsee #1\r\n"
    )


def test_sync_issue_rewrite_blank_file_writes_nothing(repos, runner, capsys, tmp_path) -> None:
    repos.write(DOC, _DOC_TEXT)
    body_file = tmp_path / "body.md"
    body_file.write_text(" \n\t\n", encoding="utf-8")
    rc, out = _rewrite(capsys, body_file)
    assert rc == 1 and out["status"] == "empty-body-file" and out["path"] == str(body_file)
    assert runner.edits == []


def test_sync_issue_rewrite_checks_preconditions_first(repos, runner, capsys, tmp_path) -> None:
    # No epic doc in the worktree: missing-doc wins over the (also missing) body file.
    rc, out = _rewrite(capsys, tmp_path / "nope.md")
    assert rc == 1 and out["status"] == "missing-doc"
    runner.labels = ["kind:bug", "target:factory", "scale:task"]
    rc, out = _rewrite(capsys, tmp_path / "nope.md")
    assert rc == 1 and out["status"] == "not-an-epic"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_non_positive_issue_is_rejected_by_argparse() -> None:
    with pytest.raises(SystemExit) as exc:
        epic_doc.main(["status", "--issue", "0"])
    assert exc.value.code == 2


def test_script_runs_as_a_subprocess_with_json_out(repos) -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/qs/epic_doc.py"), "status", "--issue", str(ISSUE)],
        cwd=repos.work, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["mode"] == "DECOMPOSE"
