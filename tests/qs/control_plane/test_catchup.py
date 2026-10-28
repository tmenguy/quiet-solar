"""``cp.py catchup`` (#440): the node's catch-up on real temporary git repositories.

A session-scoped template (a bare ``origin`` with ``main``, the task's clone on ``QS_1``, a second
clone that moves ``main`` or pushes to the task branch) is copied per test. The ops run in process
(``catchup.step(conn, Runner(), clock, …)``) and through ``run_cli`` with a real ``Runner``; failures
come from a runner wrapper that delegates to real git and injects a result.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from control_plane import catchup, db, errors, faults, node_policy, runner

from .conftest import insert_node, insert_task, open_run, run_cli, sql

BRANCH = "QS_1"
LOCATION_VARS = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR")


# --------------------------------------------------------------------------- the git template


def _git_env(home: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update({"HOME": str(home), "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0"})
    return env


def _run(cwd: Path, *args: str, env: dict[str, str] | None = None) -> str:
    res = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=False)
    assert res.returncode == 0, (args, res.stdout, res.stderr)
    return res.stdout.strip()


def _identity(clone: Path, name: str, env: dict[str, str]) -> None:
    for key, value in (("user.name", name), ("user.email", f"{name}@example.com"), ("commit.gpgsign", "false")):
        _run(clone, "config", key, value, env=env)


@pytest.fixture(scope="session")
def template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("catchup-template")
    env = _git_env(root)
    _run(root, "init", "--bare", "-b", "main", "origin.git", env=env)
    origin = root / "origin.git"
    _run(root, "clone", str(origin), "other", env=env)
    other = root / "other"
    _identity(other, "other", env)
    _run(other, "checkout", "-b", "main", env=env)
    (other / "shared.txt").write_text("base\n")
    (other / "a.txt").write_text("a\n")
    _run(other, "add", "shared.txt", "a.txt", env=env)
    _run(other, "commit", "-m", "base", env=env)
    _run(other, "push", "origin", "main", env=env)
    _run(other, "checkout", "-b", BRANCH, env=env)
    (other / "shared.txt").write_text("task\n")
    _run(other, "add", "shared.txt", env=env)
    _run(other, "commit", "-m", "task work", env=env)
    _run(other, "push", "origin", BRANCH, env=env)
    _run(other, "checkout", "main", env=env)
    _run(other, "branch", "-D", BRANCH, env=env)
    _run(root, "clone", "-b", BRANCH, str(origin), "wt", env=env)
    wt = root / "wt"
    _identity(wt, "task", env)
    with (wt / ".git" / "info" / "exclude").open("a") as handle:
        handle.write(".qs-node/\n")
    return root


class Repo:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.origin = root / "origin.git"
        self.wt = root / "wt"
        self.other = root / "other"
        self.edits = 0

    def git(self, *args: str, where: Path | None = None) -> str:
        return _run(where or self.wt, *args)

    def head(self) -> str:
        return self.git("rev-parse", "HEAD")

    def merges(self, since: str) -> list[str]:
        out = self.git("rev-list", "--merges", f"{since}..HEAD")
        return out.split() if out else []

    def move_main(self, *, conflict: bool = False) -> None:
        """The second clone advances ``main``: a new file, or a conflicting edit of ``shared.txt``."""
        self.edits += 1
        self.git("pull", "--ff-only", "origin", "main", where=self.other)
        name = "shared.txt" if conflict else f"m{self.edits}.txt"
        (self.other / name).write_text(f"main {self.edits}\n")
        self.git("add", name, where=self.other)
        self.git("commit", "-m", f"main {self.edits}", where=self.other)
        self.git("push", "origin", "main", where=self.other)

    def commit(self, name: str = "fix.txt", text: str = "fix\n") -> str:
        (self.wt / name).write_text(text)
        self.git("add", name)
        self.git("commit", "-m", f"commit {name}")
        return self.head()

    def resolve(self) -> str:
        """The implementer resolves the open merge and commits it."""
        (self.wt / "shared.txt").write_text("resolved\n")
        self.git("add", "shared.txt")
        self.git("commit", "--no-edit")
        return self.head()

    def push(self) -> None:
        self.git("push", "origin", f"HEAD:refs/heads/{BRANCH}")


@pytest.fixture
def repo(template: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Repo:
    root = tmp_path / "repo"
    shutil.copytree(template, root, symlinks=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    for name in LOCATION_VARS:
        monkeypatch.delenv(name, raising=False)
    for clone in ("wt", "other"):
        _run(root / clone, "remote", "set-url", "origin", str(root / "origin.git"))
    return Repo(root)


# --------------------------------------------------------------------------- the runner wrapper


Hook = Callable[[list[str]], runner.RunResult | None]


class Wrap(runner.Runner):
    """Delegates to real git; ``inject(pattern, result)`` answers a matching call instead (the last rule wins)."""

    def __init__(self) -> None:
        self.real = runner.Runner()
        self.rules: list[tuple[tuple[str, ...], runner.RunResult | Hook]] = []
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.after: list[tuple[tuple[str, ...], Callable[[], None]]] = []

    def inject(self, pattern: tuple[str, ...], result: runner.RunResult | Hook) -> None:
        self.rules.append((pattern, result))

    def then(self, pattern: tuple[str, ...], fn: Callable[[], None]) -> None:
        """Run ``fn`` right after a matching real call (a token rotation mid-op, a clock jump)."""
        self.after.append((pattern, fn))

    @staticmethod
    def _match(argv: list[str], pattern: tuple[str, ...]) -> bool:
        return tuple(argv[1 : 1 + len(pattern)]) == pattern

    def run(self, argv: Any, **kw: Any) -> runner.RunResult:
        argv = list(argv)
        self.calls.append((argv, kw))
        for pattern, result in reversed(self.rules):
            if self._match(argv, pattern):
                got = result(argv) if callable(result) else result
                if got is not None:
                    return got
        res = self.real.run(argv, **kw)
        for pattern, fn in self.after:
            if self._match(argv, pattern):
                fn()
        return res


# --------------------------------------------------------------------------- the node simulator


class Node:
    """The node's side: ``round.json``'s trigger and ``inv``, the ops called in process."""

    def __init__(self, world: World) -> None:
        self.w = world
        self.triggers = 0
        self.inv = 0
        self.head = world.repo.head()
        self.trigger: tuple[str, int, str] | None = None

    def start(self, from_: str) -> None:
        self.triggers += 1
        self.trigger = (from_, self.triggers, self.head)

    def call(self, **reports: Any) -> dict[str, Any]:
        assert self.trigger is not None
        from_, seq, head = self.trigger
        answer = catchup.step(
            self.w.conn,
            self.w.run,
            self.w.clock,
            token=self.w.token,
            task_id="T1",
            trigger_from=from_,
            seq=seq,
            head=head,
            **reports,
        )
        if answer["next"] == "wait" and answer["catchup_head"]:
            self.head = answer["head"]  # the node's wait write
        return answer

    def implemented(self, gate: str = "green", inv: int | None = None) -> dict[str, Any]:
        if inv is None:
            self.inv += 1
            inv = self.inv
        return self.call(implemented=gate, inv=inv)

    def drive(self, answer: dict[str, Any]) -> dict[str, Any]:
        """Follow the answers to the end of the catch-up (a green implementer, a done review, a push)."""
        for _ in range(12):
            verb = answer["next"]
            if verb == "implement":
                if answer["mode"] != "gate":
                    self.w.repo.resolve()
                answer = self.implemented()
            elif verb == "review":
                answer = self.call(reviewed=answer["head"])
            elif verb == "push":
                self.w.repo.push()
                answer = self.call()
            else:
                return answer
        raise AssertionError(answer)  # a loop would be a bug


class World:
    def __init__(self, repo: Repo, conn: Any, clock: Any, token: str, db_path: Path, run_id: str) -> None:
        self.repo = repo
        self.conn = conn
        self.clock = clock
        self.token = token
        self.db = db_path
        self.run_id = run_id
        self.run: runner.Runner = runner.Runner()
        self.node = Node(self)

    @property
    def node_dir(self) -> Path:
        return self.repo.wt / ".qs-node"

    def state(self) -> dict[str, Any]:
        return catchup.load_state(self.node_dir)

    def raw_state(self) -> str | None:
        path = self.node_dir / "catchup.json"
        return path.read_text() if path.exists() else None

    def op(self, name: str) -> dict[str, Any]:
        fn = getattr(catchup, name)
        result: dict[str, Any] = fn(self.conn, self.run, self.clock, token=self.token, task_id="T1")
        return result

    def pending(self, *entries: tuple[int, str]) -> None:
        self.node_dir.mkdir(exist_ok=True)
        data = [{"finding_id": fid, "level": level, "what": "w"} for fid, level in entries]
        (self.node_dir / "pending.json").write_text(json.dumps(data))

    def finding(self, state: str = "open", task: str = "T1") -> int:
        sql(
            self.db,
            "INSERT INTO findings (run_id, task_id, phase, round, source, severity, category, title, body,"
            " fingerprint, title_norm, replay_key, state, decided_seq, actor, created_at, updated_at)"
            " VALUES (?, ?, 'build', 0, 'orchestrator', 'must_fix', 'test', 't', 'b', 'fp', 't', ?, ?, 1,"
            " 'x', 'x', 'x')",
            [self.run_id, task, f"rk-{os.urandom(4).hex()}", state],
        )
        return int(sql(self.db, "SELECT max(id) FROM findings")[0][0])

    def close(self, fid: int, state: str = "rejected") -> None:
        sql(self.db, "UPDATE findings SET state = ? WHERE id = ?", [state, fid])

    def build_round(self, number: int, base: str | None, head: str) -> None:
        sql(
            self.db,
            "INSERT INTO rounds (task_id, phase, round, base_sha, head_sha, actor, started_at)"
            " VALUES ('T1', 'build', ?, ?, ?, 'node:T1', 'x')",
            [number, base, head],
        )

    def report(self, number: int, status: str = "converged") -> int:
        sql(
            self.db,
            "INSERT INTO reports (task_id, phase, round, status, summary, fields, at)"
            " VALUES ('T1', 'build', ?, ?, 's', '{}', 'x')",
            [number, status],
        )
        return int(sql(self.db, "SELECT max(id) FROM reports")[0][0])

    def rotate(self) -> None:
        """A new node generation: the old token is superseded."""
        sql(self.db, "UPDATE nodes SET state = 'superseded' WHERE id = 'N1'")
        self.token = insert_node(self.db, "N2", self.run_id, "T1", generation=2, session_id="S-n2", name="n2")


@pytest.fixture
def w(repo: Repo, conn: Any, migrated: Path, fake_clock: Any) -> Iterator[World]:
    run_id, _ = open_run()
    insert_task(migrated, "T1", run_id, worktree=str(repo.wt), branch=BRANCH, is_deliverable=1, issue_number=1)
    insert_task(migrated, "T2", run_id, worktree=str(repo.wt), branch="QS_2", is_deliverable=1)
    token = insert_node(migrated, "N1", run_id, "T1", session_id="S-n1", name="n1")
    yield World(repo, conn, fake_clock, token, migrated, run_id)


def _blocked_error(answer: dict[str, Any], detail: str) -> None:
    assert answer == {"next": "blocked", "cause": "error", "detail": detail}


def _code(exc: pytest.ExceptionInfo[errors.CpError]) -> str:
    return exc.value.code


# --------------------------------------------------------------------------- the module's tables


def test_the_module_constants() -> None:
    assert (catchup.CATCHUP_GIT_TIMEOUT_S, catchup.CATCHUP_OP_DEADLINE_S, catchup.CATCHUP_MIN_LEFT_S) == (120, 300, 60)
    assert node_policy.MAX_CATCHUPS == 3


ARMED_IN_TESTS = (
    "catchup.after_adopt",
    "catchup.after_intake",
    "catchup.after_reviewed",
    "catchup.after_unblock",
    "catchup.after_bound",
    "catchup.after_count",
    "catchup.after_fetch_precheck",
    "catchup.after_fetch_merge",
    "catchup.after_stale",
    "catchup.after_merge",
    "catchup.after_record",
    "catchup.after_conflict_record",
    "catchup.after_lookup_record",
    "catchup.after_push_write",
    "catchup.after_fixes_write",
    "catchup.after_done_write",
)


def test_the_fault_points_are_the_source_literals_and_the_tested_ones() -> None:
    tree = ast.parse(Path(catchup.__file__).read_text(encoding="utf-8"))
    literals = [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "hit"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "faults"
        and isinstance(node.args[0], ast.Constant)
    ]
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "hit"
    ]
    assert len(calls) == len(literals)  # every site is a string literal
    assert len(literals) == len(set(literals))  # each name at exactly one site
    assert set(literals) == set(catchup.FAULTS) == set(ARMED_IN_TESTS)
    source = Path(__file__).read_text(encoding="utf-8")
    for name in catchup.FAULTS:
        assert source.count(f'_crash("{name}"') >= 1, name  # armed through faults.arm


def _crash(name: str, fn: Callable[[], Any]) -> None:
    with faults.arm(name), pytest.raises(faults.FaultInjected):
        fn()


# --------------------------------------------------------------------------- the state file


def test_a_missing_state_file_is_the_empty_state(tmp_path: Path) -> None:
    assert catchup.load_state(tmp_path) == catchup.empty_state()
    assert catchup.load_pending(tmp_path) == []


def _valid_catchup() -> dict[str, Any]:
    sha = "a" * 40
    return {
        "from": "done",
        "trigger": {"from": "done", "seq": 2, "head": sha},
        "head": sha,
        "counted": True,
        "merge": sha,
        "clean": True,
        "green": None,
        "reviewed": None,
        "misses": ["gate red"],
        "intake": [1],
        "block": {"cause": "misses", "detail": "d", "at": "t"},
    }


def test_a_full_state_round_trips(tmp_path: Path) -> None:
    sha = "b" * 64
    state = catchup.empty_state()
    state.update(
        catchups=2,
        max_seq=3,
        catchup=_valid_catchup(),
        base_merge={"merge": sha, "head": sha},
        last={
            "trigger": {"from": "conflict", "seq": 1, "head": sha},
            "kind": "wait",
            "answer": catchup.wait_answer(sha, True),
        },
        done_report=0,
    )
    catchup.save_state(tmp_path, state)
    assert catchup.load_state(tmp_path) == state
    state["last"] = {"trigger": {"from": "conflict", "seq": 1, "head": sha}, "kind": "done", "answer": None}
    assert catchup.valid_state(state)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.update(version=2),
        lambda s: s.update(version=True),
        lambda s: s.update(catchups=-1),
        lambda s: s.update(catchups=True),
        lambda s: s.pop("max_seq"),
        lambda s: s.update(extra=1),
        lambda s: s.update(done_report="1"),
        lambda s: s.update(base_merge={"merge": "abc", "head": "a" * 40}),
        lambda s: s.update(last={"trigger": None, "kind": "wait", "answer": None}),
        lambda s: s.update(
            last={"trigger": {"from": "done", "seq": 0, "head": "a" * 40}, "kind": "done", "answer": None}
        ),
        lambda s: s.update(
            last={"trigger": {"from": "done", "seq": 1, "head": "a" * 40}, "kind": "wait", "answer": None}
        ),
        lambda s: s.update(
            last={"trigger": {"from": "done", "seq": 1, "head": "a" * 40}, "kind": "done", "answer": {}}
        ),
        lambda s: s.update(catchup={**_valid_catchup(), "misses": ["x"]}),
        lambda s: s.update(catchup={**_valid_catchup(), "misses": ["gate red"] * 3}),
        lambda s: s.update(catchup={**_valid_catchup(), "intake": [0]}),
        lambda s: s.update(catchup={**_valid_catchup(), "block": {"cause": "error", "detail": "d", "at": "t"}}),
        lambda s: s.update(catchup={**_valid_catchup(), "clean": "yes"}),
        lambda s: s.update(catchup={**_valid_catchup(), "from": "x"}),
        lambda s: s.update(catchup=[]),
    ],
)
def test_a_corrupt_state_file_is_invalid_state(tmp_path: Path, mutate: Callable[[dict[str, Any]], Any]) -> None:
    state = catchup.empty_state()
    mutate(state)
    (tmp_path / "catchup.json").write_text(json.dumps(state))
    with pytest.raises(errors.CpError) as exc:
        catchup.load_state(tmp_path)
    assert _code(exc) == "INVALID_STATE" and "corrupt" in exc.value.detail


@pytest.mark.parametrize("text", ["{not json", "\udcff"])
def test_an_unparseable_state_file_is_invalid_state(tmp_path: Path, text: str) -> None:
    (tmp_path / "catchup.json").write_bytes(text.encode("utf-8", "surrogateescape"))
    with pytest.raises(errors.CpError) as exc:
        catchup.load_state(tmp_path)
    assert _code(exc) == "INVALID_STATE"


def test_an_unreadable_state_file_is_invalid_state(tmp_path: Path) -> None:
    (tmp_path / "catchup.json").mkdir()
    with pytest.raises(errors.CpError) as exc:
        catchup.load_state(tmp_path)
    assert _code(exc) == "INVALID_STATE"


@pytest.mark.parametrize(
    "data",
    ['{"finding_id": 1}', '[{"finding_id": true, "level": "code"}]', '[{"finding_id": 1, "level": "x"}]', "[1]", "{x"],
)
def test_a_malformed_pending_file_is_invalid_state(tmp_path: Path, data: str) -> None:
    (tmp_path / "pending.json").write_text(data)
    with pytest.raises(errors.CpError) as exc:
        catchup.load_pending(tmp_path)
    assert _code(exc) == "INVALID_STATE"


def test_pending_entries_keep_their_other_keys(tmp_path: Path) -> None:
    entries = [{"finding_id": 3, "level": "plan", "what": "w", "extra": [1]}]
    (tmp_path / "pending.json").write_text(json.dumps(entries))
    assert catchup.load_pending(tmp_path) == entries


def test_a_crash_before_the_replace_leaves_the_old_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = tmp_path / "node"
    tmp_path.mkdir()
    old = catchup.empty_state()
    catchup.save_state(tmp_path, old)
    before = (tmp_path / "catchup.json").read_text()

    def boom(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(catchup.os, "replace", boom)
    with pytest.raises(errors.CpError) as exc:
        catchup.save_state(tmp_path, {**old, "catchups": 1})
    assert _code(exc) == "TOOL_FAILED"
    assert (tmp_path / "catchup.json").read_text() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["catchup.json"]  # the temp file is gone


def test_a_failed_temp_file_is_tool_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**kw: Any) -> Any:
        raise OSError("read-only")

    monkeypatch.setattr(catchup.tempfile, "mkstemp", boom)
    with pytest.raises(errors.CpError) as exc:
        catchup.save_state(tmp_path, catchup.empty_state())
    assert _code(exc) == "TOOL_FAILED" and not (tmp_path / "catchup.json").exists()


# --------------------------------------------------------------------------- arguments, token, task, lock (D1)


@pytest.mark.parametrize(
    ("kw", "word"),
    [
        ({"trigger_from": "x"}, "--from"),
        ({"implemented": "green"}, "--inv"),
        ({"inv": 1}, "--inv"),
        ({"implemented": "blue", "inv": 1}, "--implemented"),
        ({"head": "abc1234"}, "--head"),
        ({"reviewed": "A" * 40}, "--reviewed"),
        ({"seq": 0}, "--seq"),
        ({"implemented": "red", "inv": 0}, "--inv"),
    ],
)
def test_step_arguments_are_usage(w: World, kw: dict[str, Any], word: str) -> None:
    args: dict[str, Any] = {"trigger_from": "conflict", "seq": 1, "head": w.repo.head(), **kw}
    with pytest.raises(errors.CpError) as exc:
        catchup.step(w.conn, w.run, w.clock, token=w.token, task_id="T1", **args)
    assert _code(exc) == "USAGE" and word in exc.value.detail


def _cli(w: World, *argv: str) -> tuple[int, Any]:
    return run_cli("catchup", *argv, "--task", "T1", "--token", w.token)


@pytest.fixture
def real_cli_runner(deps: Any) -> None:
    deps.runner = runner.Runner()


def test_the_cli_step_round_trip(w: World, real_cli_runner: None) -> None:
    head = w.repo.head()
    code, out = _cli(w, "step", "--from", "conflict", "--seq", "1", "--head", head)
    assert (code, out) == (0, {"ok": True, "next": "wait", "head": head, "catchup_head": False})
    assert _cli(w, "base") == (0, {"ok": True, "base": None})
    assert _cli(w, "fixes") == (0, {"ok": True, "next": "fixes", "open": [], "drop": []})
    assert _cli(w, "unblock") == (0, {"ok": True, "unblocked": None})
    code, out = _cli(w, "done")
    assert code == 0 and out["next"] == "done"


@pytest.mark.parametrize(
    "extra",
    [
        ["--head", "abc1234"],
        ["--head", "a" * 40, "--reviewed", "zz"],
        ["--head", "a" * 40, "--implemented", "green"],
        ["--head", "a" * 40, "--inv", "2"],
        ["--head", "a" * 40, "--implemented", "maybe", "--inv", "2"],
    ],
)
def test_the_cli_refuses_bad_step_arguments(w: World, extra: list[str], real_cli_runner: None) -> None:
    code, out = _cli(w, "step", "--from", "conflict", "--seq", "1", *extra)
    assert code == 2 and out["error"] == "USAGE"
    assert w.raw_state() is None


@pytest.mark.parametrize("op", ["base", "fixes", "done", "unblock"])
def test_every_op_honours_stopped(w: World, op: str, real_cli_runner: None) -> None:
    sql(w.db, "UPDATE nodes SET state = 'stopped' WHERE id = 'N1'")
    code, out = _cli(w, op)
    assert code == 4 and out["error"] == "STOPPED"


def test_step_honours_stopped_before_any_git(w: World) -> None:
    sql(w.db, "UPDATE nodes SET state = 'stopped' WHERE id = 'N1'")
    w.run = wrap = Wrap()
    w.node.start("conflict")
    with pytest.raises(errors.CpError) as exc:
        w.node.call()
    assert _code(exc) == "STOPPED" and wrap.calls == []


@pytest.mark.parametrize(
    ("change", "word"),
    [
        ("UPDATE tasks SET state = 'merged' WHERE id = 'T1'", "merged"),
        ("UPDATE tasks SET deliverable_id = 'T2' WHERE id = 'T1'", ""),
        ("UPDATE tasks SET worktree = NULL WHERE id = 'T1'", "worktree"),
        ("UPDATE tasks SET branch = NULL WHERE id = 'T1'", "branch"),
        ("UPDATE tasks SET worktree = '/nonexistent/qs440' WHERE id = 'T1'", "not a directory"),
    ],
)
def test_the_task_checks_come_before_any_git(w: World, change: str, word: str) -> None:
    sql(w.db, change)
    w.run = wrap = Wrap()
    w.node.start("conflict")
    with pytest.raises(errors.CpError) as exc:
        w.node.call()
    assert _code(exc) == "INVALID_STATE" and word in exc.value.detail
    assert wrap.calls == []


def test_a_held_lock_is_busy(w: World) -> None:
    w.node_dir.mkdir()
    w.node.start("conflict")
    with db.file_lock(w.node_dir / "catchup.lock", exclusive=True, timeout=0) as got:
        assert got
        with pytest.raises(errors.CpError) as exc:
            w.node.call()
    assert _code(exc) == "BUSY"
    assert w.node.call()["next"] == "wait"  # released: the same call goes through


def test_a_mid_call_supersede_writes_nothing(w: World) -> None:
    w.repo.move_main()
    w.run = wrap = Wrap()
    wrap.then(("fetch",), w.rotate)
    w.node.start("done")
    old = w.token  # the call carries the generation the fetch supersedes
    with pytest.raises(errors.CpError) as exc:
        catchup.step(w.conn, wrap, w.clock, token=old, task_id="T1", trigger_from="done", seq=1, head=w.node.head)
    assert _code(exc) == "STALE_TOKEN"
    assert w.raw_state() is None


# --------------------------------------------------------------------------- the op deadline


@pytest.fixture
def clock_s(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [1000.0]
    monkeypatch.setattr(catchup, "_monotonic", lambda: now[0])
    return now


def test_past_the_deadline_nothing_runs(w: World, clock_s: list[float]) -> None:
    w.run = wrap = Wrap()
    wrap.inject(("symbolic-ref",), lambda argv: None)
    w.node.start("conflict")
    original = catchup._Op.__init__

    def late_init(self: Any, *a: Any, **k: Any) -> None:
        original(self, *a, **k)
        clock_s[0] += catchup.CATCHUP_OP_DEADLINE_S + 1

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(catchup._Op, "__init__", late_init)
        with pytest.raises(errors.CpError) as exc:
            w.node.call()
    assert _code(exc) == "TOOL_FAILED" and exc.value.detail == "op deadline"
    assert wrap.calls == [] and w.raw_state() is None


def test_a_git_timeout_is_cut_to_the_time_left(w: World, clock_s: list[float]) -> None:
    w.run = wrap = Wrap()

    def jump(argv: list[str]) -> None:
        clock_s[0] = 1000.0 + catchup.CATCHUP_OP_DEADLINE_S - 70  # 70 s left

    wrap.inject(("symbolic-ref",), jump)
    w.node.start("conflict")
    w.node.call()
    timeouts = [kw["timeout"] for _, kw in wrap.calls]
    assert timeouts[0] == catchup.CATCHUP_GIT_TIMEOUT_S
    assert timeouts[1:] and all(t == 70 for t in timeouts[1:])


def test_a_merge_is_not_started_under_the_minimum_left(w: World, clock_s: list[float]) -> None:
    w.repo.move_main()
    w.run = wrap = Wrap()

    def late() -> None:
        clock_s[0] = 1000.0 + catchup.CATCHUP_OP_DEADLINE_S - catchup.CATCHUP_MIN_LEFT_S + 10  # 50 s left

    wrap.then(("fetch",), late)
    w.node.start("conflict")
    with pytest.raises(errors.CpError) as exc:
        w.node.call()
    assert _code(exc) == "TOOL_FAILED" and exc.value.detail == "op deadline"
    assert not any(argv[1] == "merge" for argv, _ in wrap.calls)
    assert w.repo.merges(w.node.head) == [] and w.state()["catchups"] == 1
    clock_s[0] = 5000.0
    w.run = runner.Runner()
    assert w.node.call() == {"next": "implement", "mode": "gate", "fixes": []}  # the retry merges, never counts again
    assert w.state()["catchups"] == 1 and len(w.repo.merges(w.node.head)) == 1


# --------------------------------------------------------------------------- git failures (D4's exit table)


@pytest.mark.parametrize(
    ("pattern", "code", "error"),
    [
        (("symbolic-ref",), 124, "TOOL_FAILED"),
        (("rev-parse", "HEAD"), 128, "TOOL_FAILED"),
        (("rev-parse", "-q", "--verify", "MERGE_HEAD"), 2, "TOOL_FAILED"),
        (("merge-base",), 128, "TOOL_FAILED"),
        (("fetch",), 1, "TOOL_FAILED"),
    ],
)
def test_a_failed_git_call_is_tool_failed(w: World, pattern: tuple[str, ...], code: int, error: str) -> None:
    w.repo.move_main()
    w.run = wrap = Wrap()
    wrap.inject(pattern, runner.RunResult(code, "", "boom"))
    w.node.start("done")
    with pytest.raises(errors.CpError) as exc:
        w.node.call()
    assert _code(exc) == error and exc.value.extra["tail"] == ["boom"]


def test_a_detached_head_is_invalid_state(w: World) -> None:
    w.repo.git("checkout", "--detach")
    w.node.start("conflict")
    with pytest.raises(errors.CpError) as exc:
        w.node.call()
    assert _code(exc) == "INVALID_STATE" and exc.value.detail == "detached or not on the branch"


def test_another_branch_is_invalid_state(w: World) -> None:
    w.repo.git("checkout", "-b", "elsewhere")
    w.node.start("conflict")
    with pytest.raises(errors.CpError) as exc:
        w.node.call()
    assert exc.value.detail == "detached or not on the branch"


def test_an_unreachable_origin_is_tool_failed(w: World) -> None:
    w.repo.git("remote", "set-url", "origin", str(w.repo.root / "missing.git"))
    w.node.start("done")
    with pytest.raises(errors.CpError) as exc:
        w.node.call()
    assert _code(exc) == "TOOL_FAILED" and "fetch" in exc.value.detail


def test_a_ref_lock_on_fetch_is_busy(w: World) -> None:
    w.run = wrap = Wrap()
    wrap.inject(("fetch",), runner.RunResult(1, "", "error: cannot lock ref 'refs/remotes/origin/main'"))
    w.node.start("done")
    with pytest.raises(errors.CpError) as exc:
        w.node.call()
    assert _code(exc) == "BUSY"


@pytest.mark.parametrize(("pattern", "code"), [(("ls-remote",), 128), (("cat-file",), 124)])
def test_a_failed_remote_check_is_tool_failed(w: World, pattern: tuple[str, ...], code: int) -> None:
    w.run = wrap = Wrap()
    wrap.inject(("ls-remote",), runner.RunResult(0, "f" * 40 + "\trefs/heads/QS_1\n", ""))
    wrap.inject(pattern, runner.RunResult(code, "", "boom"))
    w.node.start("done")
    with pytest.raises(errors.CpError) as exc:
        w.node.call()
    assert _code(exc) == "TOOL_FAILED"


def test_a_failed_lookup_is_tool_failed(w: World) -> None:
    w.repo.move_main(conflict=True)
    w.node.start("conflict")
    assert w.node.call()["mode"] == "resolve"
    w.repo.resolve()  # 3c: HEAD moved — the lookup runs
    w.run = wrap = Wrap()
    wrap.inject(("log",), runner.RunResult(128, "", "boom"))
    with pytest.raises(errors.CpError) as exc:
        w.node.call()
    assert _code(exc) == "TOOL_FAILED" and exc.value.detail.startswith("git log")


def test_a_merge_refused_by_a_dirty_file_writes_nothing_more(w: World) -> None:
    w.repo.move_main()  # adds m1.txt on main
    (w.repo.wt / "m1.txt").write_text("untracked, in the merge's way\n")
    w.node.start("conflict")
    with pytest.raises(errors.CpError) as exc:
        w.node.call()
    assert _code(exc) == "TOOL_FAILED" and "merge" in exc.value.detail
    cu = w.state()["catchup"]
    assert cu["merge"] is None and cu["clean"] is None and w.state()["catchups"] == 1
    (w.repo.wt / "m1.txt").unlink()
    assert w.node.call()["mode"] == "gate"  # the same call: the merge runs once
    assert w.state()["catchups"] == 1


# --------------------------------------------------------------------------- the trigger rule (D2)


def test_a_trigger_older_than_the_last_one_is_invalid_state(w: World) -> None:
    w.node.start("conflict")
    w.node.call()  # stale: last = seq 1
    w.node.start("conflict")
    w.node.call()  # stale: last = seq 2
    with pytest.raises(errors.CpError) as exc:
        catchup.step(
            w.conn, w.run, w.clock, token=w.token, task_id="T1", trigger_from="conflict", seq=1, head=w.node.head
        )
    assert exc.value.detail == "a trigger older than the last one"


def test_a_trigger_older_than_the_catch_up_in_progress_is_invalid_state(w: World) -> None:
    w.repo.move_main(conflict=True)
    w.node.start("conflict")
    w.node.start("conflict")  # seq 2
    assert w.node.call()["mode"] == "resolve"
    for seq, from_ in ((1, "conflict"), (2, "done")):
        with pytest.raises(errors.CpError) as exc:
            catchup.step(
                w.conn, w.run, w.clock, token=w.token, task_id="T1", trigger_from=from_, seq=seq, head=w.node.head
            )
        assert exc.value.detail == "a trigger older than the catch-up in progress"


def test_a_report_with_no_catch_up_in_progress_is_ignored(w: World) -> None:
    w.node.start("conflict")
    w.node.inv = 4
    assert w.node.implemented("red")["next"] == "wait"
    assert w.state()["catchup"] is None and w.state()["catchups"] == 1


def test_reap_after_trigger_write(w: World) -> None:
    """The node wrote the trigger and was reaped before the call: the resume issues the same call."""
    w.repo.move_main()
    w.node.start("done")
    first = w.node.call()
    assert first == {"next": "implement", "mode": "gate", "fixes": []}
    assert w.node.call() == first  # a resume re-runs the trigger: no second count, no second merge
    assert w.state()["catchups"] == 1 and len(w.repo.merges(w.node.head)) == 1


# --------------------------------------------------------------------------- the fault points (AC3)


def _merge_count(w: World) -> int:
    return len(w.repo.merges(w.node.head))


def test_fault_after_adopt(w: World) -> None:
    w.repo.move_main(conflict=True)
    w.node.start("conflict")
    assert w.node.call()["mode"] == "resolve"
    w.node.start("done")  # a new trigger mid-catch-up (round.json lost the old one)
    _crash("catchup.after_adopt", w.node.call)
    assert w.node.call() == {"next": "implement", "mode": "resolve", "fixes": []}
    st = w.state()
    assert st["catchups"] == 1 and st["catchup"]["from"] == "conflict" and st["catchup"]["trigger"]["seq"] == 2


def test_fault_after_intake(w: World) -> None:
    w.repo.move_main(conflict=True)
    w.node.start("conflict")
    w.node.call()
    w.repo.resolve()
    w.node.inv = 1
    _crash("catchup.after_intake", lambda: w.node.call(implemented="green", inv=1))
    answer = w.node.call(implemented="green", inv=1)  # the retry repeats the report
    assert answer["next"] == "review"
    cu = w.state()["catchup"]
    assert cu["intake"] == [1] and cu["misses"] == [] and _merge_count(w) == 1


def test_a_retried_report_after_tool_failed_counts_once(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    w.node.call()
    w.run = wrap = Wrap()
    wrap.inject(("rev-parse", "-q", "--verify", "MERGE_HEAD"), runner.RunResult(2, "", "x"))  # after the intake
    w.node.inv = 1
    with pytest.raises(errors.CpError):
        w.node.call(implemented="red", inv=1)
    assert w.state()["catchup"]["misses"] == ["gate red"]
    w.run = runner.Runner()
    assert w.node.call(implemented="red", inv=1)["mode"] == "gate"
    assert w.state()["catchup"]["misses"] == ["gate red"]


def test_fault_after_reviewed(w: World) -> None:
    w.repo.move_main(conflict=True)
    w.node.start("conflict")
    w.node.call()
    w.repo.resolve()
    review = w.node.implemented()
    assert review["next"] == "review"
    _crash("catchup.after_reviewed", lambda: w.node.call(reviewed=review["head"]))
    assert w.node.call() == {"next": "push", "head": review["head"]}  # a resume: the review is recorded


def test_fault_after_unblock(w: World) -> None:
    st = catchup.empty_state()
    st["catchups"] = 3
    w.node_dir.mkdir()
    catchup.save_state(w.node_dir, st)
    w.repo.move_main()
    w.node.start("conflict")
    assert w.node.call()["cause"] == "catchups"
    _crash("catchup.after_unblock", lambda: w.op("unblock"))
    assert w.op("unblock") == {"unblocked": None}
    assert w.state()["catchups"] == 0
    assert w.node.call()["mode"] == "gate"
    assert w.state()["catchups"] == 1


def test_fault_after_bound(w: World) -> None:
    """AC4.4's reap: after the bound write the call still answers blocked, with no merge."""
    st = catchup.empty_state()
    st["catchups"] = node_policy.MAX_CATCHUPS
    w.node_dir.mkdir()
    catchup.save_state(w.node_dir, st)
    w.repo.move_main()
    w.node.start("conflict")
    _crash("catchup.after_bound", w.node.call)
    answer = w.node.call()
    assert answer["next"] == "blocked" and answer["cause"] == "catchups"
    assert w.state()["catchups"] == 3 and _merge_count(w) == 0 and w.state()["catchup"]["counted"] is False


def test_fault_after_count(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    _crash("catchup.after_count", w.node.call)
    assert w.node.call()["mode"] == "gate"
    assert w.state()["catchups"] == 1 and _merge_count(w) == 1


def test_fault_after_fetch_precheck(w: World) -> None:
    w.repo.move_main()
    w.node.start("done")
    _crash("catchup.after_fetch_precheck", w.node.call)
    assert w.state()["catchups"] == 0
    assert w.node.call()["mode"] == "gate"
    assert w.state()["catchups"] == 1 and _merge_count(w) == 1


def test_fault_after_fetch_merge(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    _crash("catchup.after_fetch_merge", w.node.call)
    assert w.node.call()["mode"] == "gate"
    assert w.state()["catchups"] == 1 and _merge_count(w) == 1


def test_fault_after_stale(w: World) -> None:
    w.node.start("conflict")
    _crash("catchup.after_stale", w.node.call)
    assert w.node.call() == catchup.wait_answer(w.node.head, False)
    assert w.state()["catchups"] == 1 and w.state()["catchup"] is None


def test_fault_after_merge_clean_resumes_through_the_lookup(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    _crash("catchup.after_merge", w.node.call)
    assert w.node.call()["mode"] == "gate"
    cu = w.state()["catchup"]
    assert cu["clean"] is False and cu["merge"] == w.repo.head()
    assert w.node.implemented()["next"] == "review"  # clean unknown, so reviewed: by design
    assert _merge_count(w) == 1


def test_fault_after_merge_with_conflicts(w: World) -> None:
    w.repo.move_main(conflict=True)
    w.node.start("conflict")
    _crash("catchup.after_merge", w.node.call)
    assert w.node.call()["mode"] == "resolve"
    assert w.state()["catchups"] == 1


def test_fault_after_record(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    _crash("catchup.after_record", w.node.call)
    assert w.node.call()["mode"] == "gate"
    assert w.state()["catchup"]["clean"] is True and _merge_count(w) == 1


def test_fault_after_conflict_record(w: World) -> None:
    w.repo.move_main(conflict=True)
    w.node.start("conflict")
    _crash("catchup.after_conflict_record", w.node.call)
    assert w.node.call()["mode"] == "resolve"
    assert w.state()["catchups"] == 1


def test_fault_after_lookup_record(w: World) -> None:
    w.repo.move_main(conflict=True)
    w.node.start("conflict")
    w.node.call()
    w.repo.resolve()
    _crash("catchup.after_lookup_record", w.node.call)  # a resume without the report
    assert w.node.call()["mode"] == "gate"
    assert w.state()["catchup"]["merge"] == w.repo.head() and _merge_count(w) == 1


def test_fault_after_push_write(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    answer = w.node.call()
    answer = w.node.implemented()
    assert answer["next"] == "push"
    w.repo.push()
    _crash("catchup.after_push_write", w.node.call)
    head = w.repo.head()
    assert w.node.call() == catchup.wait_answer(head, True)
    st = w.state()
    assert st["base_merge"] == {"merge": head, "head": head} and st["catchups"] == 1


def test_fault_after_fixes_write(w: World) -> None:
    fid = w.finding()
    w.pending((fid, "code"))
    w.node.start("done")
    _crash("catchup.after_fixes_write", w.node.call)
    answer = w.node.call()
    assert answer == {"next": "fixes", "open": [{"finding_id": fid, "level": "code", "what": "w"}], "drop": []}


def test_fault_after_done_write(w: World) -> None:
    w.node.start("done")
    _crash("catchup.after_done_write", w.node.call)
    assert w.node.call() == catchup.done_answer(["digest", "report", "state"], [], [])
    assert w.state()["catchups"] == 0 and w.state()["done_report"] == 0


# --------------------------------------------------------------------------- node-side reap points (D4)


def test_reap_after_push_before_call(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    w.node.call()
    assert w.node.implemented()["next"] == "push"
    w.repo.push()  # `tool push`, then a reap before the next call
    assert w.node.call() == catchup.wait_answer(w.repo.head(), True)


def test_reap_after_push_before_wait_write(w: World) -> None:
    """#06's crash: the push write is Python's, so the replay carries the head the push made."""
    w.repo.move_main()
    w.node.start("conflict")
    answer = w.node.drive(w.node.call())
    assert answer == catchup.wait_answer(w.repo.head(), True)
    w.node.head = w.node.trigger[2]  # the node's wait write was lost: round.json still holds the old head
    assert w.node.call() == answer


def test_reap_after_done_answer(w: World) -> None:
    w.node.start("done")
    assert w.node.call()["next"] == "done"
    w.repo.move_main()  # main moved meanwhile: the replay recomputes step 5 only
    assert w.node.call()["next"] == "done"
    assert w.state()["catchups"] == 0 and _merge_count(w) == 0


def test_every_answer_is_replayed_by_a_re_issued_call(w: World) -> None:
    w.repo.move_main(conflict=True)
    w.node.start("conflict")
    answer = w.node.call()
    for _ in range(6):
        assert w.node.call() == answer  # a resume re-runs the trigger: the same answer
        if answer["next"] == "implement":
            if answer["mode"] != "gate":
                w.repo.resolve()
            answer = w.node.implemented()
        elif answer["next"] == "review":
            answer = w.node.call(reviewed=answer["head"])
        elif answer["next"] == "push":
            w.repo.push()
            answer = w.node.call()
        else:
            break
    assert answer["next"] == "wait" and w.node.call() == answer


# --------------------------------------------------------------------------- the answer schema (AC1)

SCHEMA = {
    "implement": {"next", "mode", "fixes"},
    "review": {"next", "head", "diffs"},
    "push": {"next", "head"},
    "wait": {"next", "head", "catchup_head"},
    "fixes": {"next", "open", "drop"},
    "done": {"next", "left", "objections", "drop"},
    "blocked": {"next", "cause", "detail"},
}


def _check_schema(answer: dict[str, Any], seen: set[str]) -> None:
    verb = answer["next"]
    assert set(answer) == SCHEMA[verb], answer
    if verb == "implement":
        assert answer["mode"] in ("gate", "resolve", "finish")
    if verb == "blocked":
        assert answer["cause"] in ("catchups", "misses", "error")
    seen.add(verb)


def test_every_answer_matches_its_schema(w: World) -> None:
    seen: set[str] = set()
    w.repo.move_main(conflict=True)
    w.node.start("conflict")
    answer = w.node.call()
    while True:
        _check_schema(answer, seen)
        if answer["next"] == "implement":
            w.repo.resolve()
            answer = w.node.implemented()
        elif answer["next"] == "review":
            answer = w.node.call(reviewed=answer["head"])
        elif answer["next"] == "push":
            w.repo.push()
            answer = w.node.call()
        else:
            break
    fid = w.finding()
    w.pending((fid, "code"))
    w.node.start("done")
    _check_schema(w.node.call(), seen)
    w.close(fid)
    _check_schema(w.op("done"), seen)
    _check_schema(w.op("fixes"), seen)
    w.repo.git("checkout", "--detach")
    w.repo.git("checkout", BRANCH)
    w.repo.commit()
    w.node.start("done")
    _check_schema(w.node.call(), seen)  # HEAD is not the pushed head
    assert seen == set(SCHEMA)


# --------------------------------------------------------------------------- AC4: #435's cases


def test_ac4_1_a_conflict_exit_then_a_clean_catch_up_then_done(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    assert w.node.call() == {"next": "implement", "mode": "gate", "fixes": []}  # right after the clean merge
    answer = w.node.drive({"next": "implement", "mode": "gate"})
    assert answer == catchup.wait_answer(w.repo.head(), True)
    w.node.start("done")  # the catch-up's own green wait: done without a new fetch
    w.run = wrap = Wrap()
    assert w.node.call() == catchup.done_answer(["digest", "report", "state"], [], [])
    assert not any(argv[1] == "fetch" for argv, _ in wrap.calls)
    assert w.state()["catchups"] == 0 and w.state()["base_merge"]["merge"] == w.repo.head()


def test_ac4_2_a_reap_before_the_resolution_review(w: World) -> None:
    w.repo.move_main(conflict=True)
    w.node.start("conflict")
    w.node.call()
    w.repo.resolve()
    review = w.node.implemented()
    assert review["next"] == "review"
    assert w.node.call() == review  # the reap: the review is owed again
    merge = w.repo.head()
    assert review["diffs"] == [f"git show --remerge-diff {merge}"]


def test_ac4_3_red_twice_then_blocked_then_a_fix_then_unblock(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    w.node.call()
    assert w.node.implemented("red")["mode"] == "gate"
    answer = w.node.implemented("red")
    assert answer == {"next": "blocked", "cause": "misses", "detail": "gate red, then gate red"}
    fid = w.finding()
    w.pending((fid, "code"))  # the orchestrator's fix
    assert w.node.call()["cause"] == "misses"  # sticky
    assert w.op("unblock") == {"unblocked": "misses"}
    answer = w.node.call()
    assert answer == {"next": "implement", "mode": "gate", "fixes": [{"finding_id": fid, "level": "code", "what": "w"}]}
    w.repo.commit()  # the fix on top of the merge
    answer = w.node.implemented()
    merge = w.state()["catchup"]["merge"]
    assert answer == {
        "next": "review",
        "head": w.repo.head(),
        "diffs": [f"git show --remerge-diff {merge}", f"git diff {merge}..{w.repo.head()}"],
    }
    assert w.state()["catchups"] == 1


def test_ac4_4_the_bound_at_a_fourth_catch_up(w: World) -> None:
    for _ in range(3):
        w.repo.move_main()
        w.node.start("conflict")
        assert w.node.drive(w.node.call())["next"] == "wait"
    assert w.state()["catchups"] == 3
    w.repo.move_main()
    w.node.start("conflict")
    before = w.repo.head()
    answer = w.node.call()
    assert answer["next"] == "blocked" and answer["cause"] == "catchups"
    assert w.repo.head() == before and w.repo.merges(before) == []
    assert w.op("unblock") == {"unblocked": "catchups"}
    assert w.node.call()["mode"] == "gate"  # counted: false re-runs step 2
    assert w.state()["catchups"] == 1


def test_ac4_5_rotation_keeps_the_base(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    w.node.drive(w.node.call())
    merge = w.state()["base_merge"]["merge"]
    assert w.op("base") == {"base": merge}
    w.rotate()
    assert w.op("base") == {"base": merge}
    w.build_round(1, merge, w.repo.head())  # the next round consumed it
    assert w.op("base") == {"base": None}


def test_ac4_6_a_stale_conflict_keeps_base_merge(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    w.node.drive(w.node.call())
    base_merge = w.state()["base_merge"]
    w.node.start("conflict")  # GitHub's stale mergeable answer
    assert w.node.call() == catchup.wait_answer(w.node.head, False)
    assert w.state()["base_merge"] == base_merge and w.state()["catchups"] == 2


@pytest.mark.parametrize("catchups", [0, 3])
def test_ac4_7_the_done_no_op_does_not_count(w: World, catchups: int) -> None:
    st = catchup.empty_state()
    st["catchups"] = catchups
    w.node_dir.mkdir()
    catchup.save_state(w.node_dir, st)
    w.node.start("done")
    assert w.node.call()["next"] == "done"
    assert w.state()["catchups"] == 0  # done resets the counter, never counted


def test_ac4_8_a_double_reap_after_the_fix_commit_still_reviews_it(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    w.node.call()
    fid = w.finding()
    w.pending((fid, "code"))
    w.repo.commit()  # the implementer committed the fix, the node was reaped before the report
    assert w.node.call()["mode"] == "gate"
    assert w.node.call()["mode"] == "gate"  # a second reap
    answer = w.node.implemented()
    assert answer["next"] == "review" and len(answer["diffs"]) == 2


def test_ac4_9_a_conflict_exit_during_a_catch_ups_own_wait(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    first = w.node.drive(w.node.call())
    w.repo.move_main(conflict=True)
    w.node.start("conflict")
    assert w.node.call()["mode"] == "resolve"
    answer = w.node.drive({"next": "implement", "mode": "resolve"})
    assert answer["next"] == "wait" and answer["head"] != first["head"]
    assert w.state()["catchups"] == 2


def test_ac4_10_the_implementer_aborts_the_merge(w: World) -> None:
    w.repo.move_main(conflict=True)
    w.node.start("conflict")
    head = w.node.head
    w.node.call()
    w.repo.git("merge", "--abort")
    assert w.node.implemented()["mode"] == "finish"  # one miss: merged again, "finish the open merge"
    assert w.state()["catchup"]["misses"] == ["merge not committed"]
    w.repo.git("merge", "--abort")
    answer = w.node.implemented()
    assert answer == {"next": "blocked", "cause": "misses", "detail": "merge not committed, then merge not committed"}
    assert w.repo.merges(head) == []


def test_ac4_10_the_implementer_returns_without_committing(w: World) -> None:
    w.repo.move_main(conflict=True)
    w.node.start("conflict")
    w.node.call()
    assert w.node.implemented("green")["mode"] == "finish"  # MERGE_HEAD still open
    assert w.node.implemented("green")["cause"] == "misses"


def test_two_keyless_red_reports_at_an_unchanged_head_block(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    w.node.call()
    head = w.repo.head()
    w.node.implemented("red")
    assert w.repo.head() == head
    assert w.node.implemented("red")["cause"] == "misses"


def test_a_regressed_inv_is_invalid_state(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    w.node.call()
    w.node.implemented("red", inv=5)
    assert w.node.implemented("red", inv=5)["mode"] == "gate"  # the retry: ignored
    with pytest.raises(errors.CpError) as exc:
        w.node.implemented("red", inv=3)
    assert exc.value.detail == "inv regressed"
    w.node.implemented("green", inv=6)
    with pytest.raises(errors.CpError) as exc:
        w.node.implemented("red", inv=5)
    assert exc.value.detail == "inv regressed"


def test_ac4_11_a_fix_rejected_while_pending(w: World) -> None:
    fid, other = w.finding(), w.finding(task="T2")
    w.pending((fid, "code"), (fid, "plan"), (other, "code"), (9999, "code"))
    assert w.op("fixes") == {
        "next": "fixes",
        "open": [{"finding_id": fid, "level": "code", "what": "w"}],
        "drop": [other, 9999],
    }
    w.close(fid)
    assert w.op("fixes") == {"next": "fixes", "open": [], "drop": [fid, other, 9999]}
    w.node.start("done")
    assert w.node.call() == catchup.done_answer(["digest", "report", "state"], [], [fid, other, 9999])


def test_ac4_12_a_reap_inside_step_8_with_a_late_fix(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    w.node.drive(w.node.call())
    base_merge = w.state()["base_merge"]
    w.node.start("done")
    assert w.node.call()["next"] == "done"
    w.build_round(1, None, w.repo.head())
    w.report(1)  # step 8's converged report, then a reap; a late fix arrives
    fid = w.finding()
    w.pending((fid, "code"))
    assert w.op("done") == catchup.done_answer(["state"], [{"finding_id": fid, "level": "code", "what": "w"}], [])
    assert w.state()["base_merge"] == base_merge


def test_ac4_12_no_converged_report_and_a_late_fix_goes_to_the_fix_protocol(w: World) -> None:
    w.node.start("done")
    w.node.call()
    fid = w.finding()
    w.pending((fid, "code"))
    assert w.op("done")["next"] == "fixes"
    w.close(fid)
    assert w.op("done") == catchup.done_answer(["digest", "report", "state"], [], [fid])


def test_ac4_13_a_fix_after_done(w: World) -> None:
    w.node.start("done")
    w.node.call()
    w.build_round(1, None, w.repo.head())
    w.report(1)
    sql(w.db, "UPDATE tasks SET state = 'ready_to_merge' WHERE id = 'T1'")
    fid = w.finding()
    w.pending((fid, "plan"))
    assert w.op("done") == catchup.done_answer([], [{"finding_id": fid, "level": "plan", "what": "w"}], [])


def test_d6_an_older_rounds_converged_does_not_count(w: World) -> None:
    w.build_round(1, None, w.repo.head())
    w.report(1)  # converged before the done (a post-done catch-up opens no build round)
    w.node.start("done")
    w.node.call()
    assert w.op("done") == catchup.done_answer(["digest", "report", "state"], [], [])


def test_d6_no_done_recorded_is_not_converged(w: World) -> None:
    w.build_round(1, None, w.repo.head())
    w.report(1)
    assert w.op("done") == catchup.done_answer(["digest", "report", "state"], [], [])


def test_d6_a_later_blocked_supersedes_a_converged(w: World) -> None:
    w.node.start("done")
    w.node.call()
    w.build_round(1, None, w.repo.head())
    w.report(1)
    assert w.op("done")["left"] == ["state"]
    w.report(1, "blocked")
    assert w.op("done")["left"] == ["digest", "report", "state"]


def test_catchup_done_refuses_a_catch_up_in_progress(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    w.node.call()
    with pytest.raises(errors.CpError) as exc:
        w.op("done")
    assert exc.value.detail == "a catch-up is in progress"


# --------------------------------------------------------------------------- D8's guards


def test_guard_the_merge_reset_away(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    w.node.call()
    w.repo.git("reset", "--hard", w.node.head)
    w.repo.commit()
    _blocked_error(w.node.implemented(), "the catch-up merge is no longer in HEAD")


def test_guard_the_remote_ahead_unknown_locally(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    w.node.call()
    assert w.node.implemented()["next"] == "push"
    w.repo.git("fetch", "origin", BRANCH, where=w.repo.other)
    w.repo.git("checkout", "-b", BRANCH, f"origin/{BRANCH}", where=w.repo.other)
    (w.repo.other / "bot.txt").write_text("bot\n")
    w.repo.git("add", "bot.txt", where=w.repo.other)
    w.repo.git("commit", "-m", "bot", where=w.repo.other)
    w.repo.git("push", "origin", BRANCH, where=w.repo.other)
    _blocked_error(w.node.call(), "the remote branch is ahead of HEAD")


def test_guard_the_remote_ahead_known_but_diverged(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    w.node.call()
    w.node.implemented()
    merge = w.repo.head()
    w.repo.git("checkout", "-b", "side", w.node.head)
    w.repo.commit("side.txt")
    w.repo.git("push", "-f", "origin", f"side:refs/heads/{BRANCH}")
    w.repo.git("checkout", BRANCH)
    assert w.repo.head() == merge
    _blocked_error(w.node.call(), "the remote branch is ahead of HEAD")


def test_guard_head_is_not_the_pushed_head(w: World) -> None:
    w.repo.commit()
    w.node.start("done")
    _blocked_error(w.node.call(), "HEAD is not the pushed head")
    assert w.raw_state() is None


def test_guard_a_merge_open_outside_a_catch_up(w: World) -> None:
    w.repo.move_main(conflict=True)
    w.repo.git("fetch", "origin", "main")
    subprocess.run(["git", "merge", "origin/main"], cwd=w.repo.wt, capture_output=True, check=False)
    w.node.start("conflict")
    _blocked_error(w.node.call(), "a merge is open outside a catch-up")


def test_guard_a_foreign_merge(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    w.node.call()
    w.repo.git("reset", "--hard", w.node.head)
    w.repo.git("checkout", "-b", "foreign", "HEAD~1")
    w.repo.commit("foreign.txt")
    w.repo.git("checkout", BRANCH)
    w.repo.git("merge", "--no-ff", "--no-edit", "foreign")
    st = w.state()
    st["catchup"]["merge"] = None  # as if the merge was never recorded
    catchup.save_state(w.node_dir, st)
    _blocked_error(w.node.call(), "a merge that is not origin/main's")
    _blocked_error(w.node.implemented(), "a merge that is not origin/main's")
    assert w.state()["catchup"]["intake"] == []  # nothing written


def test_guard_head_moved_without_a_merge(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    w.node.call()
    st = w.state()
    st["catchup"]["merge"] = None
    catchup.save_state(w.node_dir, st)
    w.repo.git("reset", "--hard", w.node.head)
    w.repo.commit()
    _blocked_error(w.node.call(), "HEAD moved without a merge")


def test_guard_a_second_merge_is_open(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    w.node.call()
    w.repo.move_main(conflict=True)
    w.repo.git("fetch", "origin", "main")
    subprocess.run(["git", "merge", "origin/main"], cwd=w.repo.wt, capture_output=True, check=False)
    _blocked_error(w.node.call(), "a second merge is open")


def test_guard_head_is_not_pushed_at_done(w: World) -> None:
    w.repo.git("push", "-f", "origin", f"HEAD~1:refs/heads/{BRANCH}")
    w.node.start("done")
    _blocked_error(w.node.call(), "HEAD is not pushed")


def test_guard_no_remote_branch_at_done(w: World) -> None:
    w.repo.git("push", "origin", f":refs/heads/{BRANCH}")
    w.node.start("done")
    _blocked_error(w.node.call(), "HEAD is not pushed")


def test_the_remote_ahead_at_the_pre_check_then_repaired(w: World) -> None:
    w.repo.git("fetch", "origin", BRANCH, where=w.repo.other)
    w.repo.git("checkout", "-b", BRANCH, f"origin/{BRANCH}", where=w.repo.other)
    (w.repo.other / "bot.txt").write_text("bot\n")
    w.repo.git("add", "bot.txt", where=w.repo.other)
    w.repo.git("commit", "-m", "bot", where=w.repo.other)
    w.repo.git("push", "origin", BRANCH, where=w.repo.other)
    w.node.start("done")
    _blocked_error(w.node.call(), "the remote branch is ahead of HEAD")
    w.repo.git("pull", "--no-rebase", "--no-edit", "origin", BRANCH)  # the maintainer's repair
    w.repo.push()
    w.node.head = w.repo.head()  # the node's error-block re-entry to the wait
    w.node.start("done")
    assert w.node.call()["next"] == "done"


def test_the_adopt_rule_keeps_from_and_replays_the_push(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    assert w.node.call()["mode"] == "gate"
    w.node.start("done")  # another `from`: adopted, `from` kept
    assert w.node.call()["mode"] == "gate"
    assert w.state()["catchup"]["from"] == "conflict"
    assert w.node.implemented()["next"] == "push"
    w.repo.push()
    _crash("catchup.after_push_write", w.node.call)
    answer = w.node.call()
    assert answer == catchup.wait_answer(w.repo.head(), True)
    assert w.node.call() == answer  # replayed from `last`, never INVALID_STATE


def test_the_adopt_rule_keeps_from_and_replays_the_done(w: World) -> None:
    w.repo.move_main()
    w.node.start("done")
    assert w.node.call()["mode"] == "gate"
    w.repo.git("push", "-f", "origin", f"{w.node.head}:refs/heads/main", where=w.repo.wt)  # main goes back
    w.repo.git("reset", "--hard", w.node.head)
    st = w.state()
    st["catchup"].update(merge=None, clean=None)
    catchup.save_state(w.node_dir, st)
    w.node.start("conflict")  # adopted under from: done
    _crash("catchup.after_done_write", w.node.call)  # 3b: nothing to merge any more → step 5
    assert w.node.call()["next"] == "done"
    assert w.node.call()["next"] == "done"
    assert w.state()["catchup"] is None and w.state()["catchups"] == 0


def test_the_no_fetch_done_is_refused_after_a_build_round(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    w.node.drive(w.node.call())
    w.build_round(1, w.state()["base_merge"]["merge"], w.repo.head())  # a round that added no commit
    w.node.start("done")
    w.run = wrap = Wrap()
    assert w.node.call()["next"] == "done"
    assert any(argv[1] == "fetch" for argv, _ in wrap.calls)


def test_a_counted_done_whose_main_went_back_ends_at_step_5(w: World) -> None:
    w.repo.move_main()
    w.node.start("done")
    _crash("catchup.after_count", w.node.call)
    w.repo.git("push", "-f", "origin", f"{w.node.head}:refs/heads/main")  # main rewound under the catch-up
    assert w.node.call()["next"] == "done"
    assert w.state()["catchup"] is None


# --------------------------------------------------------------------------- base and unblock (D5, D4 step 1)


def test_base_refuses_an_open_merge_or_an_unpushed_catch_up_merge(w: World) -> None:
    w.repo.move_main(conflict=True)
    w.node.start("conflict")
    w.node.call()
    with pytest.raises(errors.CpError) as exc:
        w.op("base")
    assert "merge is open" in exc.value.detail
    w.repo.resolve()
    w.node.implemented()
    with pytest.raises(errors.CpError) as exc:
        w.op("base")
    assert exc.value.detail == "a catch-up merge is not pushed yet"


def test_base_is_null_once_the_merge_left_head(w: World) -> None:
    w.repo.move_main()
    w.node.start("conflict")
    w.node.drive(w.node.call())
    w.repo.git("reset", "--hard", "HEAD~1")
    assert w.op("base") == {"base": None}


def test_unblock_with_no_block_writes_nothing(w: World) -> None:
    assert w.op("unblock") == {"unblocked": None}
    w.repo.move_main()
    w.node.start("conflict")
    w.node.call()
    before = w.raw_state()
    assert w.op("unblock") == {"unblocked": None}
    assert w.raw_state() == before


def test_a_review_with_commits_beside_the_merge(w: World) -> None:
    w.repo.move_main(conflict=True)
    w.node.start("conflict")
    w.node.call()
    w.repo.git("merge", "--abort")
    w.repo.commit("beside.txt")  # a commit beside the merge, before it
    w.repo.git("merge", "--no-ff", "--no-edit", "origin/main", "-X", "ours")
    answer = w.node.implemented()
    merge = w.repo.head()
    assert answer["diffs"] == [f"git show --remerge-diff {merge}", f"git diff {w.node.head}..{merge}^1"]
    assert w.node.call(reviewed="f" * 40)["next"] == "review"  # another sha: ignored
