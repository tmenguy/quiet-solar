"""Fixtures for the Control Plane tests (QS-399, story §14).

``scripts/qs`` goes on ``sys.path`` at import time and every
``control_plane`` submodule is imported eagerly: the parent conftest's
per-test purge only evicts modules imported *during* a test, so these
survive and keep one identity (one ``CpError`` class, one registry) for
the whole session.

The autouse fixture isolates every test: temporary DB and backup paths,
no inherited session id or token, a recording ``ProcessSetup``, and every
module-level registry reset.
"""

from __future__ import annotations

import importlib
import io
import json
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

SCRIPTS_QS = Path(__file__).resolve().parents[3] / "scripts" / "qs"
if str(SCRIPTS_QS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_QS))

MODULES = (
    "errors",
    "clock",
    "faults",
    "runner",
    "procsetup",
    "paths",
    "liveness",
    "schema_v1",
    "migrations",
    "db",
    "daemon",
    "tokens",
    "runs",
    "messages",
    "wait",
    "tasks",
    "reports",
    "questions",
    "decisions",
    "nodes",
    "locks",
    "hooks",
    "merge_policy",
    "tools",
    "cli",
)
for _name in MODULES:
    importlib.import_module(f"control_plane.{_name}")

from control_plane import (  # noqa: E402
    cli,
    clock,
    db,
    faults,
    liveness,
    merge_policy,
    migrations,
    paths,
    procsetup,
    tools,
)
from control_plane.runner import RunResult  # noqa: E402

REAL_PROCSETUP_GET = procsetup.get
REAL_CODE_ROOT = paths.code_root
REAL_MAIN_CHECKOUT = paths.main_checkout
REAL_MAIN_HEAD_BRANCH = paths.main_head_branch
REAL_MAKE_DEPS = cli.make_deps

ENV_CLEARED = (
    "CLAUDE_CODE_SESSION_ID",
    "QS_CP_TOKEN",
    "QS_CP_MAX_GATES",
    "QS_CP_MAX_NODES",
)


class FakeProcessSetup:
    """Records ``become_group_leader`` / ``install_sigterm`` and changes nothing."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def become_group_leader(self) -> None:
        self.calls.append(("group_leader", None))

    def install_sigterm(self, fn: Callable[[int, Any], None]) -> None:
        self.calls.append(("sigterm", fn))


@dataclass
class Call:
    argv: list[str]
    cwd: str | None
    env_extra: dict[str, str]
    env_remove: tuple[str, ...]
    detach: bool
    timeout: float | None


def argv_matches(argv: Sequence[str], pattern: Sequence[str]) -> bool:
    """``pattern`` occurs contiguously in ``argv``; an element also matches a path ending in ``/<element>``."""
    n = len(pattern)
    for i in range(len(argv) - n + 1):
        if all(a == p or a.endswith("/" + p) for a, p in zip(argv[i : i + n], pattern)):
            return True
    return False


Response = RunResult | Callable[[Call], RunResult]


@dataclass
class FakeRunner:
    """Records every call; answers from rules (last registered wins), else exit 0."""

    calls: list[Call] = field(default_factory=list)
    rules: list[tuple[tuple[str, ...], Response]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def on(self, pattern: Sequence[str], response: Response | str, code: int = 0) -> None:
        if isinstance(response, str):
            response = RunResult(code, response, "")
        self.rules.append((tuple(pattern), response))

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | str | None = None,
        env_extra: dict[str, str] | None = None,
        env_remove: Sequence[str] = (),
        timeout: float | None = None,
        detach: bool = False,
    ) -> RunResult:
        call = Call(
            list(argv), None if cwd is None else str(cwd), dict(env_extra or {}), tuple(env_remove), detach, timeout
        )
        with self._lock:
            self.calls.append(call)
            rules = list(self.rules)
        for pattern, response in reversed(rules):
            if argv_matches(call.argv, pattern):
                return response(call) if callable(response) else response
        return RunResult(0, "", "")

    def matching(self, *pattern: str) -> list[Call]:
        with self._lock:
            return [c for c in self.calls if argv_matches(c.argv, pattern)]


@pytest.fixture
def fake_clock() -> clock.FakeClock:
    return clock.FakeClock()


@pytest.fixture
def fake_runner() -> FakeRunner:
    return FakeRunner()


class FakeProbe(liveness.ProcessProbe):
    """Every pid and group is alive until ``kill``-ed; ``me()`` is a fresh pid per call."""

    def __init__(self) -> None:
        super().__init__(FakeRunner())
        self.dead_pids: set[int] = set()
        self.dead_groups: set[int] = set()
        self._next = 50_000
        self._lock = threading.Lock()
        self.issued: dict[int, list[int]] = {}

    def alive(self, pid: int | None, start: str | None) -> bool:
        if pid is None or pid in self.dead_pids:
            return False
        return start is None or start == f"start-{pid}"

    def group_alive(self, pgid: int | None) -> bool:
        return pgid is not None and pgid not in self.dead_groups

    def start_of(self, pid: int) -> str | None:
        return None if pid in self.dead_pids else f"start-{pid}"

    def me(self) -> liveness.Holder:
        with self._lock:
            self._next += 1
            pid = self._next
            self.issued.setdefault(threading.get_ident(), []).append(pid)
        return liveness.Holder(pid, f"start-{pid}", pid)

    def reap_thread(self) -> None:
        """A finished ``cp.py`` process is dead: kill the pids this thread's call handed out."""
        with self._lock:
            pids = self.issued.pop(threading.get_ident(), [])
        for pid in pids:
            self.kill(pid)

    def kill(self, pid: int | None, *, group: bool = True) -> None:
        if pid is not None:
            self.dead_pids.add(pid)
            if group:
                self.dead_groups.add(pid)


def agent(session_id: str, name: str | None = None, **kw: Any) -> liveness.Agent:
    fields: dict[str, Any] = {
        "id": None,
        "cwd": None,
        "kind": "background",
        "status": "idle",
        "state": None,
        "pid": None,
        "started_at_ms": None,
    }
    fields.update(kw)
    return liveness.Agent(session_id=session_id, name=name, **fields)


class FakeClaude(liveness.ClaudeCli):
    """``agents()`` returns ``listing`` (``None`` → a failed listing); launches go to the ``FakeRunner``."""

    def __init__(self, run: FakeRunner) -> None:
        super().__init__(run)
        self.listing: list[liveness.Agent] | None = []
        self.listings = 0
        self.before_list: Callable[[], None] | None = None

    def agents(self) -> list[liveness.Agent]:
        self.listings += 1
        if self.before_list is not None:
            self.before_list()
        if self.listing is None:
            raise liveness.errors.CpError("INTERNAL", "fake listing failure")
        return list(self.listing)


@dataclass
class FakePopen:
    calls: list[tuple[list[str], dict[str, Any]]] = field(default_factory=list)
    on_call: Callable[[list[str], dict[str, Any]], None] | None = None

    def __call__(self, argv: list[str], **kwargs: Any) -> object:
        self.calls.append((list(argv), kwargs))
        if self.on_call is not None:
            self.on_call(list(argv), kwargs)
        return object()


@dataclass
class FakeKill:
    calls: list[tuple[int, int]] = field(default_factory=list)
    on_call: Callable[[int, int], None] | None = None

    def __call__(self, pid: int, sig: int) -> None:
        self.calls.append((pid, sig))
        if self.on_call is not None:
            self.on_call(pid, sig)


@pytest.fixture
def fake_probe() -> FakeProbe:
    return FakeProbe()


@pytest.fixture
def fake_claude(fake_runner: FakeRunner) -> FakeClaude:
    return FakeClaude(fake_runner)


@pytest.fixture
def fake_popen() -> FakePopen:
    return FakePopen()


@pytest.fixture
def fake_kill() -> FakeKill:
    return FakeKill()


@pytest.fixture
def deps(fake_clock, fake_runner, fake_probe, fake_claude, fake_popen, fake_kill) -> cli.Deps:
    return cli.Deps(
        clock=fake_clock, runner=fake_runner, probe=fake_probe, claude=fake_claude, popen=fake_popen, kill=fake_kill
    )


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state" / "test_state.db"


@pytest.fixture
def migrated(db_path: Path) -> Path:
    migrations.migrate(db_path, role="test")
    return db_path


@pytest.fixture
def conn(migrated: Path) -> Iterator[Any]:
    c = db.connect(migrated)
    try:
        yield c
    finally:
        c.close()


@pytest.fixture(autouse=True)
def _cp_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deps: cli.Deps) -> Iterator[FakeProcessSetup]:
    for name in ENV_CLEARED:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QS_CP_DB", str(tmp_path / "state" / "test_state.db"))
    monkeypatch.setenv("QS_CP_BACKUP_DIR", str(tmp_path / "backups"))
    (tmp_path / "state").mkdir()
    # A fake main checkout, identical locally (a linked worktree) and in CI.
    fake_main = tmp_path / "main"
    (fake_main / ".git").mkdir(parents=True)
    (fake_main / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    monkeypatch.setattr(paths, "code_root", lambda: fake_main)
    monkeypatch.setattr(paths, "main_checkout", lambda root: fake_main)
    monkeypatch.setattr(paths, "main_head_branch", lambda main_dir: "main")
    setup = FakeProcessSetup()
    monkeypatch.setattr(procsetup, "get", lambda: setup)
    monkeypatch.setattr(cli, "make_deps", lambda: deps)
    faults.reset()
    try:
        yield setup
    finally:
        faults.reset()
        merge_policy.reset()
        tools.reset()


@pytest.fixture
def fake_setup(_cp_isolation: FakeProcessSetup) -> FakeProcessSetup:
    return _cp_isolation


@pytest.fixture
def fake_main(tmp_path: Path) -> Path:
    return tmp_path / "main"


def run_cli(*argv: str, stdin: str = "") -> tuple[int, Any]:
    """``cli.main`` in-process → ``(exit_code, parsed JSON | raw text | None)``."""
    out = io.StringIO()
    try:
        code = cli.main(list(argv), stdin=io.StringIO(stdin), stdout=out)
    finally:
        probe = cli.make_deps().probe
        if isinstance(probe, FakeProbe):
            probe.reap_thread()
    text = out.getvalue()
    if not text.strip():
        return code, None
    try:
        return code, json.loads(text)
    except ValueError:
        return code, text


@pytest.fixture
def invoke() -> Callable[..., tuple[int, Any]]:
    return run_cli


# --------------------------------------------------------------------------- shared helpers

ORCH = "S-orch"


def open_run(name: str = "r1", session: str = ORCH, **extra: str) -> tuple[str, str]:
    """``run open`` through the CLI → ``(run_id, token)``."""
    argv = ["run", "open", "--name", name, "--title", f"title {name}", "--session-id", session]
    for key, value in extra.items():
        argv += [f"--{key.replace('_', '-')}", value]
    code, out = run_cli(*argv)
    assert code == 0, out
    return out["run_id"], out["token"]


def sql(path: Path, statement: str, params: Sequence[Any] = ()) -> list[Any]:
    """Run one statement on the test DB outside any domain code (fixtures, assertions)."""
    c = db.connect(path)
    try:
        return c.execute(statement, tuple(params)).fetchall()
    finally:
        c.close()


def insert_task(path: Path, task_id: str, run_id: str | None, state: str = "building", **cols: Any) -> None:
    fields = {"id": task_id, "run_id": run_id, "title": task_id, "kind": "feature", "state": state}
    fields.update({"created_at": "x", "updated_at": "x"})
    fields.update(cols)
    names = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    sql(path, f"INSERT INTO tasks ({names}) VALUES ({marks})", list(fields.values()))


def insert_node(
    path: Path, node_id: str, run_id: str, task_id: str, *, generation: int = 1, state: str = "running", **cols: Any
) -> str:
    """A ``nodes`` row inserted directly → its token."""
    nonce = cols.pop("nonce", f"{int(node_id[1:]):032x}")
    fields = {
        "id": node_id,
        "run_id": run_id,
        "task_id": task_id,
        "generation": generation,
        "name": f"{run_id}-{task_id}-g{generation}",
        "nonce": nonce,
        "state": state,
        "spawned_at": "x",
        "updated_at": "x",
    }
    fields.update(cols)
    names = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    sql(path, f"INSERT INTO nodes ({names}) VALUES ({marks})", list(fields.values()))
    return f"node:{node_id}.{generation}.{nonce}"
