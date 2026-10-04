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

MODULES = ("errors", "clock", "faults", "runner", "procsetup", "paths", "liveness", "cli")
for _name in MODULES:
    importlib.import_module(f"control_plane.{_name}")

from control_plane import cli, clock, faults, paths, procsetup  # noqa: E402
from control_plane.runner import RunResult  # noqa: E402

REAL_PROCSETUP_GET = procsetup.get
REAL_CODE_ROOT = paths.code_root
REAL_MAIN_CHECKOUT = paths.main_checkout
REAL_MAIN_HEAD_BRANCH = paths.main_head_branch

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


@pytest.fixture(autouse=True)
def _cp_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeProcessSetup]:
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
    faults.reset()
    try:
        yield setup
    finally:
        faults.reset()


@pytest.fixture
def fake_setup(_cp_isolation: FakeProcessSetup) -> FakeProcessSetup:
    return _cp_isolation


@pytest.fixture
def fake_main(tmp_path: Path) -> Path:
    return tmp_path / "main"


@pytest.fixture
def fake_clock() -> clock.FakeClock:
    return clock.FakeClock()


@pytest.fixture
def fake_runner() -> FakeRunner:
    return FakeRunner()


def run_cli(*argv: str, stdin: str = "") -> tuple[int, Any]:
    """``cli.main`` in-process → ``(exit_code, parsed JSON | raw text | None)``."""
    out = io.StringIO()
    code = cli.main(list(argv), stdin=io.StringIO(stdin), stdout=out)
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
