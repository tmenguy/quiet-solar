"""The lifecycle hooks (QS-406 §4): restart on new code (``code_version``) and the self-check (``selfcheck``).

``code_version`` (every tick): the disk version of the main checkout is
compared with the version the daemon loaded. A change must read the same on
two consecutive ticks, with no git operation in progress in the main
checkout, before the daemon is asked to restart (D7): it then finishes its
tick, exits ``new_code``, and ``cli._daemon`` starts the new code.

``selfcheck`` (every tick, deciding from ``meta``): while the loaded code is
the code on disk and no pass or override is recorded for it, three steps run
(``schema``, ``entry``, ``registry``), each under a broad guard that records
any exception as a failure. A failure is retried every ``SELFCHECK_RETRY_S``,
forever; ``selfcheck_failed`` is synced to every open run on each tick while
the latest record is a failure for the disk version and no override exists
(D18). ``cli``, ``tools`` and ``items`` are imported lazily (D17).
"""

from __future__ import annotations

import importlib
import json
import sqlite3
import sys
from datetime import timedelta
from typing import Any

from . import activeloop, alerts, codever, daemon, db, mergegate, migrations, ticks
from . import clock as clock_mod

CODE_VERSION = "code_version"
SELFCHECK = "selfcheck"


class CheckFailed(Exception):
    """A self-check step's own refusal (any other exception counts as a failure too)."""


# --------------------------------------------------------------------------- code_version (§4.1)


def code_version_hook(conn: sqlite3.Connection, clock: clock_mod.Clock) -> None:
    loaded = daemon.started_code_version()
    if loaded is None:
        return
    main = activeloop.seams().main
    current = codever.code_version(main)
    if current == loaded or codever.git_busy(main) is not None:
        daemon.set_restart_candidate(None)  # unchanged, or mid-operation: the debounce starts over
        return
    if daemon.restart_candidate() == current:
        daemon.request_restart("new_code")
        return
    daemon.set_restart_candidate(current)


# --------------------------------------------------------------------------- selfcheck (§4.3)


def _module(name: str) -> Any:
    return importlib.import_module(f"{__package__}.{name}")


def _expected_tools() -> set[str]:
    tools, items = _module("tools"), _module("items")
    return set(tools.BUILTIN_NAMES) | {spec.name for spec in items.SPECS}


def _require_registries(tool_names: Any, hook_names: Any) -> None:
    missing_tools = _expected_tools() - set(tool_names)
    missing_hooks = set(activeloop.BUILTIN_NAMES) - set(hook_names)
    if missing_tools or missing_hooks:
        raise CheckFailed(f"missing tools {sorted(missing_tools)}, missing tick hooks {sorted(missing_hooks)}")


def _step_schema(conn: sqlite3.Connection, seams: activeloop.Seams) -> None:
    found = db.user_version(conn)
    if found != migrations.current_schema_version():
        raise CheckFailed(f"the DB is at schema v{found}, the code at v{migrations.current_schema_version()}")
    check = conn.execute("PRAGMA quick_check").fetchone()[0]
    if check != "ok":
        raise CheckFailed(f"quick_check: {check}")


def _step_entry(conn: sqlite3.Connection, seams: activeloop.Seams) -> None:
    venv_python = seams.main / "venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable
    res = seams.runner.run(
        [python, str(seams.main / "scripts" / "qs" / "cp.py"), "version"],
        cwd=seams.main,
        timeout=ticks.HOOK_SUBPROCESS_S,
    )
    if not res.ok:
        raise CheckFailed(f"cp.py version exited {res.returncode}: {res.stderr.strip()[-300:]}")
    data = json.loads(res.stdout)
    if data.get("schema_version") != migrations.current_schema_version():
        raise CheckFailed(f"cp.py version reports schema {data.get('schema_version')!r}")
    _require_registries(data.get("tools") or [], data.get("tick_hooks") or [])


def _step_registry(conn: sqlite3.Connection, seams: activeloop.Seams) -> None:
    cli, tools = _module("cli"), _module("tools")
    _require_registries(tools.REGISTRY, [name for name, _ in ticks.registered()])
    cli.build_parser()


STEPS = (("schema", _step_schema), ("entry", _step_entry), ("registry", _step_registry))


def _open_runs(conn: sqlite3.Connection) -> tuple[str, ...]:
    return tuple(r[0] for r in conn.execute("SELECT id FROM runs WHERE state = 'open' ORDER BY rowid"))


def sync_failed_locked(conn: sqlite3.Connection, clock: clock_mod.Clock, version: str) -> dict[str, Any]:
    """D18's ``selfcheck_failed`` set for the disk ``version``, inside the caller's transaction."""
    rec = mergegate.failed_record(conn, version)
    active = []
    if rec is not None and not mergegate.overridden(conn, version):
        payload = {"code_version": version, "tries": rec.get("tries"), "failures": rec.get("failures")}
        active.append(alerts.Condition(alerts.SELFCHECK_FAILED, version, _open_runs(conn), payload))
    return alerts.sync_locked(conn, clock, kinds={alerts.SELFCHECK_FAILED}, active=active)


def _due(rec: dict[str, Any] | None, clock: clock_mod.Clock) -> bool:
    if rec is None:
        return True
    try:
        last = clock_mod.parse(str(rec.get("at")))
    except ValueError:
        return True
    return clock.now() >= last + timedelta(seconds=mergegate.SELFCHECK_RETRY_S)


def run_steps(conn: sqlite3.Connection, clock: clock_mod.Clock, seams: activeloop.Seams) -> list[dict[str, Any]]:
    failures = []
    for name, step in STEPS:
        try:
            step(conn, seams)
        except Exception as exc:  # noqa: BLE001 — any exception in a step is a recorded failure (§1)
            failures.append({"step": name, "type": type(exc).__name__, "message": str(exc)[:500]})
        daemon.beat(conn, clock)
    return failures


def selfcheck_hook(conn: sqlite3.Connection, clock: clock_mod.Clock) -> None:
    seams = activeloop.seams()
    disk = codever.code_version(seams.main)
    with db.write(conn):
        sync_failed_locked(conn, clock, disk)  # first: a version change never leaves a stale alert
    loaded = daemon.started_code_version()
    if loaded is None or loaded != disk:
        return  # no version, or a restart is pending
    if mergegate.passed(conn, disk) or mergegate.overridden(conn, disk):
        return
    if not _due(mergegate.failed_record(conn, disk), clock):
        return
    failures = run_steps(conn, clock, seams)
    if codever.code_version(seams.main) != loaded:
        return  # the code changed during the steps: this result describes nothing on disk
    with db.write(conn):
        mergegate.record(conn, clock, loaded, ok=not failures, failures=failures)
        sync_failed_locked(conn, clock, loaded)
