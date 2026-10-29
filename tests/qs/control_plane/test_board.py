"""#428: the board — its file (D2), its view model (D4, D5) and its tick hook (D8)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from control_plane import board, paths


def _mode(p: Path) -> int:
    return stat.S_IMODE(os.stat(p).st_mode)


class TestFile:
    def test_board_path_splits_per_db(self, tmp_path: Path) -> None:
        a, b = board.board_path(tmp_path / "a.db"), board.board_path(tmp_path / "b.db")
        assert a != b and a.name == b.name == "index.html"
        assert a.parent.parent == b.parent.parent == paths.board_dir() and len(a.parent.name) == 8
        assert board.board_path(tmp_path / "x" / ".." / "a.db") == a  # the resolved path

    def test_write_is_private(self, tmp_path: Path) -> None:
        dest = board.board_path(tmp_path / "a.db")
        assert board.write("<p>é</p>", dest) == dest
        assert dest.read_text(encoding="utf-8") == "<p>é</p>"
        assert _mode(dest) == 0o600 and _mode(dest.parent) == 0o700 and _mode(paths.board_dir()) == 0o700
        board.write("<p>2</p>", dest)
        assert dest.read_text() == "<p>2</p>" and _mode(dest) == 0o600
        assert sorted(p.name for p in dest.parent.iterdir()) == ["index.html"]

    def test_two_interleaved_writers(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        dest = board.board_path(tmp_path / "a.db")
        real_replace = os.replace
        nested: list[bool] = []

        def replace(src: str, dst: Path) -> None:
            if not nested:  # writer A's replace first runs a whole write for writer B
                nested.append(True)
                board.write("B" * 1000, dest)
            real_replace(src, dst)

        monkeypatch.setattr(board.os, "replace", replace)
        assert board.write("A" * 5000, dest) == dest
        assert dest.read_text() == "A" * 5000
        assert sorted(p.name for p in dest.parent.iterdir()) == ["index.html"]

    def test_a_failure_before_the_replace_keeps_the_previous_page(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dest = board.board_path(tmp_path / "a.db")
        board.write("old", dest)

        def boom(src: str, dst: Path) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(board.os, "replace", boom)
        with pytest.raises(OSError, match="disk full"):
            board.write("new", dest)
        assert dest.read_text() == "old"
        assert sorted(p.name for p in dest.parent.iterdir()) == ["index.html"]

    def test_a_board_dir_inside_a_checkout_is_refused(
        self, tmp_path: Path, fake_main: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from control_plane import errors

        monkeypatch.setenv("QS_CP_BOARD_DIR", str(fake_main / "board"))
        with pytest.raises(errors.CpError) as exc:
            board.board_path(tmp_path / "a.db")
        assert exc.value.code == "POLICY_REFUSED" and "QS_CP_BOARD_DIR" in exc.value.payload()["hint"]


# --------------------------------------------------------------------------- the view model (AC2)

from . import board_fixture as bf  # noqa: E402

MAIN = Path("/main checkout")


def _model(snap: dict, **kw) -> dict:
    return board.model(snap, now=kw.get("now", bf.NOW), main=MAIN, code_schema=kw.get("code_schema", 6))


def _run(m: dict, rid: str) -> dict:
    return next(r for r in m["runs"] if r["id"] == rid)


def _task(m: dict, tid: str) -> dict:
    return next(t for r in m["runs"] for t in r["tasks"] if t["id"] == tid)


class TestModel:
    def test_the_fixture_has_every_snapshot_key(self) -> None:
        from control_plane import snapshot

        assert set(bf.FIXTURE_SNAPSHOT) == set(snapshot.KEYS) == set(bf.MINIMAL_SNAPSHOT)

    def test_waiting_for_you(self) -> None:
        m = _model(bf.fixture())
        got = [(w["rule"], w["run"], w["where"], w["copy"]) for w in m["waiting"]]
        anywhere = {"to": "any_chat", "chats": ["alpha-chat", "beta"]}
        assert got == [
            (1, "R2", {"to": "new_session"}, ["resume", "R2"]),  # orchestrator_dead, first_seen 90 s ago
            (1, "R1", {"to": "chat", "chat": "alpha-chat"}, ["resume", "R1"]),  # not listening, 50 s ago
            (2, "R1", {"to": "chat", "chat": "alpha-chat"}, ["answer", "Q-1"]),  # blocking first
            (2, "R1", {"to": "chat", "chat": "alpha-chat"}, ["answer", "Q-2"]),
            (3, "R1", {"to": "chat", "chat": "alpha-chat"}, ["criteria", "T2"]),  # T1 (building) is not listed
            (5, None, anywhere, ["backup"]),
        ]
        assert m["waiting"][2]["detail"] == "question Q-1?" and "(blocking)" in m["waiting"][2]["text"]
        assert m["waiting"][4]["detail"] == "Snapshot keys"
        assert m["waiting"][5]["detail"] == "OSError: disk full"

    def test_orchestrator_state_per_run(self) -> None:
        m = _model(bf.fixture())
        assert _run(m, "R1")["orchestrator"] == "not listening" and _run(m, "R2")["orchestrator"] == "gone"
        assert _run(m, "R1")["chat"] == "alpha-chat" and _run(m, "R1")["short_id"] == "abc-1"
        assert _run(m, "R2")["chat"] == "beta" and _run(m, "R2")["short_id"] is None  # no lease: the run's name

    def test_placement(self) -> None:
        m = _model(bf.fixture())
        assert [r["id"] for r in m["runs"]] == ["R1", "R2"]  # open runs only, in snapshot order
        assert [t["id"] for t in _run(m, "R1")["tasks"]] == [f"T{n}" for n in range(1, 11)]
        assert _run(m, "R2")["tasks"] == [] and _run(m, "R2")["graph"]["layers"] == []
        assert [(a["kind"], a["cleared"]) for a in _run(m, "R1")["alerts"]] == [
            ("orchestrator_not_listening", False),
            ("ci_red", False),
            ("node_stalled", True),  # the cleared ones after the open ones
        ]
        assert [a["kind"] for a in _run(m, "R2")["alerts"]] == ["orchestrator_dead"]  # no cross-run kind
        assert _run(m, "R1")["alerts"][1]["severity"] == "must-fix"
        assert [(r["id"], [t["task_id"] for t in r["tasks"]]) for r in _run(m, "R1")["reviews"]] == [
            ("V1", ["T1"]),
            ("V2", ["T1"]),  # only this run's per-task outcome
        ]
        assert _run(m, "R2")["reviews"] == []
        [group] = m["cross_run"]["alerts"]
        assert group["kind"] == "overlap_cross_run" and group["runs"] == ["alpha", "beta"]
        [cross] = m["cross_run"]["reviews"]
        assert cross["id"] == "V2" and [(t["task_id"], t["outcome"]) for t in cross["tasks"]] == [
            ("T1", "fail"),
            ("T30", None),
        ]

    def test_progress_counts_work_items(self) -> None:
        p = _run(_model(bf.fixture()), "R1")["progress"]
        assert (p["done"], p["dropped"], p["remaining"], p["total"]) == (2, 1, 7, 10)
        assert p["by_state"]["building"] == 2 and p["by_state"]["blocked"] == 1

    def test_the_task_rows(self) -> None:
        m = _model(bf.fixture())
        t1 = _task(m, "T1")
        assert t1["node"]["state"] == "running" and t1["node"]["short_id"] == "n2-short"  # the highest generation
        assert t1["copies"] == [["where_is", "T1"], ["attach", "n2-short"]]
        assert (t1["criteria_met"], t1["criteria_total"], t1["criteria_validated"]) == (1, 2, False)
        assert t1["open_findings"] == 2 and _task(m, "T10")["open_findings"] == 1
        assert t1["ci_sha7"] == "0123456" and t1["issue"] == 428
        assert [q["id"] for q in t1["questions"]] == ["Q-1", "Q-3"]  # open and asked only
        assert _task(m, "T4")["node"]["taken_over"] is True
        assert _task(m, "T10")["copies"] == [["where_is", "T10"]]  # a live node with no short_id
        assert _task(m, "T6")["copies"] == [["where_is", "T6"]]  # a reaped node
        assert _task(m, "T6")["criteria_validated"] is True and _task(m, "T6")["criteria_met"] == 1
        assert _task(m, "T2")["node"] is None and _task(m, "T2")["pr_url"] is None
        assert _task(m, "T9")["blocked_from"] == "building" and _task(m, "T10")["item_of"] == "T1"

    def test_the_graph(self) -> None:
        g = _run(_model(bf.fixture()), "R1")["graph"]
        layers = [[n["id"] for n in layer] for layer in g["layers"]]
        assert layers == [["T2", "T6", "T7", "T8", "T9", "T10"], ["T1"], ["T3", "T4", "T5"]]
        assert [n["unresolved"] for n in g["layers"][-1]] == [True, True, True]
        assert g["edges"] == [("T2", "T1"), ("T4", "T3"), ("T3", "T4"), ("T3", "T5")]
        assert g["outside"] == 1

    def test_a_long_title_is_cut_in_the_model(self) -> None:
        snap = bf.fixture()
        snap["tasks"][1]["title"] = "x" * 100
        node = _run(_model(snap), "R1")["graph"]["layers"][0][0]
        assert len(node["short_title"]) == board.GRAPH_TITLE_MAX and node["title"] == "x" * 100

    def test_closed_runs_are_summary_rows(self) -> None:
        closed = _model(bf.fixture())["closed_runs"]
        assert [(c["name"], c["done"], c["dropped"], c["remaining"]) for c in closed] == [
            ("gamma", 1, 1, 0),
            ("delta", 0, 0, 1),
        ]
        assert closed[0]["closed_at"] == bf.ago(60)

    def test_system(self) -> None:
        s = _model(bf.fixture())["system"]
        assert s["daemon_running"] is False and s["pid"] is None and s["heartbeat_age_s"] is None  # NULL pid
        assert (s["db_schema"], s["code_schema"]) == (6, 6)
        assert s["backup"]["last_error"] == {"at": bf.ago(100), "error": "OSError: disk full"}
        assert [h["id"] for h in s["hook_alerts"]] == [2, 1]
        fresh = _model(bf.MINIMAL_SNAPSHOT)["system"]
        assert fresh["daemon_running"] is True and fresh["heartbeat_age_s"] == 5.0 and fresh["pid"] == 9

    def test_nothing_waiting(self) -> None:
        m = _model(bf.MINIMAL_SNAPSHOT)
        assert m["waiting"] == [] and m["empty"] is False and m["closed_runs"] == []

    def test_header(self) -> None:
        m = _model(bf.fixture())
        assert m["generated_at"] == bf.NOW_ISO and m["generated_at_s"] == int(bf.NOW.timestamp())
        assert m["generated_hms"] == "12:00:00" and m["main"] == str(MAIN)
        assert (m["refresh_s"], m["stale_s"], m["force_reload_s"]) == (10, 120, 60)

    def test_an_empty_snapshot(self) -> None:
        from control_plane import snapshot

        m = _model(snapshot.empty())
        assert m["empty"] is True and m["waiting"] == [] and m["empty_line"] == board.EMPTY_LINE
        assert "runs" not in m


FAILED = {"code_version": "v1", "ok": False, "at": "t", "tries": 3, "failures": []}


class TestHalt:
    @pytest.mark.parametrize(
        ("records", "halt"),
        [
            ({}, None),
            ({"selfcheck": FAILED}, "failed"),
            ({"selfcheck": {**FAILED, "tries": 2}}, None),  # still retrying
            ({"selfcheck": {**FAILED, "tries": "x"}}, None),  # an unreadable count is 1
            ({"selfcheck": {**FAILED, "ok": True}}, None),
            ({"selfcheck": FAILED, "override": {"code_version": "v1"}}, None),  # the override masks it
            ({"selfcheck": FAILED, "override": {"code_version": "v0"}}, "failed"),  # another version's override
            ({"pending": {"code_version": "v1", "since": bf.ago(700)}}, "stuck"),
            ({"pending": {"code_version": "v1", "since": bf.ago(60)}}, None),  # pending, not stuck
            ({"selfcheck": FAILED, "pending": {"since": bf.ago(60)}}, None),  # a pending mark is newer
            ({"pending": {"code_version": "v1", "since": "x"}}, None),  # an unreadable `since`
        ],
    )
    def test_the_gate_cases(self, records: dict, halt: str | None) -> None:
        m = _model(bf.with_gate(**records))
        assert board._halt(bf.with_gate(**records)["merge_gate"], bf.NOW) == halt
        rule4 = [w for w in m["waiting"] if w["rule"] == 4]
        assert [w["copy"] for w in rule4] == ([] if halt is None else [["halt", halt]])

    def test_a_halt_with_no_open_run_goes_to_a_new_session(self) -> None:
        snap = bf.with_gate(selfcheck=FAILED)
        for run in snap["runs"]:
            run["state"] = "closed"
        m = _model(snap)
        assert [(w["rule"], w["where"]) for w in m["waiting"]] == [
            (4, {"to": "new_session"}),
            (5, {"to": "new_session"}),
        ]
        assert m["runs"] == [] and len(m["closed_runs"]) == 4 and m["cross_run"]["alerts"] == []


class TestLayout:
    def test_empty(self) -> None:
        assert board._layout([], []) == ([], [])

    def test_a_diamond(self) -> None:
        edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("a", "d")]
        assert board._layout(["d", "c", "b", "a"], edges) == ([["a"], ["c", "b"], ["d"]], [])

    def test_a_3_cycle(self) -> None:
        assert board._layout(["a", "b", "c"], [("a", "b"), ("b", "c"), ("c", "a")]) == (
            [["a", "b", "c"]],
            ["a", "b", "c"],
        )

    def test_a_cycle_with_a_tail(self) -> None:
        edges = [("r", "a"), ("a", "b"), ("b", "a"), ("b", "t")]
        assert board._layout(["r", "a", "b", "t"], edges) == ([["r"], ["a", "b", "t"]], ["a", "b", "t"])

    def test_unknown_ends_and_duplicates_are_ignored(self) -> None:
        assert board._layout(["a", "b"], [("a", "b"), ("a", "b"), ("z", "a")]) == ([["a"], ["b"]], [])


class TestTolerance:
    @pytest.mark.parametrize(
        "mutate",
        [
            lambda s: s["questions"][0].update(created_at="x"),
            lambda s: s["questions"][0].update(created_at=None, blocking="x"),
            lambda s: s["alerts"][0].update(first_seen=None),
            lambda s: s["merge_gate"].update(selfcheck=[1, 2]),
            lambda s: s["merge_gate"].update(selfcheck={**FAILED, "ok": "true"}),
            lambda s: s["merge_gate"].update(selfcheck={"ok": False, "tries": 9}, override={"reason": "r"}),
            lambda s: s["merge_gate"].update(pending={"since": None}),
            lambda s: s.update(merge_gate=None, backup="x"),
            lambda s: s["backup"].update(last_error="oops"),
            lambda s: s["daemon"].update(pid=5, heartbeat_at="x"),
            lambda s: s.update(daemon=[1]),
            lambda s: s["nodes"][1].update(generation="x"),
            lambda s: s["tasks"].append("not a row"),
            lambda s: s.update(hook_alerts=None),
            lambda s: s["runs"][0].update(lease="x"),
        ],
    )
    def test_never_raises(self, mutate) -> None:
        snap = bf.fixture()
        mutate(snap)
        m = _model(snap)
        assert m["empty"] is False and m["system"]["code_schema"] == 6

    def test_a_malformed_override_never_masks(self) -> None:
        records = {"selfcheck": {"ok": False, "tries": 9}, "override": {"reason": "r"}}
        assert board._halt(records, bf.NOW) == "failed"

    def test_a_bad_backup_error_is_shown_raw(self) -> None:
        snap = bf.fixture()
        snap["backup"]["last_error"] = "oops"
        assert _model(snap)["waiting"][-1]["detail"] == "oops"

    def test_age(self) -> None:
        assert board._age(bf.NOW, bf.ago(30)) == 30.0
        assert board._age(bf.NOW, None) is None and board._age(bf.NOW, "x") is None and board._age(bf.NOW, 5) is None


# --------------------------------------------------------------------------- render_to and the hook (AC6)


class TestHook:
    def test_render_to_writes_the_page(self, conn, migrated: Path, fake_clock, fake_main: Path) -> None:
        dest = board.render_to(conn, fake_clock, migrated, fake_main)
        assert dest == board.board_path(migrated) and "Nothing is waiting for you." in dest.read_text()
        assert not conn.in_transaction  # the read is released

    def test_render_to_without_a_connection(self, tmp_path: Path, fake_clock, fake_main: Path) -> None:
        dest = board.render_to(None, fake_clock, tmp_path / "none.db", fake_main)
        assert board.EMPTY_LINE in dest.read_text()

    def test_writes_on_the_first_tick_then_every_period(self, conn, migrated: Path, fake_clock, monkeypatch) -> None:
        writes: list[Path] = []
        real = board.write
        monkeypatch.setattr(board, "write", lambda html, dest: writes.append(dest) or real(html, dest))
        board.board_hook(conn, fake_clock)
        assert writes == [board.board_path(migrated)] and writes[0].is_file()
        fake_clock.advance(board.BOARD_EVERY_S - 0.1)
        board.board_hook(conn, fake_clock)
        assert len(writes) == 1
        fake_clock.advance(0.1)
        board.board_hook(conn, fake_clock)
        assert len(writes) == 2

    @pytest.mark.parametrize(
        "exc",
        [OSError("disk full"), board.errors.CpError("POLICY_REFUSED", "inside"), board.sqlite3.OperationalError("x")],
    )
    def test_a_failing_write_is_logged_once_per_error(self, conn, fake_clock, monkeypatch, capsys, exc) -> None:
        def boom(html: str, dest: Path) -> Path:
            raise exc

        monkeypatch.setattr(board, "write", boom)
        for _ in range(3):
            board.board_hook(conn, fake_clock)  # never raises
            fake_clock.advance(board.BOARD_EVERY_S)
        err = capsys.readouterr().err
        assert err.count("[cp-board]") == 1 and type(exc).__name__ in err
        monkeypatch.setattr(board, "write", lambda html, dest: dest)
        board.board_hook(conn, fake_clock)  # a success forgets the logged error
        monkeypatch.setattr(board, "write", boom)
        fake_clock.advance(board.BOARD_EVERY_S)
        board.board_hook(conn, fake_clock)
        assert capsys.readouterr().err.count("[cp-board]") == 1

    def test_reset_forgets_the_logged_error(self, conn, fake_clock, monkeypatch, capsys) -> None:
        def boom(html: str, dest: Path) -> Path:
            raise OSError("full")

        monkeypatch.setattr(board, "write", boom)
        board.board_hook(conn, fake_clock)
        board._reset_for_tests()
        board.ticks._reset_for_tests()
        board.board_hook(conn, fake_clock)
        assert capsys.readouterr().err.count("[cp-board]") == 2

    def test_the_daemon_survives_a_raising_render(
        self, migrated: Path, fake_clock, fake_probe, monkeypatch, active_loop, capsys
    ) -> None:
        from control_plane import board_render, daemon, ticks

        calls: list[int] = []

        def broken(m: dict) -> str:
            calls.append(1)
            raise KeyError("renamed")

        monkeypatch.setattr(board_render, "render", broken)
        out = daemon.run(
            fake_clock,
            probe=fake_probe,
            db_path=migrated,
            tick_hooks=ticks.hooks(),
            max_ticks=2,
            tick_s=board.BOARD_EVERY_S,  # every tick is due
        )
        assert out["ticks"] == 2 and len(calls) == 2  # the next due tick runs the board again
        assert capsys.readouterr().err.count("'board' failed") == 2

    def test_cp_version_lists_it(self, invoke) -> None:
        assert invoke("version")[1]["tick_hooks"][-1] == "board"


# --------------------------------------------------------------------------- cp.py board (AC7)


class TestCommand:
    def test_no_open_prints_the_json_and_writes_the_file(
        self, migrated: Path, invoke, fake_open_url, fake_popen
    ) -> None:
        code, out = invoke("board", "--no-open")
        assert code == 0, out
        dest = board.board_path(migrated)
        assert out == {
            "ok": True,
            "path": str(dest),
            "url": dest.as_uri(),
            "daemon": out["daemon"],
            "schema": "ok",
            "opened": False,
        }
        assert out["daemon"] in ("started", "running", "restarted") and fake_popen.calls  # `ensure` ran
        assert "Nothing is waiting for you." in dest.read_text() and fake_open_url.calls == []

    def test_opens_the_file(self, migrated: Path, invoke, fake_open_url) -> None:
        fake_open_url.result = False
        code, out = invoke("board")
        assert code == 0 and fake_open_url.calls == [board.board_path(migrated).as_uri()] and out["opened"] is False
        fake_open_url.result = True
        assert invoke("board")[1]["opened"] is True

    def test_a_daemon_start_error_still_writes_the_board(self, migrated: Path, invoke, monkeypatch) -> None:
        from control_plane import cli

        def fail(io, path=None):
            raise RuntimeError("no fork")

        monkeypatch.setattr(cli, "ensure_daemon", fail)
        code, out = invoke("board", "--no-open")
        assert code == 0 and out["daemon"] == "error" and out["daemon_error"] == "RuntimeError: no fork"
        assert Path(out["path"]).is_file()

    def test_a_missing_db(self, db_path: Path, invoke) -> None:
        code, out = invoke("board", "--no-open")
        assert code == 0 and out["schema"] == "missing"
        assert board.EMPTY_LINE in Path(out["path"]).read_text() and out["path"] == str(board.board_path(db_path))

    def test_a_migration_in_progress(self, migrated: Path, invoke, fake_popen) -> None:
        from control_plane import db

        with db.file_lock(board.paths.sidecar(migrated, ".migrate.lock"), exclusive=True, timeout=0) as got:
            assert got
            code, out = invoke("board", "--no-open")
        assert code == 0 and out["schema"] == "pending" and board.EMPTY_LINE in Path(out["path"]).read_text()

    def test_an_older_db_is_pending(self, migrated: Path, invoke) -> None:
        from .conftest import CUR, sql

        sql(migrated, f"PRAGMA user_version = {CUR - 1}")
        code, out = invoke("board", "--no-open")
        assert code == 0 and out["schema"] == "pending"

    def test_a_newer_db_refuses_before_any_ensure(self, migrated: Path, invoke, fake_popen) -> None:
        from .conftest import NEXT, sql

        sql(migrated, f"PRAGMA user_version = {NEXT}")
        code, out = invoke("board", "--no-open")
        assert code == 5 and out["error"] == "SCHEMA_TOO_NEW" and fake_popen.calls == []
        assert not board.board_path(migrated).exists()

    def test_another_error_while_rendering_propagates(self, migrated: Path, invoke, monkeypatch) -> None:
        def busy(*a, **k):
            raise board.errors.CpError("BUSY", "database busy")

        monkeypatch.setattr(board, "render_to", busy)
        assert invoke("board", "--no-open") == (6, {"ok": False, "error": "BUSY", "detail": "database busy"})

    def test_a_failing_write_is_internal(self, migrated: Path, invoke, monkeypatch) -> None:
        def boom(html: str, dest: Path) -> Path:
            raise OSError("read-only file system")

        monkeypatch.setattr(board, "write", boom)
        code, out = invoke("board", "--no-open")
        assert code == 1 and out["error"] == "INTERNAL" and "board write failed" in out["detail"]
        assert out["hint"] == "check QS_CP_BOARD_DIR"

    def test_the_command_and_the_hook_write_the_same_file(self, migrated: Path, invoke, conn, fake_clock) -> None:
        board.board_hook(conn, fake_clock)
        assert invoke("board", "--no-open")[1]["path"] == str(board.board_path(migrated))

    def test_the_real_open_url(self, monkeypatch) -> None:
        import webbrowser

        from control_plane import cli

        from .conftest import REAL_MAKE_DEPS

        seen: list[str] = []
        monkeypatch.setattr(webbrowser, "open", lambda url: seen.append(url) or 1)
        assert cli._open_url("file:///x") is True and seen == ["file:///x"]
        assert REAL_MAKE_DEPS().open_url is cli._open_url
