"""Tests for ``scripts/qs/ci_reconcile_shards.py`` (QS-292, AC-10).

Exercises every branch of the reconcile script by enumeration against
synthetic ``tmp_path`` fixtures: happy path, wrong JUnit-file count,
missing ``total-collected.txt``, unparseable collected count,
unparseable XML, and executed-vs-collected count mismatch.

The module import relies on ``tests/qs/conftest.py``'s autouse
``_add_scripts_qs_to_syspath`` fixture, so it must happen inside each
test function (the fixture is not active at collection time).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _write_junit(path: Path, n_tests: int) -> None:
    """Write a minimal pytest-shaped JUnit XML file with ``n_tests`` cases."""
    cases = "".join(f'<testcase classname="tests.test_x" name="test_{i}" time="0.01"/>' for i in range(n_tests))
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<testsuites><testsuite name="pytest" tests="{n_tests}">'
        f"{cases}</testsuite></testsuites>",
        encoding="utf-8",
    )


def _make_shard_dir(tmp_path: Path, shard_sizes: list[int], collected: int | str | None) -> Path:
    """Populate ``tmp_path`` with JUnit shard files and a collected count."""
    for i, size in enumerate(shard_sizes, start=1):
        _write_junit(tmp_path / f"junit-shard-{i}.xml", size)
    if collected is not None:
        (tmp_path / "total-collected.txt").write_text(f"{collected}\n", encoding="utf-8")
    return tmp_path


def test_happy_path_prints_counts_and_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Four shards whose executed sum equals the collected total → exit 0."""
    import ci_reconcile_shards

    _make_shard_dir(tmp_path, [3, 2, 4, 1], collected=10)
    rc = ci_reconcile_shards.main([str(tmp_path), "--splits", "4"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "executed=10 collected=10" in out
    assert "::error::" not in out


def test_wrong_junit_file_count_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Three JUnit files against ``--splits 4`` → one ::error:: line, exit 1."""
    import ci_reconcile_shards

    _make_shard_dir(tmp_path, [3, 3, 4], collected=10)
    rc = ci_reconcile_shards.main([str(tmp_path), "--splits", "4"])
    out = capsys.readouterr().out
    assert rc == 1
    assert out.count("::error::") == 1
    assert "expected 4 JUnit shard files" in out
    assert "found 3" in out


def test_missing_total_collected_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """No ``total-collected.txt`` → one ::error:: line, exit 1."""
    import ci_reconcile_shards

    _make_shard_dir(tmp_path, [3, 2, 4, 1], collected=None)
    rc = ci_reconcile_shards.main([str(tmp_path), "--splits", "4"])
    out = capsys.readouterr().out
    assert rc == 1
    assert out.count("::error::") == 1
    assert "total-collected.txt" in out


def test_unparseable_collected_count_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Garbage in ``total-collected.txt`` → one ::error:: line, exit 1."""
    import ci_reconcile_shards

    _make_shard_dir(tmp_path, [3, 2, 4, 1], collected="not-a-number")
    rc = ci_reconcile_shards.main([str(tmp_path), "--splits", "4"])
    out = capsys.readouterr().out
    assert rc == 1
    assert out.count("::error::") == 1
    assert "unparseable collected count" in out


def test_unparseable_junit_xml_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A truncated JUnit file → one ::error:: line naming it, exit 1."""
    import ci_reconcile_shards

    _make_shard_dir(tmp_path, [3, 2, 4], collected=10)
    (tmp_path / "junit-shard-4.xml").write_text("<testsuites><testsuite", encoding="utf-8")
    rc = ci_reconcile_shards.main([str(tmp_path), "--splits", "4"])
    out = capsys.readouterr().out
    assert rc == 1
    assert out.count("::error::") == 1
    assert "unparseable JUnit XML" in out
    assert "junit-shard-4.xml" in out


def test_count_mismatch_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Executed sum != collected total → one ::error:: line, exit 1."""
    import ci_reconcile_shards

    _make_shard_dir(tmp_path, [3, 2, 4, 1], collected=11)
    rc = ci_reconcile_shards.main([str(tmp_path), "--splits", "4"])
    out = capsys.readouterr().out
    assert rc == 1
    assert out.count("::error::") == 1
    assert "executed test count 10" in out
    assert "collected 11" in out
