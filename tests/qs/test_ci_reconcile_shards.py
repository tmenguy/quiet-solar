"""Tests for ``scripts/qs/ci_reconcile_shards.py`` (QS-292, AC-10).

Exercises every branch of the reconcile script by enumeration against
synthetic ``tmp_path`` fixtures: happy path, wrong JUnit-file count,
unexpected artifact name, bad shard-index set, missing
``total-collected.txt``, unparseable/unreadable collected count,
unparseable XML, duplicate test cases across shards, and an
executed-vs-collected count mismatch.

The module import relies on ``tests/qs/conftest.py``'s autouse
``_add_scripts_qs_to_syspath`` fixture, so it must happen inside each
test function (the fixture is not active at collection time).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _write_junit(path: Path, cases: list[tuple[str, str]]) -> None:
    """Write a minimal pytest-shaped JUnit XML file for ``cases``.

    ``cases`` is a list of ``(classname, name)`` pairs, so a test can
    control cross-shard uniqueness explicitly.
    """
    body = "".join(f'<testcase classname="{classname}" name="{name}" time="0.01"/>' for classname, name in cases)
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<testsuites><testsuite name="pytest" tests="{len(cases)}">'
        f"{body}</testsuite></testsuites>",
        encoding="utf-8",
    )


def _make_shard_dir(
    tmp_path: Path,
    shard_sizes: list[int],
    collected: int | str | None,
    indices: list[int] | None = None,
) -> Path:
    """Populate ``tmp_path`` with JUnit shard files and a collected count.

    Each shard gets its own ``classname``, so the synthetic
    ``(classname, name)`` pairs are unique across shards unless a test
    deliberately arranges a collision.
    """
    shard_indices = indices if indices is not None else list(range(1, len(shard_sizes) + 1))
    for idx, size in zip(shard_indices, shard_sizes, strict=True):
        cases = [(f"tests.test_shard{idx}", f"test_{i}") for i in range(size)]
        _write_junit(tmp_path / f"junit-shard-{idx}.xml", cases)
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


def test_missing_artifacts_error_names_the_remedy(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Zero artifacts (expired) → the error tells the user to re-run ALL jobs.

    Review fix #01 S5: a delayed "Re-run failed jobs" on the aggregation
    job alone downloads nothing once the shard artifacts have expired.
    It fails safe, but the message must say how to clear it.
    """
    import ci_reconcile_shards

    (tmp_path / "total-collected.txt").write_text("10\n", encoding="utf-8")
    rc = ci_reconcile_shards.main([str(tmp_path), "--splits", "4"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "found 0" in out
    assert "expired" in out
    assert "re-run ALL jobs" in out


def test_unexpected_artifact_name_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A non-numeric shard suffix → one ::error:: line naming it, exit 1."""
    import ci_reconcile_shards

    _make_shard_dir(tmp_path, [3, 2, 4], collected=10)
    _write_junit(tmp_path / "junit-shard-oops.xml", [("tests.x", "test_a")])
    rc = ci_reconcile_shards.main([str(tmp_path), "--splits", "4"])
    out = capsys.readouterr().out
    assert rc == 1
    assert out.count("::error::") == 1
    assert "junit-shard-oops.xml" in out


def test_shard_indices_must_be_one_through_n(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Right file COUNT but wrong index set → exit 1.

    Review fix #01 S2(a): a missing shard 3 plus a stray shard 99 passes
    a count-only check. (Two files with the SAME index cannot coexist in
    one directory, so the index-set equality is what covers that class.)
    """
    import ci_reconcile_shards

    _make_shard_dir(tmp_path, [3, 2, 4, 1], collected=10, indices=[1, 2, 4, 99])
    rc = ci_reconcile_shards.main([str(tmp_path), "--splits", "4"])
    out = capsys.readouterr().out
    assert rc == 1
    assert out.count("::error::") == 1
    assert "shard indices" in out
    assert "99" in out


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
    assert "unreadable collected count" in out


def test_undecodable_collected_count_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Undecodable bytes → the same single ::error:: line, not a traceback.

    Review fix #01 N7: ``read_text`` can raise ``OSError`` /
    ``UnicodeDecodeError`` as well as ``ValueError`` from ``int()``; the
    module docstring promises one ``::error::`` line for every failure.
    """
    import ci_reconcile_shards

    _make_shard_dir(tmp_path, [3, 2, 4, 1], collected=None)
    (tmp_path / "total-collected.txt").write_bytes(b"\xff\xfe\x00garbage")
    rc = ci_reconcile_shards.main([str(tmp_path), "--splits", "4"])
    out = capsys.readouterr().out
    assert rc == 1
    assert out.count("::error::") == 1
    assert "unreadable collected count" in out


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


def test_offsetting_duplicate_and_drop_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A test run twice while another never runs → exit 1.

    Review fix #01 S2(b): the counts still BALANCE here (executed ==
    collected == 10), so a count-only guard exits 0 while the split is
    not a partition. Identity, not arithmetic, is what catches it.
    """
    import ci_reconcile_shards

    _make_shard_dir(tmp_path, [3, 2, 4, 1], collected=10)
    # Shard 2 re-runs one of shard 1's cases instead of its own second
    # case: the total is unchanged, but one pair is now duplicated and
    # another was never executed anywhere.
    _write_junit(
        tmp_path / "junit-shard-2.xml",
        [("tests.test_shard1", "test_0"), ("tests.test_shard2", "test_0")],
    )
    rc = ci_reconcile_shards.main([str(tmp_path), "--splits", "4"])
    out = capsys.readouterr().out
    assert rc == 1
    assert out.count("::error::") == 1
    assert "duplicate" in out
    assert "not a partition" in out


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
