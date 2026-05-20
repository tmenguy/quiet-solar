"""Tests for ``scripts/qs/check_doc_drift.py``.

The drift checker exists so the doc hierarchy under ``docs/agents/``
stays anchored to the source files it claims to ``covers:``. Every
test here drives the script through an isolated ``tmp_path`` repo —
no dependency on the real ``docs/agents/`` contents (which don't
exist yet at the time these tests were written).

The 12 acceptance-criterion tests from AC-8 are present and named
verbatim. A handful of additional tests cover branches that the
named 12 don't reach (``--paths``-less mode that calls git, the
non-JSON text output paths, the "covers not a list" malformed
branch, etc.) so the script reaches 100% coverage without any
``# pragma: no cover`` markers (AC-8 requirement).

The autouse ``_add_scripts_qs_to_syspath`` fixture in
``tests/qs/conftest.py`` puts ``scripts/qs/`` on ``sys.path``, so
tests can ``import check_doc_drift`` in-process — both faster than
subprocess and visible to coverage measurement.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# Module-level constants for path math (review-fix #01 N5). Keeping
# the layout assumption in one place means a future move of this
# test file produces a clear ``script_path.is_file()`` failure
# instead of an opaque ``ModuleNotFoundError`` deep inside ``runpy``.
TESTS_QS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_QS_DIR.parents[1]
CHECK_DOC_DRIFT_PATH = REPO_ROOT / "scripts" / "qs" / "check_doc_drift.py"


# ---------------------------------------------------------------------------
# Test helpers


def _setup_repo(tmp_path: Path) -> Path:
    """Create a minimal ``repo_root`` layout under ``tmp_path``.

    The drift checker expects ``docs/agents/`` and
    ``custom_components/quiet_solar/`` to live at the repo root. We
    materialize both directories so the tests can drop docs and
    source files into them without further setup.
    """
    (tmp_path / "docs" / "agents").mkdir(parents=True)
    (tmp_path / "custom_components" / "quiet_solar").mkdir(parents=True)
    return tmp_path


def _make_source(repo: Path, rel: str) -> Path:
    """Touch a file at ``repo/rel`` so a ``covers:`` entry resolves."""
    src = repo / rel
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("# source\n", encoding="utf-8")
    return src


def _write_doc(
    repo: Path,
    slug: str,
    *,
    covers: list[str] | None,
    last_verified: str = "2026-05-19",
    kind: str = "concept",
    subdir: str = "concepts",
) -> Path:
    """Write a doc with the canonical frontmatter shape.

    ``covers=None`` omits the key entirely (mimics an index/glossary
    doc that isn't drift-tracked). ``covers=[]`` writes an empty list.
    """
    lines = [
        "---",
        f"title: Title for {slug}",
        f"slug: {slug}",
        f"kind: {kind}",
    ]
    if covers is not None:
        lines.append("covers:")
        for entry in covers:
            lines.append(f"  - {entry}")
    lines.extend([f"last_verified: {last_verified}", "---", "", f"# {slug}", "", "body", ""])
    doc_path = repo / "docs" / "agents" / subdir / f"{slug}.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines), encoding="utf-8")
    return doc_path


def _import_module() -> Any:
    """Return the freshly-imported ``check_doc_drift`` module.

    Imported per-test (under the autouse conftest fixture) so the
    syspath setup/teardown stays consistent with the other
    ``tests/qs/`` modules.
    """
    import check_doc_drift  # type: ignore[import-not-found]

    return check_doc_drift


# ---------------------------------------------------------------------------
# AC-8 — the 12 named tests


def test_clean_when_no_changes(tmp_path: Path) -> None:
    """No modified sources → clean gate, exit 0."""
    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/foo.py")
    _write_doc(repo, "foo", covers=["custom_components/quiet_solar/foo.py"])

    mod = _import_module()
    assert mod.main(["--repo-root", str(repo), "--paths"]) == 0


def test_drift_when_source_modified_alone(tmp_path: Path) -> None:
    """A covered source is modified but the doc is not → exit 1."""
    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/foo.py")
    _write_doc(repo, "foo", covers=["custom_components/quiet_solar/foo.py"])

    mod = _import_module()
    exit_code = mod.main(
        [
            "--repo-root",
            str(repo),
            "--paths",
            "custom_components/quiet_solar/foo.py",
        ]
    )
    assert exit_code == 1


def test_silent_when_doc_co_modified(tmp_path: Path) -> None:
    """Source AND doc co-modified → no drift."""
    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/foo.py")
    doc = _write_doc(repo, "foo", covers=["custom_components/quiet_solar/foo.py"])

    mod = _import_module()
    exit_code = mod.main(
        [
            "--repo-root",
            str(repo),
            "--paths",
            "custom_components/quiet_solar/foo.py",
            str(doc.relative_to(repo)),
        ]
    )
    assert exit_code == 0


def test_handles_added_source(tmp_path: Path) -> None:
    """An added source covered by a doc still flags drift unless co-mod.

    ``--diff-filter=ACMRD`` includes ``A`` (added). The drift
    detector should treat an added source identically to a modified
    one: if a doc covers it but isn't co-modified, drift.
    """
    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/new.py")
    _write_doc(repo, "new", covers=["custom_components/quiet_solar/new.py"])

    mod = _import_module()
    exit_code = mod.main(
        [
            "--repo-root",
            str(repo),
            "--paths",
            "custom_components/quiet_solar/new.py",
        ]
    )
    assert exit_code == 1


def test_handles_renamed_source_exits_2(tmp_path: Path) -> None:
    """Rename = delete + add. The doc still references the old path → exit 2."""
    repo = _setup_repo(tmp_path)
    # Old path doesn't exist any more (the rename has happened); the
    # doc's ``covers:`` is stale.
    _write_doc(
        repo, "renamed", covers=["custom_components/quiet_solar/old_path.py"]
    )

    mod = _import_module()
    assert mod.main(["--repo-root", str(repo), "--paths"]) == 2


def test_handles_deleted_source_exits_2(tmp_path: Path) -> None:
    """A doc covering a deleted source → exit 2 (missing covers path)."""
    repo = _setup_repo(tmp_path)
    _write_doc(
        repo, "deleted", covers=["custom_components/quiet_solar/deleted.py"]
    )

    mod = _import_module()
    assert mod.main(["--repo-root", str(repo), "--paths"]) == 2


def test_malformed_frontmatter_exits_2(tmp_path: Path) -> None:
    """A doc without the YAML frontmatter delimiters → exit 2."""
    repo = _setup_repo(tmp_path)
    bad = repo / "docs" / "agents" / "concepts" / "bad.md"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("# no frontmatter at all\n", encoding="utf-8")

    mod = _import_module()
    assert mod.main(["--repo-root", str(repo), "--paths"]) == 2


def test_missing_covers_path_exits_2(tmp_path: Path) -> None:
    """A ``covers:`` path that doesn't exist on disk → exit 2."""
    repo = _setup_repo(tmp_path)
    _write_doc(
        repo,
        "missing",
        covers=["custom_components/quiet_solar/never_existed.py"],
    )

    mod = _import_module()
    assert mod.main(["--repo-root", str(repo), "--paths"]) == 2


def test_paths_flag_overrides_staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--paths`` bypasses the git invocation entirely.

    The test installs a ``subprocess.run`` that would explode if it
    were called for the git path. ``--paths`` must short-circuit
    before reaching it.
    """
    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/foo.py")
    _write_doc(repo, "foo", covers=["custom_components/quiet_solar/foo.py"])

    mod = _import_module()

    def _no_git_calls(*_args: object, **_kwargs: object) -> SimpleNamespace:
        raise AssertionError("git should not have been invoked when --paths is supplied")

    monkeypatch.setattr(mod.subprocess, "run", _no_git_calls)

    # Empty --paths means "no modifications" → clean.
    assert mod.main(["--repo-root", str(repo), "--paths"]) == 0


def test_json_schema_matches(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``--json`` emits the AC-7 schema: stale_docs / missing_covers / malformed_frontmatter."""
    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/foo.py")
    _write_doc(repo, "foo", covers=["custom_components/quiet_solar/foo.py"])

    mod = _import_module()
    exit_code = mod.main(
        [
            "--repo-root",
            str(repo),
            "--paths",
            "custom_components/quiet_solar/foo.py",
            "--json",
        ]
    )
    assert exit_code == 1

    payload = json.loads(capsys.readouterr().out)
    assert set(payload.keys()) >= {"stale_docs", "missing_covers", "malformed_frontmatter"}
    assert isinstance(payload["stale_docs"], list)
    assert len(payload["stale_docs"]) == 1
    entry = payload["stale_docs"][0]
    assert set(entry.keys()) >= {"doc", "stale_sources", "last_verified"}
    assert entry["doc"].endswith("foo.md")
    assert entry["stale_sources"] == ["custom_components/quiet_solar/foo.py"]
    assert entry["last_verified"] == "2026-05-19"
    assert payload["missing_covers"] == []
    assert payload["malformed_frontmatter"] == []


def test_ignores_non_quiet_solar_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A ``covers:`` entry outside ``custom_components/quiet_solar/`` → warning only.

    The off-tree path must NOT be added to ``missing_covers`` (which
    would trigger exit 2). A warning lands on stderr instead.
    """
    repo = _setup_repo(tmp_path)
    _write_doc(repo, "off_tree", covers=["docs/workflow/foo.md"])

    mod = _import_module()
    assert mod.main(["--repo-root", str(repo), "--paths"]) == 0
    captured = capsys.readouterr()
    assert "docs/workflow/foo.md" in captured.err


def test_works_in_arbitrary_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The script is cwd-independent — only ``--repo-root`` matters."""
    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/foo.py")
    _write_doc(repo, "foo", covers=["custom_components/quiet_solar/foo.py"])

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    monkeypatch.chdir(foreign)

    mod = _import_module()
    assert mod.main(["--repo-root", str(repo), "--paths"]) == 0


# ---------------------------------------------------------------------------
# Additional tests for branches the named-12 don't reach (kept for the
# AC-8 "100% coverage, no # pragma: no cover" rule).


def test_staged_diff_default_path_invokes_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``--paths``, the script reads from ``git diff --cached``."""
    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/foo.py")
    _write_doc(repo, "foo", covers=["custom_components/quiet_solar/foo.py"])

    mod = _import_module()

    captured_cmd: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: object) -> SimpleNamespace:
        captured_cmd.append(cmd)
        return SimpleNamespace(
            returncode=0,
            stdout="custom_components/quiet_solar/foo.py\n\n",
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    exit_code = mod.main(["--repo-root", str(repo)])
    assert exit_code == 1  # source modified, doc not co-modified → drift
    assert captured_cmd, "git was not invoked"
    # Review-fix #01 S8 inserts ``-c core.quotepath=false`` before
    # the ``diff`` subcommand. Verify the canonical post-S8 layout.
    assert captured_cmd[0][:6] == [
        "git",
        "-c",
        "core.quotepath=false",
        "diff",
        "--cached",
        "--diff-filter=ACMRD",
    ]


def test_staged_diff_git_failure_handled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """If ``git diff`` fails, the script logs and treats the diff as empty."""
    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/foo.py")
    _write_doc(repo, "foo", covers=["custom_components/quiet_solar/foo.py"])

    mod = _import_module()

    def fake_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=128, stdout="", stderr="fatal: not a git repo\n")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    exit_code = mod.main(["--repo-root", str(repo)])
    assert exit_code == 0
    assert "git diff failed" in capsys.readouterr().err


def test_malformed_missing_closing_delimiter(tmp_path: Path) -> None:
    """Frontmatter with no closing ``---`` → malformed, exit 2."""
    repo = _setup_repo(tmp_path)
    bad = repo / "docs" / "agents" / "concepts" / "bad.md"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("---\ntitle: bad\nslug: bad\n# never closed\n", encoding="utf-8")

    mod = _import_module()
    assert mod.main(["--repo-root", str(repo), "--paths"]) == 2


def test_malformed_yaml_payload(tmp_path: Path) -> None:
    """YAML that can't be parsed → malformed, exit 2."""
    repo = _setup_repo(tmp_path)
    bad = repo / "docs" / "agents" / "concepts" / "yaml_bad.md"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text(
        "---\n:\n  - this is\n   - intentionally invalid yaml mapping\n---\n\nbody\n",
        encoding="utf-8",
    )

    mod = _import_module()
    assert mod.main(["--repo-root", str(repo), "--paths"]) == 2


def test_malformed_frontmatter_not_a_mapping(tmp_path: Path) -> None:
    """Frontmatter whose YAML payload is a list (not a dict) → malformed."""
    repo = _setup_repo(tmp_path)
    bad = repo / "docs" / "agents" / "concepts" / "list_fm.md"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("---\n- foo\n- bar\n---\n\nbody\n", encoding="utf-8")

    mod = _import_module()
    assert mod.main(["--repo-root", str(repo), "--paths"]) == 2


def test_malformed_covers_not_a_list(tmp_path: Path) -> None:
    """``covers: <string>`` instead of a list → malformed."""
    repo = _setup_repo(tmp_path)
    bad = repo / "docs" / "agents" / "concepts" / "bad_covers_scalar.md"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text(
        "---\ntitle: x\nslug: x\nkind: concept\ncovers: custom_components/quiet_solar/x.py\n---\n\nbody\n",
        encoding="utf-8",
    )

    mod = _import_module()
    assert mod.main(["--repo-root", str(repo), "--paths"]) == 2


def test_malformed_covers_item_not_string(tmp_path: Path) -> None:
    """A ``covers:`` list with a non-string item → malformed."""
    repo = _setup_repo(tmp_path)
    bad = repo / "docs" / "agents" / "concepts" / "bad_covers_item.md"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text(
        "---\ntitle: x\nslug: x\nkind: concept\ncovers:\n  - 123\n---\n\nbody\n",
        encoding="utf-8",
    )

    mod = _import_module()
    assert mod.main(["--repo-root", str(repo), "--paths"]) == 2


def test_doc_without_covers_is_not_tracked(tmp_path: Path) -> None:
    """A doc without a ``covers:`` key (like ``index.md``) is silently skipped."""
    repo = _setup_repo(tmp_path)
    _write_doc(repo, "index", covers=None, subdir=".")

    mod = _import_module()
    assert mod.main(["--repo-root", str(repo), "--paths"]) == 0


def test_docs_root_missing_is_clean(tmp_path: Path) -> None:
    """No ``docs/agents/`` tree → clean (nothing to check)."""
    # Don't call _setup_repo here — we want docs/agents/ to NOT exist.
    (tmp_path / "custom_components" / "quiet_solar").mkdir(parents=True)

    mod = _import_module()
    assert mod.main(["--repo-root", str(tmp_path), "--paths"]) == 0


def test_text_output_for_stale(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Non-JSON mode prints stale-doc lines to stdout."""
    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/foo.py")
    _write_doc(repo, "foo", covers=["custom_components/quiet_solar/foo.py"])

    mod = _import_module()
    exit_code = mod.main(
        [
            "--repo-root",
            str(repo),
            "--paths",
            "custom_components/quiet_solar/foo.py",
        ]
    )
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "foo.md" in out
    assert "stale" in out.lower()


def test_text_output_for_missing_covers(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Non-JSON mode prints missing-covers lines to stdout."""
    repo = _setup_repo(tmp_path)
    _write_doc(repo, "gone", covers=["custom_components/quiet_solar/gone.py"])

    mod = _import_module()
    exit_code = mod.main(["--repo-root", str(repo), "--paths"])
    assert exit_code == 2
    out = capsys.readouterr().out
    assert "gone.py" in out
    assert "gone.md" in out


def test_text_output_for_malformed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Non-JSON mode prints malformed-doc lines to stdout."""
    repo = _setup_repo(tmp_path)
    bad = repo / "docs" / "agents" / "concepts" / "bad.md"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("no frontmatter\n", encoding="utf-8")

    mod = _import_module()
    assert mod.main(["--repo-root", str(repo), "--paths"]) == 2
    out = capsys.readouterr().out
    assert "bad.md" in out
    assert "malformed" in out.lower()


def test_last_verified_non_date_string_stringified(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A quoted ``last_verified`` value (or non-date) is stringified verbatim.

    PyYAML parses ``last_verified: 2026-05-19`` as ``datetime.date``;
    we want to preserve quoted strings (e.g., ``"unknown"``) as-is.
    Exercises the ``str(value)`` fallback in
    ``_stringify_last_verified``.
    """
    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/foo.py")
    bad_lv = repo / "docs" / "agents" / "concepts" / "foo.md"
    bad_lv.parent.mkdir(parents=True, exist_ok=True)
    bad_lv.write_text(
        "---\n"
        "title: foo\n"
        "slug: foo\n"
        "kind: concept\n"
        "covers:\n"
        "  - custom_components/quiet_solar/foo.py\n"
        'last_verified: "unknown"\n'
        "---\n\nbody\n",
        encoding="utf-8",
    )

    mod = _import_module()
    exit_code = mod.main(
        [
            "--repo-root",
            str(repo),
            "--paths",
            "custom_components/quiet_solar/foo.py",
            "--json",
        ]
    )
    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["stale_docs"][0]["last_verified"] == "unknown"


def test_script_main_entry_invokes_sys_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``python check_doc_drift.py`` exits via ``sys.exit(main())``.

    Executes the script through ``runpy`` with ``run_name='__main__'``
    so the ``if __name__ == '__main__':`` guard fires under the
    coverage tracer. AC-8's "no ``# pragma: no cover``" rule means
    every line must be reached by a real test.
    """
    import runpy

    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/foo.py")
    _write_doc(repo, "foo", covers=["custom_components/quiet_solar/foo.py"])

    # Review-fix #01 N5: assert the path exists before invoking runpy
    # so a future relocation produces a clear failure here, not deep
    # inside ``runpy.run_path``.
    script_path = CHECK_DOC_DRIFT_PATH
    assert script_path.is_file(), (
        f"check_doc_drift.py not found at {script_path} — has the script moved?"
    )

    monkeypatch.setattr(
        "sys.argv",
        [str(script_path), "--repo-root", str(repo), "--paths"],
    )

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(script_path), run_name="__main__")

    assert excinfo.value.code == 0


def test_json_schema_for_missing_and_malformed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """JSON output carries both ``missing_covers`` and ``malformed_frontmatter``."""
    repo = _setup_repo(tmp_path)
    _write_doc(repo, "gone", covers=["custom_components/quiet_solar/gone.py"])
    bad = repo / "docs" / "agents" / "concepts" / "bad.md"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("no frontmatter\n", encoding="utf-8")

    mod = _import_module()
    assert mod.main(["--repo-root", str(repo), "--paths", "--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["stale_docs"] == []
    assert any("gone.py" in entry for entry in payload["missing_covers"])
    assert any("bad.md" in entry for entry in payload["malformed_frontmatter"])


# ---------------------------------------------------------------------------
# Review-fix #01 — tests for the must-fix / should-fix / nice-to-have
# findings landed in docs/stories/QS-185.story_review_fix_#01.md.


# M1: ``_git_staged_paths`` doesn't catch ``FileNotFoundError``


def test_git_not_on_path_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``FileNotFoundError`` from a missing ``git`` binary is treated as 'no git env'.

    Review-fix #01 M1: in a sandboxed CI image with no ``git`` on
    ``PATH``, ``subprocess.run(['git', ...])`` raises
    ``FileNotFoundError``. The module contract promises a clean
    empty-diff return; this test pins that contract.
    """
    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/foo.py")
    _write_doc(repo, "foo", covers=["custom_components/quiet_solar/foo.py"])

    mod = _import_module()

    def _no_git(*_args: object, **_kwargs: object) -> SimpleNamespace:
        raise FileNotFoundError("git: command not found")

    monkeypatch.setattr(mod.subprocess, "run", _no_git)

    assert mod.main(["--repo-root", str(repo)]) == 0
    assert "git" in capsys.readouterr().err.lower()


# M2: ``--paths`` doesn't normalize path forms


def test_paths_flag_accepts_absolute_paths(tmp_path: Path) -> None:
    """An absolute path inside the repo is normalized to a relative POSIX form."""
    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/foo.py")
    _write_doc(repo, "foo", covers=["custom_components/quiet_solar/foo.py"])

    mod = _import_module()
    abs_source = str(repo / "custom_components" / "quiet_solar" / "foo.py")
    assert mod.main(["--repo-root", str(repo), "--paths", abs_source]) == 1


def test_paths_flag_accepts_dot_prefix(tmp_path: Path) -> None:
    """A leading ``./`` is stripped before matching."""
    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/foo.py")
    _write_doc(repo, "foo", covers=["custom_components/quiet_solar/foo.py"])

    mod = _import_module()
    assert (
        mod.main(
            [
                "--repo-root",
                str(repo),
                "--paths",
                "./custom_components/quiet_solar/foo.py",
            ]
        )
        == 1
    )


def test_paths_flag_accepts_backslashes(tmp_path: Path) -> None:
    """Windows-style backslashes are converted to forward slashes."""
    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/foo.py")
    _write_doc(repo, "foo", covers=["custom_components/quiet_solar/foo.py"])

    mod = _import_module()
    assert (
        mod.main(
            [
                "--repo-root",
                str(repo),
                "--paths",
                "custom_components\\quiet_solar\\foo.py",
            ]
        )
        == 1
    )


def test_paths_flag_strips_trailing_slash(tmp_path: Path) -> None:
    """Trailing-slash variants normalize the same way as the canonical form."""
    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/foo.py")
    _write_doc(repo, "foo", covers=["custom_components/quiet_solar/foo.py"])

    mod = _import_module()
    # No trailing slash on a *file* is impossible to construct
    # naturally, but the normalizer should still tolerate a stray
    # one. Exercise the strip branch explicitly.
    assert (
        mod.main(
            [
                "--repo-root",
                str(repo),
                "--paths",
                "custom_components/quiet_solar/foo.py/",
            ]
        )
        == 1
    )


def test_paths_flag_absolute_outside_repo_kept_as_is(tmp_path: Path) -> None:
    """An absolute path outside ``repo_root`` keeps its absolute form.

    The normalizer can't relativize it; the entry is then simply
    irrelevant to drift detection (no source matches), so the gate
    stays clean.
    """
    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/foo.py")
    _write_doc(repo, "foo", covers=["custom_components/quiet_solar/foo.py"])

    # Use ``/etc/hostname`` as a stable absolute path that won't be
    # under ``repo``. On systems where it doesn't exist, the test is
    # still valid because we only care about path-string handling.
    outside = "/etc/hostname"

    mod = _import_module()
    assert mod.main(["--repo-root", str(repo), "--paths", outside]) == 0


# S4: ``subprocess.run`` lacks ``timeout``


def test_git_timeout_handled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A hung ``git diff`` (index lock contention) is bounded by ``timeout=``."""
    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/foo.py")
    _write_doc(repo, "foo", covers=["custom_components/quiet_solar/foo.py"])

    mod = _import_module()

    def _timeout(*_args: object, **_kwargs: object) -> SimpleNamespace:
        raise subprocess.TimeoutExpired(cmd=["git"], timeout=30)

    monkeypatch.setattr(mod.subprocess, "run", _timeout)

    assert mod.main(["--repo-root", str(repo)]) == 0
    assert "timeout" in capsys.readouterr().err.lower()


def test_git_invocation_passes_timeout_and_check_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_git_staged_paths`` invokes ``subprocess.run`` with safety kwargs."""
    repo = _setup_repo(tmp_path)

    mod = _import_module()

    captured_kwargs: dict[str, Any] = {}

    def _capture(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        captured_kwargs.update(kwargs)
        captured_kwargs["__cmd__"] = cmd
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _capture)

    mod.main(["--repo-root", str(repo)])

    assert captured_kwargs.get("check") is False, (
        "subprocess.run must pass check=False explicitly (review-fix #01 S4)"
    )
    assert "timeout" in captured_kwargs and captured_kwargs["timeout"] >= 1, (
        "subprocess.run must pass a finite timeout (review-fix #01 S4)"
    )


# S5: closing-delimiter / line-ending robustness


def test_frontmatter_without_trailing_newline_is_valid(tmp_path: Path) -> None:
    """A doc that ends with ``---`` (no trailing newline) is accepted."""
    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/foo.py")
    doc_path = repo / "docs" / "agents" / "concepts" / "foo.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(
        "---\n"
        "title: foo\n"
        "slug: foo\n"
        "kind: concept\n"
        "covers:\n"
        "  - custom_components/quiet_solar/foo.py\n"
        "last_verified: 2026-05-19\n"
        "---",  # no trailing newline; no body
        encoding="utf-8",
    )

    mod = _import_module()
    assert mod.main(["--repo-root", str(repo), "--paths"]) == 0


def test_frontmatter_with_bare_cr_lineendings_is_normalized(tmp_path: Path) -> None:
    """Bare ``\\r`` line endings (legacy macOS / corrupted files) parse correctly."""
    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/foo.py")
    doc_path = repo / "docs" / "agents" / "concepts" / "foo.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_bytes(
        b"---\r"
        b"title: foo\r"
        b"slug: foo\r"
        b"kind: concept\r"
        b"covers:\r"
        b"  - custom_components/quiet_solar/foo.py\r"
        b"last_verified: 2026-05-19\r"
        b"---\r"
        b"\r"
        b"body\r"
    )

    mod = _import_module()
    assert mod.main(["--repo-root", str(repo), "--paths"]) == 0


# S6: UTF-8 BOM at start of doc


def test_doc_with_utf8_bom_parses_correctly(tmp_path: Path) -> None:
    """A doc saved with a UTF-8 BOM (Windows editors) is not flagged malformed."""
    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/foo.py")
    doc_path = repo / "docs" / "agents" / "concepts" / "foo.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_bytes(
        b"\xef\xbb\xbf"  # UTF-8 BOM
        b"---\n"
        b"title: foo\n"
        b"slug: foo\n"
        b"kind: concept\n"
        b"covers:\n"
        b"  - custom_components/quiet_solar/foo.py\n"
        b"last_verified: 2026-05-19\n"
        b"---\n"
        b"\n"
        b"body\n"
    )

    mod = _import_module()
    assert mod.main(["--repo-root", str(repo), "--paths"]) == 0


# S7: duplicate ``covers:`` entries


def test_duplicate_covers_entries_dedup_in_json_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A doc with the same source in ``covers:`` twice de-dups in the JSON report."""
    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/foo.py")
    doc_path = repo / "docs" / "agents" / "concepts" / "foo.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(
        "---\n"
        "title: foo\n"
        "slug: foo\n"
        "kind: concept\n"
        "covers:\n"
        "  - custom_components/quiet_solar/foo.py\n"
        "  - custom_components/quiet_solar/foo.py\n"
        "last_verified: 2026-05-19\n"
        "---\n",
        encoding="utf-8",
    )

    mod = _import_module()
    exit_code = mod.main(
        [
            "--repo-root",
            str(repo),
            "--paths",
            "custom_components/quiet_solar/foo.py",
            "--json",
        ]
    )
    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    sources = payload["stale_docs"][0]["stale_sources"]
    assert sources == ["custom_components/quiet_solar/foo.py"], (
        f"Duplicate covers entries should de-dup in the JSON output; got {sources!r}"
    )


# S8: git path quoting / non-ASCII filenames


def test_git_invocation_disables_quotepath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_git_staged_paths`` passes ``-c core.quotepath=false`` to git.

    Without this flag, git's default ``core.quotepath=true`` returns
    non-ASCII paths in C-style escapes, which never match
    ``source_to_docs`` keys (review-fix #01 S8).
    """
    repo = _setup_repo(tmp_path)

    mod = _import_module()
    captured_cmd: list[str] = []

    def _capture(cmd: list[str], **_kwargs: Any) -> SimpleNamespace:
        captured_cmd.extend(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _capture)
    mod.main(["--repo-root", str(repo)])

    cmd_str = " ".join(captured_cmd)
    assert "core.quotepath=false" in cmd_str, (
        f"git invocation must disable quotepath (review-fix #01 S8); got: {cmd_str}"
    )


# N7: leading whitespace on covers entry


def test_covers_entry_with_stray_whitespace_is_normalized(tmp_path: Path) -> None:
    """``covers:`` entries with stray leading/trailing whitespace still resolve.

    Review-fix #01 N7: a YAML quoted-string indent can produce
    ``"  custom_components/quiet_solar/foo.py"``, which previously
    failed the ``startswith(TRACKED_PREFIX)`` check and was treated
    as off-tree (warning only).
    """
    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/foo.py")
    doc_path = repo / "docs" / "agents" / "concepts" / "foo.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(
        "---\n"
        "title: foo\n"
        "slug: foo\n"
        "kind: concept\n"
        "covers:\n"
        '  - "  custom_components/quiet_solar/foo.py  "\n'
        "last_verified: 2026-05-19\n"
        "---\n",
        encoding="utf-8",
    )

    mod = _import_module()
    # The normalized entry resolves under ``custom_components/quiet_solar/``,
    # so modifying that source flags drift on the doc.
    exit_code = mod.main(
        [
            "--repo-root",
            str(repo),
            "--paths",
            "custom_components/quiet_solar/foo.py",
        ]
    )
    assert exit_code == 1


# N8: directory ``covers:`` entries


def test_covers_entry_pointing_to_directory_is_malformed(tmp_path: Path) -> None:
    """A ``covers:`` entry pointing to a directory is flagged malformed.

    Directory entries don't cascade to their contents, so accepting
    them silently misses every drift inside that directory. Explicit
    rejection is the safer contract.
    """
    repo = _setup_repo(tmp_path)
    # Create a directory inside the tracked tree.
    (repo / "custom_components" / "quiet_solar" / "home_model").mkdir(parents=True, exist_ok=True)
    _write_doc(
        repo, "dir_covers", covers=["custom_components/quiet_solar/home_model"]
    )

    mod = _import_module()
    assert mod.main(["--repo-root", str(repo), "--paths"]) == 2


# N9: rglob follows symlinks


def test_rglob_skips_symlinks(tmp_path: Path) -> None:
    """Symlinked docs inside ``docs/agents/`` are skipped to avoid cycles.

    Review-fix #01 N9: ``Path.rglob('*.md')`` follows symlinks by
    default. A self-referential symlink would hang the scan; a
    symlink pointing into another tree would duplicate reports.
    """
    if not hasattr(os, "symlink"):  # pragma: no cover — platform guard
        pytest.skip("symlinks unavailable on this platform")

    repo = _setup_repo(tmp_path)
    _make_source(repo, "custom_components/quiet_solar/foo.py")
    real_doc = _write_doc(repo, "foo", covers=["custom_components/quiet_solar/foo.py"])

    symlink_target = repo / "docs" / "agents" / "concepts" / "symlinked.md"
    try:
        os.symlink(real_doc, symlink_target)
    except (OSError, NotImplementedError):  # pragma: no cover — Windows without admin
        pytest.skip("symlink creation requires elevated privileges on this platform")

    mod = _import_module()
    # The symlinked doc shouldn't double-report drift or cause a cycle.
    assert (
        mod.main(
            [
                "--repo-root",
                str(repo),
                "--paths",
                "custom_components/quiet_solar/foo.py",
            ]
        )
        == 1
    )


# N10: misleading "NoneType" message when ``covers:`` is null


def test_null_covers_value_emits_clear_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``covers:`` with no value yields a help-tier message, not 'NoneType'."""
    repo = _setup_repo(tmp_path)
    doc_path = repo / "docs" / "agents" / "concepts" / "nullc.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(
        "---\n"
        "title: nullc\n"
        "slug: nullc\n"
        "kind: concept\n"
        "covers:\n"  # no value → parses to None
        "last_verified: 2026-05-19\n"
        "---\n",
        encoding="utf-8",
    )

    mod = _import_module()
    assert mod.main(["--repo-root", str(repo), "--paths"]) == 2
    err = capsys.readouterr().err
    # The legacy message would say "NoneType"; the new message names
    # the actionable fix.
    assert "covers" in err.lower()
    assert "omit the key" in err or "covers: []" in err


# N11: ``--repo-root`` validation


def test_nonexistent_repo_root_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-existent ``--repo-root`` exits with a clear non-zero error code."""
    missing = tmp_path / "does_not_exist"
    assert not missing.exists()

    mod = _import_module()
    exit_code = mod.main(["--repo-root", str(missing), "--paths"])
    assert exit_code != 0
    err = capsys.readouterr().err
    assert "repo-root" in err.lower() or str(missing) in err


# ---------------------------------------------------------------------------
# Cross-harness agent sync tests (QS-193 AC-3 / T6)


def _write_agent(
    repo: Path,
    harness: str,
    name: str,
    *,
    frontmatter: str = "",
    body: str = "# Agent body\n\nShared content.\n",
) -> Path:
    """Write an agent file under ``.<harness>/agents/<name>.md``.

    ``frontmatter`` is the YAML content between the ``---`` delimiters
    (without the delimiters themselves). ``body`` is everything after
    the closing ``---``.
    """
    agent_dir = repo / f".{harness}" / "agents"
    agent_dir.mkdir(parents=True, exist_ok=True)
    path = agent_dir / f"{name}.md"
    fm = frontmatter or f"name: {name}\ndescription: test agent"
    path.write_text(f"---\n{fm}\n---\n\n{body}", encoding="utf-8")
    return path


def test_harness_sync_detects_missing_counterpart(tmp_path: Path) -> None:
    """Agent file in .claude/agents/ modified, .cursor/agents/ counterpart not in modified set → drift."""
    repo = _setup_repo(tmp_path)
    _write_agent(repo, "claude", "qs-test-agent")
    _write_agent(repo, "cursor", "qs-test-agent")
    _write_agent(repo, "opencode", "qs-test-agent")

    mod = _import_module()
    exit_code = mod.main(
        [
            "--repo-root",
            str(repo),
            "--paths",
            ".claude/agents/qs-test-agent.md",
        ]
    )
    assert exit_code == 1


def test_harness_sync_allows_body_divergence(
    tmp_path: Path,
) -> None:
    """Bodies may differ — co-modification is sufficient.

    Co-modification check only cares that all three were modified, not
    that their bodies are identical. Bodies legitimately differ across
    harnesses (harness-specific session spawn, handoff instructions).
    """
    repo = _setup_repo(tmp_path)
    _write_agent(repo, "claude", "qs-test-agent", body="# Body A\n")
    _write_agent(repo, "cursor", "qs-test-agent", body="# Body B\n")
    _write_agent(repo, "opencode", "qs-test-agent", body="# Body A\n")

    mod = _import_module()
    exit_code = mod.main(
        [
            "--repo-root",
            str(repo),
            "--paths",
            ".claude/agents/qs-test-agent.md",
            ".cursor/agents/qs-test-agent.md",
            ".opencode/agents/qs-test-agent.md",
        ]
    )
    assert exit_code == 0


def test_harness_sync_passes_when_bodies_match(tmp_path: Path) -> None:
    """All three modified and bodies match (frontmatter differs) → no drift."""
    repo = _setup_repo(tmp_path)
    body = "# Shared body\n\nIdentical content.\n"
    _write_agent(repo, "claude", "qs-test-agent", frontmatter="name: qs-test-agent\ntools: Bash", body=body)
    _write_agent(repo, "cursor", "qs-test-agent", frontmatter="name: qs-test-agent\nmodel: inherit", body=body)
    _write_agent(repo, "opencode", "qs-test-agent", frontmatter="description: test\nmode: primary", body=body)

    mod = _import_module()
    exit_code = mod.main(
        [
            "--repo-root",
            str(repo),
            "--paths",
            ".claude/agents/qs-test-agent.md",
            ".cursor/agents/qs-test-agent.md",
            ".opencode/agents/qs-test-agent.md",
        ]
    )
    assert exit_code == 0


def test_harness_sync_ignores_non_agent_files(tmp_path: Path) -> None:
    """Modified file in .claude/ but not under agents/ → no harness sync check."""
    repo = _setup_repo(tmp_path)
    # Create a non-agent file in .claude/
    claude_dir = repo / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    (claude_dir / "settings.json").write_text("{}", encoding="utf-8")

    mod = _import_module()
    exit_code = mod.main(
        [
            "--repo-root",
            str(repo),
            "--paths",
            ".claude/settings.json",
        ]
    )
    assert exit_code == 0


def test_harness_sync_handles_missing_harness_dir(tmp_path: Path) -> None:
    """Agents dir exists in other harnesses but file is absent → no drift.

    Harness-specific agents (present in only one harness) are exempt
    from the co-modification check. Only counterpart files that
    actually exist on disk trigger drift.
    """
    repo = _setup_repo(tmp_path)
    _write_agent(repo, "claude", "qs-test-agent")
    # Create the agents dirs (harness is set up) but don't create the files
    (repo / ".cursor" / "agents").mkdir(parents=True, exist_ok=True)
    (repo / ".opencode" / "agents").mkdir(parents=True, exist_ok=True)

    mod = _import_module()
    exit_code = mod.main(
        [
            "--repo-root",
            str(repo),
            "--paths",
            ".claude/agents/qs-test-agent.md",
        ]
    )
    assert exit_code == 0
