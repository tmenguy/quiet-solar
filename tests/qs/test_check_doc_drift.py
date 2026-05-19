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
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


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
    assert captured_cmd[0][:5] == [
        "git",
        "diff",
        "--cached",
        "--diff-filter=ACMRD",
        "--name-only",
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

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "qs" / "check_doc_drift.py"

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
