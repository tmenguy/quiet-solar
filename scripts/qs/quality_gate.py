#!/usr/bin/env python3
"""Run all quality gates and report results.

Usage:
    python scripts/qs/quality_gate.py [--fix] [--json] [--cache] [--no-cache] [--full]
    python scripts/qs/quality_gate.py --quick PATH [PATH ...]
    python scripts/qs/quality_gate.py --impacted
    python scripts/qs/quality_gate.py --seed-testmon

Options:
    --fix                    Auto-fix what can be fixed (ruff format, ruff check --fix)
    --json                   Output JSON instead of human-readable text
    --cache                  Enable caching — skip gates if git state matches a previous pass
    --no-cache               Force fresh run even when --cache is present
    --full                   Force full gate run even if only dev/test files changed
    --quick PATH [PATH ...]  Fast iteration: run only the cited test paths (files
                             or directories) with xdist + sysmon; skip ruff/mypy/
                             translations/coverage/cache/scope. Mutex with
                             --cache/--no-cache/--full/--fix.
    --impacted               Inner-loop gate (QS-276): run only the testmon-selected
                             tests under --cov=<package>, write coverage.xml, then
                             diff-cover --fail-under=100 against the resolved diff
                             base. Guarantees the lines YOU changed are 100% covered
                             (the whole-repo 100% gate stays authoritative in CI).
                             Mutex with --quick/--cache/--no-cache/--full/--fix.
    --seed-testmon           Sanctioned non-gate subcommand: refresh .testmondata via
                             `pytest --testmon` (no coverage, no pass/fail verdict).
                             Used by finish-task to rebuild the main baseline.

Smart scope detection:
    When only dev-infrastructure files are modified (tests/, legacy/, docs/,
    scripts/, *.md, .claude/, .cursor/, .opencode/), the quality gate skips the
    full suite (ruff, mypy, translations, full pytest+coverage) and only runs
    the modified test files. Use --full to override.

Exit codes:
    0 = all gates pass
    1 = one or more gates failed (incl. --impacted changed-lines <100%)
    2 = argument parsing error (e.g., --quick with no paths, or mutex violation)
    3 = required tooling missing (--impacted: testmon / diff-cover not importable)
    4 = --impacted could not resolve a diff base in CI
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Resolve paths relative to repo root
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "custom_components" / "quiet_solar"
TESTS_DIR = REPO_ROOT / "tests"
VENV_BIN = REPO_ROOT / "venv" / "bin"
STRINGS_JSON = SRC_DIR / "strings.json"
TRANSLATIONS_EN = SRC_DIR / "translations" / "en.json"


VENV_PYTHON = str(VENV_BIN / "python")

# Patterns for files that are "dev-only" (no production code impact).
#
# Known gap (QS-184 review-fix #01 N8): a ``git mv
# custom_components/quiet_solar/foo.py legacy/foo.py`` change registers
# as dev-only (the post-move path starts with ``legacy/``), even though
# it has removed production code from coverage. The implement-task
# agents always run the full gate (their scope detection routes
# through the production-code path), so the gap only matters when an
# implement-setup-task agent performs a mixed-source ``git mv``. The
# implement-setup-task agent's hard rules limit it to dev paths, so
# this is documented as an accepted edge case rather than fixed by a
# rename-aware detector.
_DEV_ONLY_PATTERNS = (
    "tests/",
    "legacy/",
    "scripts/",
    "docs/",
    ".claude/",
    ".cursor/",
    ".opencode/",
    ".github/",
)
_DEV_ONLY_EXTENSIONS = (".md",)

# QS-208: UI fast-path classification.
#
# The canonical UI test exercises every J2 template and every JS card via
# regex assertions on the file contents — it is the right one-stop pytest
# target when only UI assets changed.
#
# Kept as a tuple (not a string) so the ui-only branch can build
# `sorted({*_UI_FAST_PATH_TESTS, *changed_test_files})` uniformly whether
# the list grows to 2+ files or stays at one. Adding a second UI test
# file is a one-line PR.
#
# tests/test_ui_dashboard.py is intentionally NOT here — it covers
# ui/dashboard.py (Python module). If dashboard.py is in the diff,
# scope becomes "full" and test_ui_dashboard.py runs via the full
# pytest. If only .j2 / resources/ changed, that file is not needed.
_UI_FAST_PATH_TESTS: tuple[str, ...] = ("tests/test_dashboard_rendering.py",)

_UI_PREFIX = "custom_components/quiet_solar/ui/"
_UI_RESOURCES_PREFIX = "custom_components/quiet_solar/ui/resources/"


def _venv_tool(name: str) -> str:
    """Return the absolute path to a tool inside the venv."""
    return str(VENV_BIN / name)


def _run(cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or str(REPO_ROOT))


def _emit(name: str, line: str) -> None:
    """Write a gate-prefixed line to stderr.

    Used by every gate so its log output is line-prefixed exactly once.
    """
    print(f"[{name}] {line.rstrip()}", file=sys.stderr)


_HAS_XDIST_CACHE: bool | None = None

# The probe runs inside VENV_PYTHON's interpreter so we check the same
# site-packages pytest will use, not the orchestrator's. Using a triple-quoted
# `-c` body keeps the cmd line shell-safe (no shell parsing involved — we pass
# argv directly to subprocess).
_XDIST_PROBE = "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('xdist') else 1)"


def _has_xdist() -> bool:
    """Return True when `pytest-xdist` is importable from `VENV_PYTHON`'s site-packages.

    Probes the venv interpreter via subprocess rather than the orchestrator
    process. The orchestrator may be a different Python (system, pyenv shim,
    Claude harness runtime, etc.) than the venv that pytest will actually be
    launched against — an in-process `find_spec` would lie in both directions.

    Cached after the first call via `_HAS_XDIST_CACHE` to avoid repeated probes.
    """
    global _HAS_XDIST_CACHE  # noqa: PLW0603 — module-level cache
    if _HAS_XDIST_CACHE is None:
        result = _run([VENV_PYTHON, "-c", _XDIST_PROBE])
        _HAS_XDIST_CACHE = result.returncode == 0
    return _HAS_XDIST_CACHE


def _pytest_workers() -> str | None:
    """Return the worker count for `pytest -n`, or None for serial mode.

    Returns None when:
    - `pytest-xdist` is not available in the venv, or
    - `QS_QG_PYTEST_WORKERS` is set to `"0"` or all-whitespace/empty
      (explicit serial mode).

    Defaults to `"auto"` when xdist is present and no override is set.

    The raw env value is normalized: leading/trailing whitespace is stripped,
    case-insensitive `"auto"` is accepted, and positive integers are returned
    as their canonical decimal string. Invalid values (negative numbers,
    non-numeric text) emit a one-line stderr warning and fall back to `"auto"`
    rather than silently breaking pytest's argparse with an unfriendly error.
    """
    if not _has_xdist():
        return None
    raw = os.environ.get("QS_QG_PYTEST_WORKERS", "auto")
    val = raw.strip()
    if val == "" or val == "0":
        return None
    if val.lower() == "auto":
        return "auto"
    try:
        parsed = int(val)
    except ValueError:
        sys.stderr.write(f'[pytest] invalid QS_QG_PYTEST_WORKERS={raw!r}, falling back to "auto"\n')
        sys.stderr.flush()
        return "auto"
    if parsed > 0:
        return str(parsed)
    sys.stderr.write(f'[pytest] invalid QS_QG_PYTEST_WORKERS={raw!r}, falling back to "auto"\n')
    sys.stderr.flush()
    return "auto"


def _pytest_env() -> dict[str, str]:
    """Return the env mapping for the main pytest subprocess.

    Adds `COVERAGE_CORE=sysmon` on top of the parent env so coverage uses
    the faster `sys.monitoring` core on Python 3.12+.
    """
    return {**os.environ, "COVERAGE_CORE": "sysmon"}


CACHE_FILE = REPO_ROOT / ".quality_gate_cache"


def _get_git_state() -> tuple[str, str, bool]:
    """Return (branch_name, commit_hash, is_clean) from git."""
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    commit = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    status = _run(["git", "status", "--porcelain", "-uno"]).stdout.strip()
    is_clean = len(status) == 0
    return branch, commit, is_clean


def _get_changed_files() -> list[str]:
    """Return list of files changed vs origin/main (staged + unstaged + committed)."""
    # Files changed in commits not yet on origin/main
    r1 = _run(["git", "diff", "--name-only", "origin/main...HEAD"])
    # Uncommitted changes (staged + unstaged)
    r2 = _run(["git", "diff", "--name-only", "HEAD"])
    # Staged but not committed
    r3 = _run(["git", "diff", "--name-only", "--cached"])

    files: set[str] = set()
    for r in (r1, r2, r3):
        if r.returncode == 0:
            files.update(line.strip() for line in r.stdout.strip().split("\n") if line.strip())
    return sorted(files)


def _is_dev_only(filepath: str) -> bool:
    """Check if a file path is dev-only (not production source code)."""
    for pat in _DEV_ONLY_PATTERNS:
        if filepath.startswith(pat):
            return True
    for ext in _DEV_ONLY_EXTENSIONS:
        if filepath.endswith(ext):
            return True
    # Top-level config files
    basename = filepath.split("/")[-1]
    if basename in (
        "CLAUDE.md",
        "AGENTS.md",
        ".cursorrules",
        ".gitignore",
        "pyproject.toml",
        "setup.cfg",
        "requirements.txt",
        "requirements_test.txt",
    ):
        return True
    return False


def _is_ui_asset(filepath: str) -> bool:
    """Classify a repo-root-relative POSIX path (as produced by
    ``git diff --name-only``) as a non-Python UI asset.

    True iff the path is under ``custom_components/quiet_solar/ui/`` AND
    either ends with ``.j2`` (top-level or nested) OR sits under
    ``resources/`` (any extension, any depth).

    Crucially returns False for any ``.py`` file directly under ``ui/``
    — including ``ui/dashboard.py`` and ``ui/__init__.py`` — so Python
    edits route to the full gate. Convention: nothing under
    ``ui/resources/`` is Python; if a ``.py`` ever lands there, that's a
    category error and should be moved out of ``resources/`` rather
    than handled by a narrower extension allowlist here.

    Examples:
        ui/quiet_solar_dashboard_template.yaml.j2  → True
        ui/subdir/partial.j2                       → True  (nested .j2)
        ui/resources/qs-car-card.js                → True
        ui/resources/sub/foo.css                   → True
        ui/dashboard.py                            → False (Python)
        ui/__init__.py                             → False (Python)
        ui/something.py                            → False (Python)
        home_model/load.py                         → False (not under ui/)
    """
    if not filepath.startswith(_UI_PREFIX):
        return False
    if filepath.startswith(_UI_RESOURCES_PREFIX):
        return True
    return filepath.endswith(".j2")


def _detect_scope(changed_files: list[str]) -> dict:
    """Determine which gates to run based on changed files.

    Returns dict with:
        scope: "full" | "dev-only" | "ui-only"
        changed_test_files: list of test files that changed (dev-only / ui-only)
        reason: human-readable explanation
    """
    if not changed_files:
        return {"scope": "full", "changed_test_files": [], "reason": "no changes detected, running full"}

    non_dev_non_ui = [f for f in changed_files if not _is_dev_only(f) and not _is_ui_asset(f)]
    if non_dev_non_ui:
        return {
            "scope": "full",
            "changed_test_files": [],
            "reason": f"production files changed: {', '.join(non_dev_non_ui[:5])}",
        }

    test_files = [f for f in changed_files if f.startswith("tests/") and f.endswith(".py")]
    ui_assets = [f for f in changed_files if _is_ui_asset(f)]
    if ui_assets:
        return {
            "scope": "ui-only",
            "changed_test_files": test_files,
            "reason": (
                f"only UI assets and dev files changed "
                f"({len(ui_assets)} UI asset(s), {len(changed_files)} total)"
            ),
        }

    return {
        "scope": "dev-only",
        "changed_test_files": test_files,
        "reason": f"only dev/test files changed ({len(changed_files)} files)",
    }


def _read_cache() -> dict | None:
    """Read and validate the cache file. Return None if missing or invalid."""
    if not CACHE_FILE.exists():
        return None
    try:
        data = json.loads(CACHE_FILE.read_text())
    except json.JSONDecodeError, OSError:
        return None
    if not isinstance(data, dict) or "branch" not in data or "commit" not in data:
        return None
    return data


def _write_cache(branch: str, commit: str, results: list[dict]) -> None:
    """Write cache file with current git state and gate results."""
    data = {
        "branch": branch,
        "commit": commit,
        "all_passed": True,
        "results": results,
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }
    CACHE_FILE.write_text(json.dumps(data, indent=2))


def _is_cache_valid(cache: dict | None, branch: str, commit: str, is_clean: bool) -> bool:
    """Check if cached results match the current git state."""
    if cache is None:
        return False
    if not is_clean:
        return False
    return cache.get("branch") == branch and cache.get("commit") == commit


_PROGRESS_CHARS = ".FEsxX"
_XDIST_PREFIX_RE = re.compile(r"^\[gw\d+\]\s*")
_PERCENT_SUFFIX_RE = re.compile(r"\s*\[\s*\d+%\]\s*$")
# S2: track xfailed/xpassed/skipped as separate keys; recognize the
# `error`/`errors` singular/plural collapse on the parser side.
_COUNT_RE = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed)")
# S3: anchor summary parsing to pytest's canonical timing suffix `in <N.NNN>s`
# to avoid misclassifying noise lines like `"5 passed checks remaining"` as
# the summary. The token must appear after the counts on the same line.
_SUMMARY_TIMING_RE = re.compile(r"\bin [\d.]+\s*s\b")

# Counts dict shape used across parser, mid-run tally, and final summary.
_COUNT_KEYS: tuple[str, ...] = ("passed", "failed", "errors", "skipped", "xfailed", "xpassed")


def _new_count_dict() -> dict[str, int]:
    """Return a fresh zero-filled counts dict matching `_COUNT_KEYS`."""
    return {k: 0 for k in _COUNT_KEYS}


def _clean_pytest_line(line: str) -> str:
    """Strip xdist worker prefix and trailing `[NN%]` from a pytest stdout line."""
    line = _XDIST_PREFIX_RE.sub("", line)
    line = _PERCENT_SUFFIX_RE.sub("", line)
    return line.rstrip()


def _parse_pytest_output(text: str) -> dict[str, int]:
    """Parse pytest -q stdout into pass/fail/error/skip/xfailed/xpassed counts.

    Counts dotted-progress characters (`.FEsxX`) across the full output AND
    parses the final `N passed, M failed, …` summary line. The summary
    line, when present, is authoritative; otherwise the per-character
    tally is returned.

    Only lines containing the canonical `in <duration>s` timing suffix are
    treated as summary candidates — this anchors against noise lines such as
    `"5 passed checks remaining"` that would otherwise overwrite the true
    summary if they happened to appear later in the stream.

    `x`/`X` in the progress stream map to `xfailed`/`xpassed` rather than
    `skipped`, matching pytest's actual semantics.

    This parser is output-mode-agnostic — it handles sequential `-q`,
    sequential `-q --cov`, and xdist `-n auto -q` output uniformly.
    """
    tally = _new_count_dict()
    summary: dict[str, int] | None = None

    for raw in text.splitlines():
        cleaned = _clean_pytest_line(raw)
        if not cleaned:
            continue

        # Progress line — characters only.
        if all(c in _PROGRESS_CHARS for c in cleaned):
            tally["passed"] += cleaned.count(".")
            tally["failed"] += cleaned.count("F")
            tally["errors"] += cleaned.count("E")
            tally["skipped"] += cleaned.count("s")
            tally["xfailed"] += cleaned.count("x")
            tally["xpassed"] += cleaned.count("X")
            continue

        # Authoritative summary line — last match wins. S3: require pytest's
        # canonical "in <duration>s" timing token so we never misclassify a
        # log line that happens to mention "N passed" as a summary.
        if not _SUMMARY_TIMING_RE.search(cleaned):
            continue
        matches = _COUNT_RE.findall(cleaned)
        if matches:
            local = _new_count_dict()
            for n, label in matches:
                value = int(n)
                if label == "passed":
                    local["passed"] = value
                elif label == "failed":
                    local["failed"] = value
                elif label in ("error", "errors"):
                    local["errors"] = value
                elif label == "skipped":
                    local["skipped"] = value
                elif label == "xfailed":
                    local["xfailed"] = value
                elif label == "xpassed":
                    local["xpassed"] = value
            summary = local

    return summary if summary is not None else tally


def _stream_pytest(
    cmd: list[str],
    collect_targets: list[str] | None = None,
) -> dict:
    """Run pytest with real-time progress reporting.

    Prints a progress line every ~10% of tests. Returns the same dict
    format as check_pytest(). The main subprocess runs with
    `COVERAGE_CORE=sysmon` for faster coverage tracking; the upfront
    `--collect-only` count subprocess uses the parent env (coverage is
    not active during collection).

    `collect_targets` (review-fix #01 finding 2): when provided, the
    upfront `--collect-only` subprocess walks ONLY those paths instead
    of the whole `TESTS_DIR` tree. `check_pytest_files` passes the same
    `abs_files` it gives to the main run so `--quick tests/test_foo.py`
    doesn't pay the 1–3 s cold cost of collecting the entire suite.
    The full-coverage caller (`check_pytest`) passes `None` and keeps
    the whole-tree collection semantics.

    Both subprocess invocations explicitly pass `encoding="utf-8"` and
    `errors="replace"` so decoding pytest output never crashes under
    `LANG=C` / `LC_ALL=POSIX` (S5). The default `text=True` would fall
    back to the system locale and raise `UnicodeDecodeError` mid-stream
    on non-ASCII content (Unicode ellipses, curly quotes in parametrize
    ids, non-ASCII assertion messages).
    """
    # First, collect test count — single-process, no coverage env override.
    # We use Popen directly (instead of `_run`) so the test suite can verify
    # this subprocess gets utf-8 encoding AND does NOT inherit COVERAGE_CORE.
    count_targets = collect_targets if collect_targets is not None else [str(TESTS_DIR)]
    count_cmd = [VENV_PYTHON, "-m", "pytest", "--collect-only", "-q", *count_targets]
    count_proc = subprocess.Popen(
        count_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPO_ROOT),
    )
    count_stdout, _ = count_proc.communicate()
    total_tests = 0
    for line in count_stdout.split("\n"):
        m = re.match(r"(\d+) tests? collected", line.strip())
        if m:
            total_tests = int(m.group(1))
            break

    milestone_every = max(1, total_tests // 10) if total_tests > 0 else 50

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPO_ROOT),
        env=_pytest_env(),
    )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    live_passed = 0
    live_failed = 0
    live_errors = 0
    last_milestone = 0

    def _read_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_lines.append(line)

    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stderr_thread.start()

    assert proc.stdout is not None
    for line in proc.stdout:
        stdout_lines.append(line)
        cleaned = _clean_pytest_line(line)

        # Mid-run progress: count chars on dotted-progress lines.
        if cleaned and all(c in _PROGRESS_CHARS for c in cleaned):
            live_passed += cleaned.count(".")
            live_failed += cleaned.count("F")
            live_errors += cleaned.count("E")
            current = live_passed + live_failed + live_errors
            if total_tests > 0 and current >= last_milestone + milestone_every:
                pct = min(100, int(current / total_tests * 100))
                sys.stderr.write(
                    f"  pytest: {pct}% ({current}/{total_tests})"
                    f" | passed={live_passed} failed={live_failed} errors={live_errors}\n"
                )
                sys.stderr.flush()
                last_milestone = current

    proc.wait()
    stderr_thread.join(timeout=5)

    stdout_text = "".join(stdout_lines)
    stderr_text = "".join(stderr_lines)

    # Final count line uses the authoritative parser (summary line wins).
    # S2: include skipped/xfailed/xpassed in the total so suites with xfail
    # markers don't appear short. Suppress zero-count labels for readability.
    counts = _parse_pytest_output(stdout_text)
    final_total = sum(counts[k] for k in _COUNT_KEYS)
    if total_tests > 0 or final_total > 0:
        denom = total_tests if total_tests > 0 else final_total
        parts = [
            f"passed={counts['passed']}",
            f"failed={counts['failed']}",
            f"errors={counts['errors']}",
        ]
        for optional_key in ("skipped", "xfailed", "xpassed"):
            if counts[optional_key]:
                parts.append(f"{optional_key}={counts[optional_key]}")
        sys.stderr.write(f"  pytest: done ({final_total}/{denom}) | {' '.join(parts)}\n")
        sys.stderr.flush()

    passed = proc.returncode == 0
    coverage = None
    missing_lines = []
    for line in stdout_text.split("\n"):
        if "TOTAL" in line and "%" in line:
            parts = line.split()
            for p in parts:
                if p.endswith("%"):
                    coverage = p
        if "FAIL" in line and "%" in line:
            missing_lines.append(line.strip())
    return {
        "name": "pytest",
        "passed": passed,
        "coverage": coverage,
        "missing": missing_lines[:10],
        "detail": stdout_text[-500:] if not passed else "",
        "stderr": stderr_text[-300:] if not passed else "",
    }


def check_pytest() -> dict:
    """Run pytest with 100% coverage check and progress reporting.

    Adds `-n <workers>` when `pytest-xdist` is available (configurable via
    `QS_QG_PYTEST_WORKERS`, default `"auto"`), prepends `--cov-report=`
    before `--cov-report=term-missing` to override `pytest.ini`'s html
    default without wiping our explicit report, and runs the subprocess
    with `COVERAGE_CORE=sysmon` (set inside `_stream_pytest`).

    Stderr warnings (mutually exclusive):
    - `[pytest] xdist not available, running single-process`
      when pytest-xdist is missing from the venv (silent fallback would
      otherwise hide a major perf regression);
    - `[pytest] QS_QG_PYTEST_WORKERS=<v> — running single-process by request`
      when xdist is available but the user explicitly forced serial mode
      via the env var (acknowledges the override took effect).
    """
    cmd = [
        VENV_PYTHON,
        "-m",
        "pytest",
        str(TESTS_DIR),
        f"--cov={SRC_DIR}",
        # S1: clear inherited --cov-report defaults from pytest.ini FIRST,
        # then add our explicit term-missing report. Reversing this order
        # lets the trailing empty value wipe term-missing — pytest-cov treats
        # `--cov-report=` (empty) as "disable all reports".
        "--cov-report=",
        "--cov-report=term-missing",
        "--cov-fail-under=100",
        "-q",
    ]
    workers = _pytest_workers()
    if workers is not None:
        cmd.extend(["-n", workers])
    else:
        # Emit a serial-mode rationale so users see *why* xdist isn't in use.
        if not _has_xdist():
            sys.stderr.write("[pytest] xdist not available, running single-process\n")
        else:
            # xdist available, but user-forced serial via QS_QG_PYTEST_WORKERS=0 or "".
            raw = os.environ.get("QS_QG_PYTEST_WORKERS", "").strip()
            sys.stderr.write(f"[pytest] QS_QG_PYTEST_WORKERS={raw!r} — running single-process by request\n")
        sys.stderr.flush()
    return _stream_pytest(cmd)


def check_pytest_files(test_files: list[str]) -> dict:
    """Run pytest on specific test paths (files or directories), no coverage.

    Honours `_pytest_workers()` for xdist parallelization — same resolver as
    the full gate (`check_pytest`), so `QS_QG_PYTEST_WORKERS` overrides apply
    here too. Silent serial fallback when xdist is missing or explicitly
    disabled (no stderr warning; only the full gate warns to avoid duplicate
    noise from the dev-only fast path and `--quick`).
    """
    if not test_files:
        return {"name": "pytest (no test files changed)", "passed": True, "detail": ""}

    abs_files = [str(REPO_ROOT / f) for f in test_files]
    workers = _pytest_workers()
    cmd = [VENV_PYTHON, "-m", "pytest", *abs_files, "-q"]
    if workers is not None:
        cmd.extend(["-n", workers])
    # Narrow the collect-only subprocess to the same paths so we don't
    # walk the whole `tests/` tree just to count one file's tests.
    return _stream_pytest(cmd, collect_targets=abs_files)


def check_ruff_lint(fix: bool = False) -> dict:
    """Run ruff check."""
    _emit("ruff_lint", "running")
    cmd = [_venv_tool("ruff"), "check", str(SRC_DIR)]
    if fix:
        cmd.append("--fix")
    result = _run(cmd)
    passed = result.returncode == 0
    _emit("ruff_lint", "PASS" if passed else "FAIL")
    return {
        "name": "ruff_lint",
        "passed": passed,
        "detail": result.stdout.strip()[-500:] if not passed else "",
    }


def check_ruff_format(fix: bool = False) -> dict:
    """Run ruff format check."""
    _emit("ruff_format", "running")
    if fix:
        cmd = [_venv_tool("ruff"), "format", str(SRC_DIR)]
    else:
        cmd = [_venv_tool("ruff"), "format", "--check", str(SRC_DIR)]
    result = _run(cmd)
    passed = result.returncode == 0
    _emit("ruff_format", "PASS" if passed else "FAIL")
    return {
        "name": "ruff_format",
        "passed": passed,
        "detail": result.stdout.strip()[-500:] if not passed else "",
    }


def check_mypy() -> dict:
    """Run mypy."""
    _emit("mypy", "running")
    cmd = [VENV_PYTHON, "-m", "mypy", str(SRC_DIR)]
    result = _run(cmd)
    passed = result.returncode == 0
    _emit("mypy", "PASS" if passed else "FAIL")
    return {
        "name": "mypy",
        "passed": passed,
        "detail": result.stdout.strip()[-500:] if not passed else "",
    }


def check_translations() -> dict:
    """Check if translations need regeneration."""
    _emit("translations", "checking")
    if not STRINGS_JSON.exists():
        _emit("translations", "SKIP (no strings.json)")
        return {"name": "translations", "passed": True, "detail": "no strings.json"}

    gen_script = REPO_ROOT / "scripts" / "generate-translations.sh"
    if not gen_script.exists():
        _emit("translations", "SKIP (no generate script)")
        return {"name": "translations", "passed": True, "detail": "no generate script"}

    # Capture current en.json content
    en_before = TRANSLATIONS_EN.read_text() if TRANSLATIONS_EN.exists() else ""

    # Run generation — ignore exit code (may fail for unrelated integrations)
    _run(["bash", str(gen_script)])

    # Check if our translations file changed (the only thing that matters)
    en_after = TRANSLATIONS_EN.read_text() if TRANSLATIONS_EN.exists() else ""
    if en_before != en_after:
        _emit("translations", "FAIL (outdated)")
        return {
            "name": "translations",
            "passed": False,
            "detail": "translations/en.json was outdated and has been regenerated. Stage the updated file.",
        }

    _emit("translations", "PASS")
    return {"name": "translations", "passed": True, "detail": ""}


def _run_cheap_gates_parallel(
    specs: list[tuple[Callable[..., Any], dict[str, Any], list[str]]],
) -> list[dict]:
    """Run cheap gates concurrently and return their results (unordered).

    Each spec is `(callable, kwargs, gate_names)`. The callable returns
    either a single result dict or a list of result dicts (the latter is
    used by the M1 serialized ruff-pair wrapper that produces two results
    from one future). `gate_names` lists the gate names this callable
    is responsible for — used to synthesize failure results when the
    callable raises (S6), so an unhandled exception in one gate doesn't
    abort the whole pipeline and let the traceback escape `main()`.
    """
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        futures = {pool.submit(fn, **kwargs): names for fn, kwargs, names in specs}
        for f in as_completed(futures):
            names = futures[f]
            try:
                result = f.result()
            except Exception as exc:  # noqa: BLE001 — synthesize a failure
                # Synthesize one failure dict per gate name this future owned.
                for name in names:
                    results.append(
                        {
                            "name": name,
                            "passed": False,
                            "detail": "",
                            "stderr": f"<exception>: {exc}",
                        }
                    )
                continue
            # The ruff-pair composite returns list[dict]; everything else
            # returns a single dict.
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)
    return results


def _output_results(
    results: list[dict],
    *,
    all_passed: bool,
    cached: bool,
    json_mode: bool,
    scope: str = "full",
) -> None:
    """Print gate results in human-readable or JSON format."""
    if json_mode:
        print(
            json.dumps(
                {
                    "all_passed": all_passed,
                    "cached": cached,
                    "scope": scope,
                    "gates": results,
                },
                indent=2,
            )
        )
    else:
        if cached:
            print("  [CACHED] All gates passed (cached result)\n")
        else:
            for r in results:
                status = "PASS" if r["passed"] else "FAIL"
                print(f"  [{status}] {r['name']}")
                if not r["passed"] and r.get("detail"):
                    for line in r["detail"].split("\n")[:5]:
                        print(f"         {line}")
            print()
            if all_passed:
                print("All quality gates passed.")
            else:
                failed = [r["name"] for r in results if not r["passed"]]
                print(f"FAILED gates: {', '.join(failed)}")


# --- QS-276: impacted-tests inner loop (`--impacted`) + testmon seeding ---
#
# `--impacted` runs only the testmon-selected tests under
# `--cov=custom_components/quiet_solar`, writes `coverage.xml`, and lets
# `diff-cover` assert the CHANGED lines are 100% covered. The whole-repo
# 100% gate stays authoritative in CI; this is the sub-~15s inner loop.

COVERAGE_XML = REPO_ROOT / "coverage.xml"
TESTMON_DATA = REPO_ROOT / ".testmondata"

# Static decision (Design A): pytest-testmon 2.2.0 does not reliably
# attribute per-test coverage across xdist workers, so the selection pass
# runs serially. Measured worst case (a full select against a cold
# `.testmondata`) approaches the warm full suite; a small diff selects a
# handful of tests and finishes in seconds. A *runtime* probe would add an
# uncoverable branch, so this is a module constant by design.
_TESTMON_SUPPORTS_XDIST = False


def _impacted_tooling_available() -> bool:
    """Return True iff `testmon` AND `diff_cover` import from VENV_PYTHON.

    Probes the venv interpreter (same reasoning as `_has_xdist`) so the
    check reflects the environment pytest / diff-cover actually run in,
    not the orchestrator's.
    """
    probe = (
        "import importlib.util as u, sys; "
        "sys.exit(0 if u.find_spec('testmon') and u.find_spec('diff_cover') else 1)"
    )
    return _run([VENV_PYTHON, "-c", probe]).returncode == 0


def _is_ci() -> bool:
    """Return True when running under CI (GitHub Actions sets `CI=true`)."""
    return os.environ.get("CI", "").strip().lower() in ("1", "true")


def _resolve_diff_base() -> str | None:
    """Resolve the `--impacted` diff base via the fallback ladder.

    Order: `origin/main` → `main` → merge-base of HEAD with the tracked
    upstream. `origin/main` is fetched fresh first so the local
    changed-line set matches CI's. Returns the resolved ref / sha, or
    None when nothing resolves (offline / fresh worktree with no
    upstream). The base is live git, independent of the seeded
    `.testmondata`.
    """
    # Best-effort fresh fetch; ignore failure — an offline dev still
    # resolves a local `main` or the upstream merge-base below.
    _run(["git", "fetch", "origin", "main"])
    for ref in ("origin/main", "main"):
        if _run(["git", "rev-parse", "--verify", ref]).returncode == 0:
            return ref
    upstream = _run(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    tracked = upstream.stdout.strip()
    if upstream.returncode == 0 and tracked:
        mb = _run(["git", "merge-base", "HEAD", tracked])
        sha = mb.stdout.strip()
        if mb.returncode == 0 and sha:
            return sha
    return None


def _ensure_testmon_db_safe() -> None:
    """Delete `.testmondata` when it is not a readable SQLite database.

    testmon recovers from a schema-version mismatch on its own (it
    rebuilds and selects all tests), but a genuinely corrupt /
    partially-written / non-SQLite file (a killed run, a dependency bump,
    a path relocation) makes it raise `sqlite3.DatabaseError` mid-run
    instead. Removing the file forces a clean rebuild → select-all,
    honouring the 'never silently under-select' fail-safe.
    """
    if not TESTMON_DATA.exists():
        return
    try:
        conn = sqlite3.connect(str(TESTMON_DATA))
        try:
            conn.execute("PRAGMA schema_version").fetchone()
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        _emit("impacted", "corrupt .testmondata detected — removing to force select-all")
        TESTMON_DATA.unlink()


def _build_testmon_cmd() -> list[str]:
    """Build the single-process testmon selection + coverage pytest argv.

    Design A preferred path (Task 2 confirmed testmon ⊕ pytest-cov share
    one process): the testmon-selected tests run under `--cov=<package>`,
    the leading empty `--cov-report=` clears pytest.ini's term default
    before the XML report, and there is NO `--cov-fail-under` — the 100%
    verdict is delegated to diff-cover on the changed lines only (a
    whole-tree subset run is always <100% and would otherwise fail before
    diff-cover runs). xdist is gated on the static `_TESTMON_SUPPORTS_XDIST`
    constant.
    """
    cmd = [
        VENV_PYTHON,
        "-m",
        "pytest",
        "--testmon",
        f"--cov={SRC_DIR}",
        "--cov-report=",
        f"--cov-report=xml:{COVERAGE_XML}",
        "-q",
    ]
    if _TESTMON_SUPPORTS_XDIST:
        workers = _pytest_workers()
        if workers is not None:
            cmd.extend(["-n", workers])
    return cmd


def _build_diff_cover_cmd(base: str) -> list[str]:
    """Build the diff-cover argv asserting changed lines are 100% covered."""
    return [
        _venv_tool("diff-cover"),
        str(COVERAGE_XML),
        f"--compare-branch={base}",
        "--fail-under=100",
    ]


def check_impacted() -> int:
    """Run the `--impacted` inner-loop gate; return the process exit code.

    Pipeline: tooling probe → diff-base ladder → testmon-selected tests
    under `--cov` (writes coverage.xml) → diff-cover --fail-under=100 on
    the changed lines.

    Exit codes: 0 pass · 1 selected-test failure OR diff-coverage <100% ·
    3 testmon / diff-cover not importable · 4 no diff base resolvable in
    CI (warn-and-skip → 0 locally so an offline dev isn't blocked).

    Fail-safe: a corrupt / schema-incompatible `.testmondata` makes
    testmon select *all* tests (its native recovery), never silently
    under-select.
    """
    if not _impacted_tooling_available():
        _emit("impacted", "pytest-testmon / diff-cover not importable — install requirements_test.txt")
        return 3

    base = _resolve_diff_base()
    if base is None:
        if _is_ci():
            _emit("impacted", "no diff base (origin/main) resolvable in CI")
            return 4
        _emit("impacted", "no diff base resolvable — skipping diff-coverage check (offline/fresh worktree)")
        return 0

    _ensure_testmon_db_safe()
    _emit("impacted", f"selecting impacted tests (testmon) vs base {base}")
    result = _stream_pytest(_build_testmon_cmd())
    if not result["passed"]:
        _emit("impacted", "FAIL (selected tests failed)")
        return 1

    dc = _run(_build_diff_cover_cmd(base))
    sys.stdout.write(dc.stdout)
    sys.stdout.flush()
    if dc.returncode != 0:
        _emit("impacted", "FAIL (changed lines <100% covered)")
        sys.stderr.write(dc.stderr)
        sys.stderr.flush()
        return 1
    _emit("impacted", "PASS (changed lines 100% covered)")
    return 0


def seed_testmon() -> int:
    """Refresh `.testmondata` via `pytest --testmon`; no coverage, no verdict.

    Sanctioned non-gate subcommand (see the project-rules raw-`pytest`
    carve-out) used by `finish-task` to rebuild the main-worktree
    baseline after a merge. Returns 0 once the selection completes (a
    failing test still updates the DB — there is no pass/fail verdict);
    returns 3 when testmon is not importable.
    """
    if not _impacted_tooling_available():
        _emit("seed-testmon", "pytest-testmon not importable — skipping")
        return 3
    _emit("seed-testmon", "refreshing .testmondata (pytest --testmon)")
    _stream_pytest([VENV_PYTHON, "-m", "pytest", "--testmon", "-q"])
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quality gates")
    parser.add_argument("--fix", action="store_true", help="Auto-fix formatting/lint issues")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--cache", action="store_true", help="Enable result caching based on git state")
    parser.add_argument("--no-cache", action="store_true", help="Force fresh run (overrides --cache)")
    parser.add_argument("--full", action="store_true", help="Force full gate run regardless of scope")
    parser.add_argument(
        "--quick",
        nargs="+",
        metavar="PATH",
        help=(
            "Fast iteration mode: run only the cited test paths (files "
            "or directories) with xdist + sysmon; skip ruff/mypy/"
            "translations/coverage."
        ),
    )
    parser.add_argument(
        "--impacted",
        action="store_true",
        help=(
            "Inner-loop gate: testmon-selected tests under --cov, then "
            "diff-cover --fail-under=100 on the changed lines. Mutex with "
            "--quick/--cache/--no-cache/--full/--fix."
        ),
    )
    parser.add_argument(
        "--seed-testmon",
        action="store_true",
        help="Refresh .testmondata via `pytest --testmon` (no coverage, no verdict).",
    )
    args = parser.parse_args()

    # --quick is mutually exclusive with cache/full/fix. argparse's nargs="+"
    # validation fires first for the "no paths" case (exit 2 with its own
    # "expected at least one argument" message), so this only handles the
    # explicit-conflict case.
    if args.quick and (args.cache or args.no_cache or args.full or args.fix):
        parser.error(
            "you cannot combine --quick with --cache, --no-cache, --full, or --fix"
        )

    # --impacted joins the same mutex: it is a self-contained inner-loop
    # gate with its own selection + coverage semantics, incompatible with
    # the cache/scope/fix machinery and with --quick's bare-pytest run.
    if args.impacted and (args.quick or args.cache or args.no_cache or args.full or args.fix):
        parser.error(
            "you cannot combine --impacted with --quick, --cache, --no-cache, --full, or --fix"
        )

    # Review-fix #01 finding 8 — `--quick ""` would otherwise resolve to
    # REPO_ROOT and silently collect the whole suite. Reject before any
    # path resolution.
    if args.quick and any(not p for p in args.quick):
        parser.error("--quick paths must be non-empty")

    # Review-fix #01 finding 7 — `REPO_ROOT / "/etc/passwd"` discards
    # REPO_ROOT (pathlib semantics) and `..` escapes the worktree. Both
    # would silently run pytest outside the repo. Resolve each path and
    # confirm it sits under REPO_ROOT before continuing.
    if args.quick:
        for raw in args.quick:
            resolved = (REPO_ROOT / raw).resolve()
            try:
                resolved.relative_to(REPO_ROOT)
            except ValueError:
                parser.error(f"--quick path must be inside the repo: {raw}")

    if args.quick:
        sys.stderr.write(
            f"[quick] running {len(args.quick)} path(s) with xdist + "
            f"sysmon (no coverage / ruff / mypy / translations)\n"
        )
        sys.stderr.flush()
        result = check_pytest_files(args.quick)
        result["name"] = f"pytest --quick ({len(args.quick)} path(s))"
        _output_results(
            [result],
            all_passed=result["passed"],
            cached=False,
            json_mode=args.json,
            scope="quick",
        )
        sys.exit(0 if result["passed"] else 1)

    # --impacted short-circuits before scope detection / caching, exactly
    # like --quick: it is its own gate with a bespoke exit-code table.
    if args.impacted:
        sys.exit(check_impacted())

    # --seed-testmon is a non-gate maintenance subcommand: refresh the DB
    # and exit, never touching scope detection or the coverage gate.
    if args.seed_testmon:
        sys.exit(seed_testmon())

    use_cache = args.cache and not args.fix and not args.no_cache

    # Check cache before running gates
    if use_cache:
        branch, commit, is_clean = _get_git_state()
        cache = _read_cache()
        if cache is not None and _is_cache_valid(cache, branch, commit, is_clean):
            _output_results(
                cache["results"],
                all_passed=True,
                cached=True,
                json_mode=args.json,
            )
            sys.exit(0)

    # Detect scope
    changed_files = _get_changed_files()
    scope_info = _detect_scope(changed_files)
    scope = scope_info["scope"]
    if args.full:
        scope = "full"

    if scope == "dev-only":
        sys.stderr.write(f"  scope: DEV-ONLY ({scope_info['reason']})\n")
        sys.stderr.write("  skipping ruff, mypy, translations, full coverage\n")
        sys.stderr.flush()

        results = []
        test_files = scope_info["changed_test_files"]
        result = check_pytest_files(test_files)
        # Rename for clarity
        if test_files:
            result["name"] = f"pytest ({len(test_files)} changed test files)"
        results.append(result)
    elif scope == "ui-only":
        sys.stderr.write(f"  scope: UI-ONLY ({scope_info['reason']})\n")
        sys.stderr.write("  skipping ruff, mypy, translations, full coverage\n")
        sys.stderr.flush()

        results = []
        # Always run the canonical dashboard-rendering test; dedupe against
        # any test files that also appear in the diff via set semantics.
        test_files = sorted({*_UI_FAST_PATH_TESTS, *scope_info["changed_test_files"]})
        result = check_pytest_files(test_files)
        result["name"] = f"pytest (UI fast path, {len(test_files)} file(s))"
        results.append(result)
    else:
        sys.stderr.write(f"  scope: FULL ({scope_info['reason']})\n")
        sys.stderr.flush()

        # Run the cheap gates concurrently — they all spawn short-lived
        # subprocesses and benefit from overlap. Pytest runs serially
        # afterwards so xdist's `-n auto` workers don't oversubscribe the CPU.
        #
        # M1: under --fix, `ruff format` and `ruff check --fix` both rewrite
        # the same files; running them concurrently produces a write race
        # (lost edits, truncated writes, non-deterministic results). Serialize
        # them inside one composite future while the read-only gates (mypy,
        # translations) keep running concurrently.
        gate_order = ["ruff_format", "ruff_lint", "mypy", "translations"]

        if args.fix:

            def _ruff_pair_serial() -> list[dict]:
                """Run the two write-mutating ruff gates back-to-back."""
                return [
                    check_ruff_format(fix=True),
                    check_ruff_lint(fix=True),
                ]

            specs: list[tuple[Callable[..., Any], dict[str, Any], list[str]]] = [
                (_ruff_pair_serial, {}, ["ruff_format", "ruff_lint"]),
                (check_mypy, {}, ["mypy"]),
                (check_translations, {}, ["translations"]),
            ]
        else:
            specs = [
                (check_ruff_format, {"fix": False}, ["ruff_format"]),
                (check_ruff_lint, {"fix": False}, ["ruff_lint"]),
                (check_mypy, {}, ["mypy"]),
                (check_translations, {}, ["translations"]),
            ]

        cheap_results = _run_cheap_gates_parallel(specs)
        results = sorted(cheap_results, key=lambda r: gate_order.index(r["name"]))
        results.append(check_pytest())

    all_passed = all(r["passed"] for r in results)

    # Write cache on pass (only when caching is enabled and full scope)
    if use_cache and all_passed and scope == "full":
        branch, commit, is_clean = _get_git_state()
        if is_clean:
            _write_cache(branch, commit, results)

    _output_results(results, all_passed=all_passed, cached=False, json_mode=args.json, scope=scope)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
