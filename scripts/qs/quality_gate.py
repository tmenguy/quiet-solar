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
import math
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
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


def _run(
    cmd: list[str], cwd: str | None = None, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess, capturing stdout/stderr.

    `timeout` (QS-276 review-fix S1) bounds calls on the inner-loop hot
    path (e.g. the `--impacted` `git fetch`): a hung/slow remote (VPN
    drop, dead SSH host) must not block the sub-15s gate indefinitely. A
    timeout is surfaced as a non-zero `CompletedProcess` (returncode
    124, matching the shell convention) rather than a raised
    `TimeoutExpired`, so callers handle it via the normal returncode
    path. Callers that omit `timeout` keep the original unbounded
    behavior.

    QS-276 review-fix SF1: pins `encoding="utf-8", errors="replace"`
    (matching `_stream_pytest`) so non-ASCII output from git / diff-cover
    can't raise `UnicodeDecodeError` under `LANG=C` / `LC_ALL=POSIX` and
    bypass the normal return-code handling.

    QS-276 review-fix MF1 (#04): a missing executable (e.g. the venv
    interpreter absent on a CI runner) is surfaced as returncode 127
    (shell "command not found"), NOT a raised `FileNotFoundError`. This
    lets the tooling probes (`_impacted_tooling_available`,
    `_testmon_available`, `_has_xdist`) degrade to "unavailable" → the
    integration tests skip cleanly instead of erroring.
    """
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd or str(REPO_ROOT),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        # NH1: preserve any partial output captured before the timeout —
        # it aids diagnosis. `exc.stdout`/`.stderr` may be bytes (decode),
        # str (already decoded), or None.
        out = _decode_maybe(exc.stdout)
        err = _decode_maybe(exc.stderr)
        timeout_note = f"timed out after {timeout}s"
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=124,
            stdout=out,
            # SF-C (#04): use `err.strip()` so a whitespace-only stderr is
            # treated as empty and doesn't inject a leading blank line.
            stderr=f"{err}\n{timeout_note}".strip() if err.strip() else timeout_note,
        )
    except FileNotFoundError as exc:
        # MF1: executable not found — degrade to a non-zero result so
        # callers handle it via the normal return-code path (probes treat
        # 127 as "tool unavailable").
        return subprocess.CompletedProcess(args=cmd, returncode=127, stdout="", stderr=str(exc))


def _decode_maybe(value: bytes | str | None) -> str:
    """Return `value` as text — decode bytes (utf-8/replace), pass through str, "" for None."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


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
        # QS-276 review-fix NH4 (#03): expose the raw pytest exit code so
        # callers (e.g. seed_testmon) can distinguish a mere test failure
        # (1, DB still updated) from a collection error / internal crash
        # (>=2, .testmondata may be partial).
        "returncode": proc.returncode,
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

# NH1 (review-fix #04): `--impacted` writes a single fixed coverage.xml at
# the repo root and immediately consumes it with diff-cover. This assumes
# ONE `--impacted` run per worktree at a time — two overlapping runs in the
# same worktree would race on this path (run A's diff-cover could read run
# B's freshly-overwritten XML). The implement agents never run `--impacted`
# concurrently in one worktree (each task has its own worktree), so this
# documented single-run constraint is the accepted resolution rather than a
# per-run temp path.
COVERAGE_XML = REPO_ROOT / "coverage.xml"
TESTMON_DATA = REPO_ROOT / ".testmondata"
# QS-286: sidecar completion marker for the detached `--seed-testmon` run.
# REPO_ROOT is `__file__`-relative (cwd- and symlink-resolved), so the marker
# always lands beside the main worktree's `.testmondata` regardless of the
# detached process's cwd — a later `--seed-testmon-status` query from
# $MAIN_DIR reads the same file.
SEED_STATUS = REPO_ROOT / ".testmondata.seed-status"
# QS-278: the persistent coverage DATA file (coverage.py's default). The
# `--impacted` gate runs testmon-selected tests with `--cov-append` so
# coverage ACCUMULATES across inner-loop invocations. testmon 2.2.0
# correctly selects 0 tests for a no-op re-run (and only a small subset for
# a single-file edit); without accumulation, pytest-cov would erase
# `.coverage` and emit a report covering only THIS run, so any line changed
# vs origin/main but not re-exercised this run would read as uncovered and
# diff-cover would FAIL spuriously. Accumulation keeps prior coverage so the
# changed-line verdict stays correct while the run stays fast. The data is
# reset only on a fresh select-all baseline (see `_reset_coverage_data`).
COVERAGE_DATA = REPO_ROOT / ".coverage"

# QS-276 review-fix S1: cap the inner-loop `git fetch origin main` so a
# dead/slow remote degrades to a stale base + warning instead of hanging
# the gate. CI re-fetches authoritatively, so a short bound is safe.
_FETCH_TIMEOUT_SECONDS = 15.0

# QS-276 review-fix NH1: bound the diff-cover subprocess too — a
# pathological hang there would break the same sub-15s inner-loop promise
# as an unbounded fetch (S1). diff-cover only parses an already-written
# coverage.xml + a git diff, so a generous cap is safe.
_DIFF_COVER_TIMEOUT_SECONDS = 60.0

# Static decision (Design A): whether the pinned pytest-testmon attributes
# per-test coverage correctly across xdist workers, so the selection /
# seeding passes can run in parallel.
#
# QS-276 review-fix: empirically verified TRUE for pytest-testmon==2.2.0 +
# pytest-xdist==3.8.0. testmon 2.2.0 ships first-class xdist support
# (`TestmonXdistSync`, `pytest_xdist_node_collection_finished`,
# controller↔worker `workerinput` plumbing). Probed on a throwaway repo:
# under `-n auto` the superset property holds exactly (no-op → 0 selected,
# covered edit → reselect, uncovered edit → 0) AND testmon ⊕ pytest-cov
# combine the per-worker `.coverage` files into a valid `coverage.xml`
# that diff-cover reads at 100%. Leaving this serial made the select-all
# worst case (~45 min) *slower* than the `-n auto` full gate — the whole
# point of the inner loop is lost. A *runtime* probe of testmon's own
# xdist support would add an uncoverable branch, so this stays a module
# constant; `_pytest_workers()` (already probed + tested) still decides
# the actual worker count and honours `QS_QG_PYTEST_WORKERS`.
_TESTMON_SUPPORTS_XDIST = True


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


def _testmon_available() -> bool:
    """Return True iff `testmon` imports from VENV_PYTHON.

    Narrower probe than `_impacted_tooling_available` (QS-276 review-fix
    S2): `--seed-testmon` only runs `pytest --testmon` — it never invokes
    diff-cover — so the post-merge baseline refresh must not be blocked
    when diff-cover happens to be absent.
    """
    probe = "import importlib.util as u, sys; sys.exit(0 if u.find_spec('testmon') else 1)"
    return _run([VENV_PYTHON, "-c", probe]).returncode == 0


# QS-276 review-fix N3/NH4: accept the common truthy spellings CI systems
# use — `1`, `true`, `yes`, `on`, plus the single-letter `y` / `t` some
# providers emit — and also honor GitHub Actions' own provider var, which
# it always sets to `true`.
_CI_TRUTHY = ("1", "true", "yes", "on", "y", "t")


def _is_ci() -> bool:
    """Return True when running under CI.

    Recognizes a truthy `CI` env var (accepted values:
    `1`/`true`/`yes`/`on`/`y`/`t`, case-insensitive; GitHub Actions sets
    `CI=true`) and falls back to GitHub Actions' own `GITHUB_ACTIONS=true`
    provider variable.
    """
    if os.environ.get("CI", "").strip().lower() in _CI_TRUTHY:
        return True
    return os.environ.get("GITHUB_ACTIONS", "").strip().lower() in _CI_TRUTHY


def _resolve_diff_base() -> str | None:
    """Resolve the `--impacted` diff base via the fallback ladder.

    Order: `origin/main` → `main` → merge-base of HEAD with the tracked
    upstream. `origin/main` is fetched fresh first so the local
    changed-line set matches CI's. Returns the resolved ref / sha, or
    None when nothing resolves (offline / fresh worktree with no
    upstream). The base is live git, independent of the seeded
    `.testmondata`.

    QS-276 review-fix NH2: a candidate ref is only accepted if it shares
    a merge-base with HEAD. A shallow CI clone (`fetch-depth: 1`) can
    resolve `origin/main` to a commit with no common ancestor, which
    would make diff-cover diff against unrelated history and report a
    spuriously huge changed-line set. An unreachable ref is skipped (and
    a warning emitted); if nothing reachable resolves, we return None and
    the caller applies the no-base policy (warn-skip locally, exit 4 CI).
    """
    # Best-effort fresh fetch; ignore failure — an offline dev still
    # resolves a local `main` or the upstream merge-base below.
    #
    # QS-276 review-fix S1: bound the fetch so a hung/slow remote can't
    # block the sub-15s inner loop, and WARN on a non-zero result so a
    # silently-failed fetch (offline / timeout) — which leaves a stale
    # `origin/main` whose changed-line set may diverge from CI's — is
    # observable instead of pretending the base was fetched fresh.
    fetch = _run(["git", "fetch", "origin", "main"], timeout=_FETCH_TIMEOUT_SECONDS)
    if fetch.returncode != 0:
        _emit(
            "impacted",
            "warning: `git fetch origin main` failed/timed out — diff base may be stale vs CI",
        )
    for ref in ("origin/main", "main"):
        if _run(["git", "rev-parse", "--verify", ref]).returncode != 0:
            continue
        # NH2 (#02): require a common ancestor with HEAD before trusting the ref.
        if _run(["git", "merge-base", ref, "HEAD"]).returncode == 0:
            # NH2 (#05): name the chosen base so an unexpected diff-cover range is debuggable.
            _emit("impacted", f"diff base: {ref}")
            return ref
        _emit("impacted", f"warning: {ref} has no merge-base with HEAD (shallow clone?) — skipping")
    upstream = _run(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    tracked = upstream.stdout.strip()
    if upstream.returncode == 0 and tracked:
        mb = _run(["git", "merge-base", "HEAD", tracked])
        sha = mb.stdout.strip()
        if mb.returncode == 0 and sha:
            _emit("impacted", f"diff base: {sha} (merge-base with {tracked})")
            return sha
    return None


def _testmon_schema_version() -> int | None:
    """The DB schema version (`PRAGMA user_version`) the installed testmon
    expects, probed in VENV_PYTHON — the interpreter pytest actually runs
    under (mirroring `_testmon_available`). Returns None when testmon is not
    importable there or the constant is unreadable, so callers fall back to
    leaving the DB untouched rather than purging on an unknown.
    """
    res = _run([VENV_PYTHON, "-c", "import testmon.db as d; print(int(d.DATA_VERSION))"])
    if res.returncode != 0:
        return None
    try:
        return int(res.stdout.strip())
    except (TypeError, ValueError):
        return None


def _purge_testmon_db(reason: str) -> None:
    """Unlink `.testmondata` AND its SQLite WAL/SHM sidecars so the next run
    rebuilds from scratch (select-all).

    QS-278 review-fix #01-2: the `-wal` / `-shm` sidecars must go too. A fresh
    `.testmondata` opened against an orphaned `-wal` can replay stale WAL
    frames or raise, so a corrupt-DB recovery that left them behind merely
    relocated the breakage to the next run. `missing_ok=True` everywhere
    (QS-276 N6): a file that vanishes between probe and unlink (a concurrent
    run / another worktree) already satisfies the select-all intent.
    """
    _emit("impacted", f"{reason} — removing .testmondata (+WAL/SHM) to force select-all")
    TESTMON_DATA.unlink(missing_ok=True)
    for suffix in ("-wal", "-shm"):
        (TESTMON_DATA.parent / (TESTMON_DATA.name + suffix)).unlink(missing_ok=True)


def _ensure_testmon_db_safe() -> None:
    """Purge `.testmondata` (+WAL/SHM) when it is unusable as an INCREMENTAL
    baseline, so testmon rebuilds cleanly and `check_impacted`'s absence→reset
    branch clears the accumulated `--cov-append` coverage. Two purge triggers:

    1. **Not a readable SQLite DB** — a genuinely corrupt / partially-written
       / non-SQLite file (a killed run, a path relocation) raises
       `sqlite3.DatabaseError`. (QS-276)
    2. **Schema-version mismatch** (QS-278 review-fix #01-1) — testmon normally
       rebuilds IN PLACE on a `PRAGMA user_version` mismatch (e.g. after a
       testmon version bump), leaving the file present, so
       `TESTMON_DATA.exists()` stays True and the coverage reset would be
       skipped while a select-all runs and `--cov-append`s fresh coverage onto
       STALE accumulated `.coverage` — masking a genuine changed-line gap.
       Detecting the mismatch up front and purging unifies every select-all
       trigger behind the absence→reset path.

    QS-276 review-fix SF1: `sqlite3.OperationalError` ("database is locked")
    is a *subclass* of `sqlite3.DatabaseError`. A VALID DB that is transiently
    locked by a concurrent `--seed-testmon` / `--impacted` run must NOT be
    deleted — wiping it (and its sidecars) would corrupt that live run's
    baseline. So catch `OperationalError` FIRST and leave everything intact;
    the schema probe is skipped too (we could not read it under the lock
    anyway). Only a genuine corruption or a confirmed schema mismatch purges.
    """
    if not TESTMON_DATA.exists():
        # QS-283 A2: the primary is gone but its `-wal`/`-shm` sidecars may
        # linger — a tool-reachable state, NOT just a manual-`rm` artifact:
        # `_purge_testmon_db` unlinks the primary BEFORE its sidecars, so a
        # run killed mid-purge leaves exactly absent-primary + orphan-sidecar.
        # testmon would then reopen an empty DB against a stale WAL and select
        # `0 tests` (the QS_281 dead end). Unlink the orphans so the next
        # `pytest --testmon` rebuilds cleanly (select-all). Reuse the same
        # suffix loop as `_purge_testmon_db`; `missing_ok=True` (N6) tolerates
        # a sidecar vanishing under a concurrent run.
        for suffix in ("-wal", "-shm"):
            (TESTMON_DATA.parent / (TESTMON_DATA.name + suffix)).unlink(missing_ok=True)
        return
    try:
        conn = sqlite3.connect(str(TESTMON_DATA))
        try:
            stored = int(conn.execute("PRAGMA user_version").fetchone()[0])
        finally:
            conn.close()
    except sqlite3.OperationalError as exc:
        # Valid-but-busy (locked) — leave it; testmon handles the lock.
        _emit("impacted", f".testmondata is locked/busy — leaving intact ({exc})")
        return
    except sqlite3.DatabaseError:
        _purge_testmon_db("corrupt .testmondata detected")
        return

    expected = _testmon_schema_version()
    if expected is not None and stored != expected:
        _purge_testmon_db(f"testmon schema-version mismatch (db={stored}, expected={expected})")


def _reset_coverage_data() -> None:
    """Erase the accumulated coverage data before a fresh select-all baseline.

    QS-278: `--impacted` runs with `--cov-append`, so coverage accumulates
    across inner-loop invocations (keeping the changed-line verdict correct
    when testmon reselects 0 tests). That accumulation must be reset
    whenever testmon is about to select *all* tests — a first-ever run, a
    rebuilt/purged `.testmondata`, or a schema-version mismatch (see
    `_ensure_testmon_db_safe`) — otherwise stale coverage from an earlier
    branch state could mask a genuine gap. A select-all run rewrites the full
    picture from scratch, so a clean slate here is both safe and necessary.
    Removes the primary `.coverage` plus any leftover xdist per-worker shards.

    QS-278 review-fix #01-5: the shard glob is resolved from `COVERAGE_DATA`'s
    OWN directory, never a separately-resolved root, so the primary file and
    its `.coverage.*` shards can never target divergent dirs. Single-writer
    assumption (see the `COVERAGE_XML` note): at most one `--impacted` runs per
    worktree at a time, so the glob-then-unlink need not be atomic.
    """
    COVERAGE_DATA.unlink(missing_ok=True)
    for shard in COVERAGE_DATA.parent.glob(COVERAGE_DATA.name + ".*"):
        shard.unlink(missing_ok=True)


def _clean_orphan_cov_shards() -> None:
    """Remove pre-existing `.coverage.*` xdist shards (NEVER the combined
    `.coverage`) at the top of the normal incremental `--impacted` path.

    QS-283 A1: a killed / crashed `--impacted` run leaves per-worker coverage
    shards (`.coverage.<host>.<pid>.*`) that never got combined into the
    primary `.coverage`. The next `--cov-append` run would fold those STALE
    fragments into `coverage.xml`, so a line covered only by an old branch
    state appears "covered" now ("partial coverage appears" in the QS_281
    transcript). Reaping the orphan shards before the run removes that false
    signal while preserving the combined `.coverage` so the warm-baseline
    accumulation model (QS-278) still works.

    Reuses the SAME glob expression as `_reset_coverage_data`
    (`COVERAGE_DATA.parent.glob(COVERAGE_DATA.name + ".*")`) so the two helpers
    never drift into divergent shard-matching rules — they differ ONLY in
    whether the primary `.coverage` is also unlinked (here it is not). The
    glob `.coverage.*` does not match the bare `.coverage`, so the combined
    file always survives. Called BEFORE the testmon/cov pytest subprocess
    spawns, so it can never reap a live worker's shard mid-run (single-writer
    assumption, same as `_reset_coverage_data` / `COVERAGE_XML`).
    """
    for shard in COVERAGE_DATA.parent.glob(COVERAGE_DATA.name + ".*"):
        shard.unlink(missing_ok=True)


def _rebuild_testmon_baseline() -> None:
    """Purge `.testmondata` (+WAL/SHM) AND clear accumulated coverage so the
    next `pytest --testmon` pass is a clean from-scratch select-all.

    QS-283: the shared rebuild used by BOTH `seed_testmon` (A3) and the
    `check_impacted` self-heal retry (A4). Spelling the rebuild — purge +
    coverage reset + shard clear — once guarantees the two callers can never
    diverge. `_reset_coverage_data()` already unlinks the combined `.coverage`
    AND globs/unlinks its `.coverage.*` shards, so after this helper the next
    select-all run `--cov-append`s onto an empty store: no orphan shard can
    re-fold in, and a stale baseline cannot survive to under-select.
    """
    _purge_testmon_db("rebuilding testmon baseline")
    _reset_coverage_data()


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
        # QS-276 review-fix MF1: keep the slow testmon SELF-tests (the
        # real-subprocess git-init + nested-pytest cases in
        # tests/test_quality_gate.py) out of the inner loop WITHOUT
        # deselecting the shared `integration` marker. The earlier
        # `-m "not integration"` (review-fix N5) was wrong: ~9 DOMAIN
        # integration test files (charger rebalancing, solver forecast,
        # constraint boundaries, …) carry that marker AND exercise
        # `custom_components/quiet_solar`. Deselecting them meant a
        # production line covered ONLY by a domain integration test ran
        # zero times → 0% in coverage.xml → diff-cover FAIL the dev could
        # never clear locally. Ignoring by PATH targets only the
        # self-tests (which never cover production code, so the diff-cover
        # verdict is unaffected) and leaves every domain test selectable.
        f"--ignore={TESTS_DIR / 'test_quality_gate.py'}",
        f"--cov={SRC_DIR}",
        # QS-278: accumulate coverage across inner-loop runs. testmon
        # reselects 0 tests for a no-op re-run and only an edit's subset for
        # an incremental change; `--cov-append` keeps the prior runs'
        # coverage so the diff-cover changed-line verdict (vs origin/main)
        # stays correct without re-running the whole impacted set every
        # time. The data is reset on a fresh select-all baseline so it can't
        # grow stale across unrelated branch states.
        "--cov-append",
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
    """Build the diff-cover argv asserting changed lines are 100% covered.

    QS-276 review-fix SF-A (#04): `--include-untracked` so a brand-new,
    not-yet-`git add`-ed source file with an untested function is counted
    as changed lines (diff-cover 10.3.0 defaults to excluding untracked
    files, which made the dominant inner-loop case — new code starts
    untracked — score a vacuous 100% PASS before staging, defeating AC#6
    locally).
    """
    return [
        _venv_tool("diff-cover"),
        str(COVERAGE_XML),
        f"--compare-branch={base}",
        "--include-untracked",
        "--fail-under=100",
    ]


# QS-276 review-fix NH2 (#03): single source of truth for the reseed command.
_SEED_TESTMON_CMD = "python scripts/qs/quality_gate.py --seed-testmon"


def _emit_reseed_guidance() -> None:
    """Emit the reseed hint on a diff-cover coverage failure.

    QS-276 review-fix SF-B (#04): the previous "baseline stale" branch was
    removed. diff-cover's pinned default uses three-dot range notation
    (`merge-base(base, HEAD)...HEAD`), so an advanced `origin/main` does
    NOT inflate the changed-line set — the staleness false-FAIL it guarded
    against cannot occur, and the branch only mislabeled genuine
    uncovered-line failures. This single hint remains useful because
    testmon's *selection* (vs the seeded baseline) is independent of
    diff-cover's range: a stale baseline can still under-select, so a
    reseed is a reasonable thing to suggest on any coverage failure.
    """
    _emit(
        "impacted",
        f"hint: if a changed line looks covered but failed, your testmon baseline "
        f"may be stale — reseed with `{_SEED_TESTMON_CMD}` and re-run",
    )


# QS-283 A4: verdicts returned by `_run_impacted_pass`. Only
# `changed_lines_uncovered` on an incremental run is self-heal-retriable;
# every other non-pass verdict is a genuine, non-retriable failure.
_IMPACTED_PASS = "pass"
_IMPACTED_TESTS_FAILED = "tests_failed"
_IMPACTED_NO_COVERAGE_XML = "no_coverage_xml"
_IMPACTED_DIFF_COVER_TIMEOUT = "diff_cover_timeout"
_IMPACTED_CHANGED_LINES_UNCOVERED = "changed_lines_uncovered"


def _run_impacted_pass(base: str) -> tuple[str, bool]:
    """Run ONE testmon-selected pass against `base`; return `(verdict, ran_select_all)`.

    Pipeline: DB hygiene (`_ensure_testmon_db_safe`, which may purge a corrupt /
    schema-mismatched baseline) → reset accumulated coverage iff the baseline is
    now absent (so the pass starts clean) → testmon-selected tests under `--cov`
    (writes coverage.xml) → diff-cover --fail-under=100 on the changed lines.
    Factored out of `check_impacted` (QS-283 A4) so the self-heal retry can
    re-run the SAME pass against the already-resolved base — no second
    `git fetch` / tooling probe.

    `ran_select_all` reports whether THIS pass ran as a select-all: it is the
    post-hygiene `.testmondata` absence, so it is True both for a genuinely
    fresh baseline AND when hygiene just purged a corrupt/schema-mismatched DB
    mid-pass. `check_impacted` uses it to suppress a pointless self-heal retry
    when the first pass already select-all'd (review fix #01). Returns one of
    the `_IMPACTED_*` verdicts paired with that flag.
    """
    _ensure_testmon_db_safe()
    # QS-278: a missing `.testmondata` here (first-ever run, or just purged by
    # the hygiene above as corrupt/schema-mismatched) means testmon is about to
    # select ALL tests — a fresh baseline. Reset the accumulated `--cov-append`
    # coverage data so it reflects only this clean full run, never stale lines
    # from an earlier branch state. When the DB exists, accumulation is
    # intentional. `ran_select_all` records this select-all decision for the
    # caller's self-heal gate (review fix #01: a baseline purged mid-pass means
    # this pass IS the clean select-all, so no retry can improve on it).
    ran_select_all = not TESTMON_DATA.exists()
    if ran_select_all:
        _reset_coverage_data()
    # QS-276 review-fix SF1 (#05): delete any stale coverage.xml from a
    # previous run BEFORE the testmon/cov pass. Otherwise a pytest-cov
    # emission failure would be masked — the N7 exists-guard below would
    # see the *old* report and diff-cover would score the changed lines
    # against stale data instead of failing loudly.
    COVERAGE_XML.unlink(missing_ok=True)
    _emit("impacted", f"selecting impacted tests (testmon) vs base {base}")
    result = _stream_pytest(_build_testmon_cmd())
    if not result["passed"]:
        _emit("impacted", "FAIL (selected tests failed)")
        return _IMPACTED_TESTS_FAILED, ran_select_all

    # QS-276 review-fix N7: pytest-cov writes coverage.xml even on a
    # zero-collection run (verified for the pinned version), but guard
    # defensively — if a future pytest-cov stops emitting it, diff-cover
    # would silently read a stale/missing file. Combined with the SF1
    # pre-delete above, a fresh report is guaranteed or we fail loudly.
    if not COVERAGE_XML.exists():
        _emit("impacted", f"FAIL (coverage report not written: {COVERAGE_XML})")
        return _IMPACTED_NO_COVERAGE_XML, ran_select_all

    # NH1: bound diff-cover so a hang can't break the sub-15s promise.
    dc = _run(_build_diff_cover_cmd(base), timeout=_DIFF_COVER_TIMEOUT_SECONDS)

    # QS-276 review-fix SF2 (#03) + NH2 (#04): a timed-out diff-cover
    # returns the synthetic 124 from `_run`. Report THAT as the primary
    # verdict FIRST — before writing dc.stdout (which is empty/partial on
    # timeout, NH2) and before the coverage messaging (which would
    # mislabel a timeout as "<100% covered", SF2).
    if dc.returncode == 124:
        _emit("impacted", f"FAIL (diff-cover timed out after {_DIFF_COVER_TIMEOUT_SECONDS}s)")
        sys.stderr.write(dc.stderr)
        sys.stderr.flush()
        return _IMPACTED_DIFF_COVER_TIMEOUT, ran_select_all

    sys.stdout.write(dc.stdout)
    sys.stdout.flush()
    if dc.returncode != 0:
        # SF-B (#04): the "baseline stale" branch was removed (diff-cover's
        # three-dot range means an advanced origin/main can't inflate the
        # changed-line set, so that scenario never occurs). A non-zero,
        # non-timeout diff-cover is a changed-line coverage gap — possibly a
        # testmon/coverage desync (self-heal-retriable) or a genuine gap.
        _emit("impacted", "FAIL (changed lines <100% covered)")
        _emit_reseed_guidance()
        sys.stderr.write(dc.stderr)
        sys.stderr.flush()
        return _IMPACTED_CHANGED_LINES_UNCOVERED, ran_select_all
    _emit("impacted", "PASS (changed lines 100% covered)")
    return _IMPACTED_PASS, ran_select_all


def check_impacted() -> int:
    """Run the `--impacted` inner-loop gate; return the process exit code.

    Pipeline: tooling probe → diff-base ladder → orphan-shard hygiene →
    one `_run_impacted_pass` → (on an incremental changed-line FAIL) exactly
    one self-heal rebuild + retry.

    Exit codes: 0 pass · 1 selected-test failure OR diff-coverage <100% ·
    3 testmon / diff-cover not importable · 4 no diff base resolvable in
    CI (warn-and-skip → 0 locally so an offline dev isn't blocked).

    Fail-safe: a corrupt / schema-incompatible `.testmondata` makes
    testmon select *all* tests (its native recovery), never silently
    under-select.

    QS-283 A4 self-heal: on an INCREMENTAL run (`was_incremental` —
    `.testmondata` present and non-empty, captured BEFORE any purge — AND the
    first pass did not itself select-all) that reports changed lines <100%, the
    desync killer fires: rebuild the testmon baseline (purge + coverage reset +
    shard clear) and re-run the SAME pass once as a clean select-all. A false
    FAIL (a covering test wrongly deselected) recovers to PASS; a genuine gap
    still exits 1. The retry calls `_run_impacted_pass` directly (not
    `check_impacted`), so it cannot recurse.
    A select-all run skips the retry — its FAIL is already ground truth. The
    select-all signal is the pass's own `ran_select_all` (post-hygiene), so a
    baseline that hygiene purged mid-pass (corrupt/schema-mismatched) is
    correctly treated as non-incremental and never triggers a wasted retry
    (review fix #01). The retry's own per-verdict diagnostics are emitted by
    `_run_impacted_pass` itself before the exit code is collapsed.
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

    # QS-283 A4 trigger gate: capture the incremental signal BEFORE any
    # hygiene step can purge the DB. A non-empty `.testmondata` means testmon
    # has a warm baseline and is about to INCREMENTALLY select a subset — the
    # only case where a changed-line FAIL might be a testmon/coverage desync
    # worth self-healing. An absent/empty DB means this run already select-alls
    # (ground truth), so a FAIL is genuine and the retry is skipped — no wasted
    # select-all on the normal TDD-red case.
    #
    # Review fix #02: derive existence AND size from a SINGLE `stat()` inside a
    # try/except so the file vanishing (a concurrent `--seed-testmon`/`--impacted`
    # run, another worktree, or a mid-`_purge_testmon_db`) between a probe and a
    # read cannot crash the gate — a vanished/unreadable baseline is treated as
    # non-incremental, matching the module's tolerate-vanish concurrency model.
    try:
        was_incremental = TESTMON_DATA.stat().st_size > 0
    except OSError:
        # OSError covers FileNotFoundError (vanished baseline) plus any other
        # transient stat failure — treat all as non-incremental.
        was_incremental = False

    # QS-283 A1: reap orphaned `.coverage.*` shards from a killed prior run
    # before the cov pass, so stale fragments can't `--cov-append` into this
    # run's coverage.xml. The combined `.coverage` survives (warm baseline).
    _clean_orphan_cov_shards()

    verdict, ran_select_all = _run_impacted_pass(base)
    if verdict == _IMPACTED_PASS:
        return 0
    # Self-heal only a genuine INCREMENTAL desync. Two conditions must hold:
    # the verdict is a changed-line miss, and the run was truly incremental —
    # the pre-hygiene baseline was warm (`was_incremental`) AND the first pass
    # did not itself select-all (`not ran_select_all`). Review fix #01: a
    # corrupt/schema-mismatched `.testmondata` is warm pre-hygiene but gets
    # purged inside the first pass, which then select-alls; without the
    # `ran_select_all` guard a genuine gap there would fire a pointless rebuild
    # + second full select-all and emit a misleading notice.
    if (
        verdict == _IMPACTED_CHANGED_LINES_UNCOVERED
        and was_incremental
        and not ran_select_all
    ):
        _emit(
            "impacted",
            "rebuilding testmon baseline and re-checking (changed lines <100% on an incremental run)",
        )
        _rebuild_testmon_baseline()
        retry_verdict, _ = _run_impacted_pass(base)
        return 0 if retry_verdict == _IMPACTED_PASS else 1
    return 1


def _write_seed_status(state: str, **fields: object) -> None:
    """Write the `SEED_STATUS` marker atomically (temp sibling + os.replace),
    best-effort (swallow any error so a write failure never aborts the
    detached rebuild), cleaning up the temp file in a `finally`. No `fsync`.
    """
    tmp = SEED_STATUS.with_suffix(SEED_STATUS.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps({"state": state, **fields}), encoding="utf-8")
        os.replace(tmp, SEED_STATUS)
    except Exception as exc:  # noqa: BLE001 — courtesy marker; never abort the rebuild
        # Best-effort still emits one diagnostic line so a marker that never
        # appears is traceable (review-fix #02) — contract unchanged: no raise.
        _emit("seed-testmon", f"warning: could not write status marker ({state}): {exc}")
    finally:
        tmp.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    """Return True if `pid` is a live process (QS-286).

    `os.kill(pid, 0)` sends no signal but validates the target:
    `ProcessLookupError` → the pid is gone (dead); `PermissionError` → the
    pid exists but is owned by another user (alive-but-not-ours → alive);
    success → alive. Extracted as a patchable seam so the status reader's
    `running` branches are coverable without spawning real processes.

    Total by design (review-fix #04): the pid comes from an untrusted marker,
    so an out-of-`pid_t`-range value (`10**19` → `OverflowError`) or any other
    bad input (`ValueError`/`TypeError`) is treated as dead rather than
    escaping and crashing the read-only status query. An out-of-range pid is
    unambiguously not a live process, so "dead" (→ interrupted, exit 1) is the
    correct verdict.
    """
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except (ProcessLookupError, OverflowError, ValueError, TypeError):
        return False
    return True


def seed_testmon() -> int:
    """Refresh `.testmondata` via `pytest --testmon`; no coverage, no verdict.

    Sanctioned non-gate subcommand (see the project-rules raw-`pytest`
    carve-out) used by `finish-task` to rebuild the main-worktree
    baseline after a merge. Returns 0 once the selection completes (a
    failing test still updates the DB — there is no pass/fail verdict);
    returns 3 when testmon is not importable.

    QS-286: writes the `SEED_STATUS` completion marker so the detached
    run (launched by `finish-task`) has a pollable signal — see
    `seed_testmon_status`. The marker writes are the only additions here;
    the best-effort return semantics are unchanged.
    """
    # QS-276 review-fix S2: probe ONLY testmon — seeding never invokes
    # diff-cover, so a missing diff-cover must not block the refresh.
    if not _testmon_available():
        _emit("seed-testmon", "pytest-testmon not importable — skipping")
        # QS-286: record the skip (no `running` marker on this path).
        _write_seed_status("skipped", reason="pytest-testmon not importable")
        return 3
    # QS-283 A3: rebuild from scratch. The old code ran `pytest --testmon`
    # against the EXISTING DB, so an already-advanced baseline yielded
    # "0 changed" and the reseed did nothing — the QS_281 recovery dead end.
    # Purging first (+ clearing accumulated coverage, since seeding
    # establishes the baseline the NEXT `--impacted` accumulates onto) forces
    # a true full re-fingerprint (select-all) and prevents a stale `.coverage`
    # from re-introducing the desync A1 fixes.
    # QS-286: mark the run as started (after the probe, before the pytest
    # pass) so a status query mid-run reports "still running".
    started = time.time()
    _write_seed_status("running", pid=os.getpid(), started=started)
    _rebuild_testmon_baseline()
    _emit("seed-testmon", "refreshing .testmondata (pytest --testmon)")
    result = _stream_pytest(_build_seed_testmon_cmd())
    # QS-276 review-fix NH4 (#03): a pytest exit code >= 2 means a
    # collection error / internal crash (NOT a mere test failure, which is
    # 1 and still updates the DB). Surface it as a warning so a partial /
    # unwritten `.testmondata` is observable — but stay best-effort:
    # finish-task runs this detached and a stale/absent baseline only
    # over-selects, so we still return 0.
    rc = result.get("returncode", 0)
    if rc >= 2:
        _emit("seed-testmon", f"warning: pytest exited {rc} (collection error/crash) — .testmondata may be incomplete")
    # QS-286: overwrite the marker on completion. rc < 2 means testmon wrote
    # `.testmondata` (0 = clean, 1 = failures but DB persisted — "ok" means
    # "baseline written", NOT "tests passed"); rc >= 2 (usage/collection
    # error, crash, or no-tests) means the baseline is suspect → "incomplete".
    # No `pid` in the completion marker: the process is finishing, so the
    # reader's liveness check only ever consumes the `running` marker's pid
    # (review-fix #04).
    _write_seed_status(
        "ok" if rc < 2 else "incomplete",
        started=started,
        finished=time.time(),
        returncode=rc,
    )
    return 0


def _build_seed_testmon_cmd() -> list[str]:
    """Build the `--seed-testmon` baseline-refresh pytest argv.

    Runs the whole suite under `--testmon` to (re)build `.testmondata`;
    no coverage, no verdict. Parallelized with xdist (QS-276 review-fix)
    when `_TESTMON_SUPPORTS_XDIST` is set and workers resolve — seeding
    the baseline is the heaviest testmon pass (a full select-all), so it
    benefits the most from `-n auto`. Honours `QS_QG_PYTEST_WORKERS` via
    the shared `_pytest_workers()` resolver.

    NH1 (review-fix #05): unlike the inner-loop `_build_testmon_cmd`, this
    deliberately does NOT `--ignore tests/test_quality_gate.py`. Seeding is
    the FULL baseline — the testmon self-tests must be fingerprinted too,
    or a later change to the gate would force a select-all. Do not mirror
    the inner-loop `--ignore` here (it would leave the baseline incomplete).
    """
    cmd = [VENV_PYTHON, "-m", "pytest", "--testmon", "-q"]
    if _TESTMON_SUPPORTS_XDIST:
        workers = _pytest_workers()
        if workers is not None:
            cmd.extend(["-n", workers])
    return cmd


def _fmt_seed_time(value: object) -> str:
    """Render a marker epoch timestamp as a readable UTC ISO string (QS-286).

    `started`/`finished` are stored as raw `time.time()` floats; format them
    for display only (the stored marker format is unchanged). A missing or
    non-numeric value (torn/hand-edited marker) yields the readable
    "an unknown time" placeholder rather than a bare float or `None`.

    Non-finite (`inf`/`-inf`/`NaN` — `json.loads` accepts these literals) and
    out-of-range epochs are also placeholdered rather than crashing the
    read-only query: `math.isfinite` rejects the former and the `try/except`
    catches `datetime.fromtimestamp` overflow on a huge finite magnitude
    (review-fix #03).
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            # A huge int raises `OverflowError` in `math.isfinite` itself
            # (caught below), so the finiteness check lives inside the
            # try/except rather than as a short-circuit guard condition.
            if math.isfinite(value):
                return datetime.fromtimestamp(value, tz=UTC).isoformat(timespec="seconds")
        except (OverflowError, ValueError, OSError):
            pass
    return "an unknown time"


def seed_testmon_status() -> int:
    """Report the detached `--seed-testmon` run's status (QS-286).

    Read-only non-gate subcommand — invokes NO pytest, coverage, or testmon
    import; it only stat/reads the `SEED_STATUS` marker and checks PID
    liveness. Prints a human answer and returns a scriptable exit code on a
    deliberately small 4-code contract: 0 = ok / safe to close; 4 = still
    running / keep the terminal open; 1 = not trustworthy, rerun (interrupted,
    incomplete, or skipped); 3 = no readable status (missing or unparseable).

    The reader TRUSTS the marker (best-effort) — it does not cross-check that
    `.testmondata` still exists for an `ok` verdict. It defends against a
    malformed marker (review-fix #01): a non-`dict` payload, an unrecognized
    `state`, or a `running` marker whose `pid` is missing/non-int all route to
    the "unreadable → 3" path rather than crashing, and display-only fields
    that are absent print a readable placeholder instead of `None`.
    """
    def _unreadable() -> int:
        print("The baseline refresh status is unreadable.")
        return 3

    if not SEED_STATUS.exists():
        print("No baseline refresh has been recorded.")
        return 3
    try:
        marker = json.loads(SEED_STATUS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _unreadable()
    # A parseable-but-non-object payload (`[1]`, `5`, `null`, `"x"`) has no
    # `.get`; treat it as unreadable rather than letting `.get` raise.
    if not isinstance(marker, dict):
        return _unreadable()

    state = marker.get("state")
    if state == "ok":
        print(
            f"Baseline written at {_fmt_seed_time(marker.get('finished'))} "
            "(tests may have failed) — safe to close this terminal."
        )
        return 0
    if state == "running":
        pid = marker.get("pid")
        # `_pid_alive` calls `os.kill(pid, 0)`; a `running` marker whose pid is
        # not a positive, non-bool int is unreadable. Excluding bool (an int
        # subclass) and pid <= 0 is deliberate: `os.kill(0, 0)` targets the
        # caller's process group and `os.kill(-1, 0)` every signalable process
        # — both would spuriously report "still running" (review-fix #02).
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            return _unreadable()
        if _pid_alive(pid):
            print(
                f"Still running (pid {pid}, started {_fmt_seed_time(marker.get('started'))}) "
                "— keep this terminal open."
            )
            return 4
        print(
            f"Interrupted (pid {pid} no longer running) — `.testmondata` "
            "may be partial; rerun the refresh."
        )
        return 1
    if state == "incomplete":
        print(
            f"Finished with errors (exit {marker.get('returncode', 'unknown')}) — "
            "`.testmondata` may be partial; rerun the refresh."
        )
        return 1
    if state == "skipped":
        print(
            f"Refresh was skipped ({marker.get('reason', 'unknown reason')}) — no baseline was "
            "written; rerun if needed."
        )
        return 1
    # Any other/absent state value is an unreadable status.
    return _unreadable()


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
        help=(
            "Refresh .testmondata via `pytest --testmon` (no coverage, no verdict). "
            "Mutex with --impacted/--quick/--cache/--no-cache/--full/--fix."
        ),
    )
    parser.add_argument(
        "--seed-testmon-status",
        action="store_true",
        help=(
            "Report the detached --seed-testmon run's completion status "
            "(read-only; no pytest). Exit 0=ok/safe to close, 4=still running, "
            "1=rerun, 3=no readable status. Mutex with the same modes as "
            "--seed-testmon."
        ),
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

    # --seed-testmon (QS-276 review-fix M1) is a self-contained maintenance
    # subcommand, not a gate mode: combining it with any execution mode
    # silently dropped the seeding request (the main() dispatch order
    # short-circuited --impacted before it, and it before cache/scope/full).
    # Make the conflict an explicit usage error instead.
    if args.seed_testmon and (
        args.impacted or args.quick or args.cache or args.no_cache or args.full or args.fix
    ):
        parser.error(
            "you cannot combine --seed-testmon with --impacted, --quick, --cache, "
            "--no-cache, --full, or --fix"
        )

    # QS-286: --seed-testmon-status is a read-only query subcommand, not a
    # gate mode. Mirror the --seed-testmon mutex so combining it with any
    # execution mode (or the seed itself) is an explicit usage error.
    if args.seed_testmon_status and (
        args.impacted
        or args.quick
        or args.cache
        or args.no_cache
        or args.full
        or args.fix
        or args.seed_testmon
    ):
        parser.error(
            "you cannot combine --seed-testmon-status with --impacted, --quick, "
            "--cache, --no-cache, --full, --fix, or --seed-testmon"
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

    # QS-286: --seed-testmon-status is a read-only query — short-circuit
    # before cache/scope detection like the other subcommands.
    if args.seed_testmon_status:
        sys.exit(seed_testmon_status())

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
