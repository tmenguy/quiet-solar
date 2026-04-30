#!/usr/bin/env python3
"""Run all quality gates and report results.

Usage:
    python scripts/qs/quality_gate.py [--fix] [--json] [--cache] [--no-cache] [--full]

Options:
    --fix       Auto-fix what can be fixed (ruff format, ruff check --fix)
    --json      Output JSON instead of human-readable text
    --cache     Enable caching — skip gates if git state matches a previous pass
    --no-cache  Force fresh run even when --cache is present
    --full      Force full gate run even if only dev/test files changed

Smart scope detection:
    When only dev-infrastructure files are modified (tests/, _qsprocess*/, docs/,
    scripts/, *.md, .claude/, .cursor/, .opencode/), the quality gate skips the
    full suite (ruff, mypy, translations, full pytest+coverage) and only runs
    the modified test files. Use --full to override.

Exit codes:
    0 = all gates pass
    1 = one or more gates failed
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

# Resolve paths relative to repo root
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "custom_components" / "quiet_solar"
TESTS_DIR = REPO_ROOT / "tests"
VENV_BIN = REPO_ROOT / "venv" / "bin"
STRINGS_JSON = SRC_DIR / "strings.json"
TRANSLATIONS_EN = SRC_DIR / "translations" / "en.json"


VENV_PYTHON = str(VENV_BIN / "python")

# Patterns for files that are "dev-only" (no production code impact)
_DEV_ONLY_PATTERNS = (
    "tests/",
    "_qsprocess/",
    "_qsprocess_opencode/",
    "_bmad-output/",
    "scripts/",
    "docs/",
    ".claude/",
    ".cursor/",
    ".opencode/",
)
_DEV_ONLY_EXTENSIONS = (".md",)


def _venv_tool(name: str) -> str:
    """Return the absolute path to a tool inside the venv."""
    return str(VENV_BIN / name)


def _run(cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or str(REPO_ROOT))


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
        "CLAUDE.md", "AGENTS.md", ".cursorrules", ".gitignore",
        "pyproject.toml", "setup.cfg", "requirements.txt",
        "requirements_test.txt",
    ):
        return True
    return False


def _detect_scope(changed_files: list[str]) -> dict:
    """Determine which gates to run based on changed files.

    Returns dict with:
        scope: "full" | "dev-only"
        changed_test_files: list of test files that changed (if dev-only)
        reason: human-readable explanation
    """
    if not changed_files:
        return {"scope": "full", "changed_test_files": [], "reason": "no changes detected, running full"}

    non_dev = [f for f in changed_files if not _is_dev_only(f)]
    if non_dev:
        return {
            "scope": "full",
            "changed_test_files": [],
            "reason": f"production files changed: {', '.join(non_dev[:5])}",
        }

    test_files = [f for f in changed_files if f.startswith("tests/") and f.endswith(".py")]
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
    except (json.JSONDecodeError, OSError):
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
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    CACHE_FILE.write_text(json.dumps(data, indent=2))


def _is_cache_valid(cache: dict | None, branch: str, commit: str, is_clean: bool) -> bool:
    """Check if cached results match the current git state."""
    if cache is None:
        return False
    if not is_clean:
        return False
    return cache.get("branch") == branch and cache.get("commit") == commit


def _stream_pytest(cmd: list[str]) -> dict:
    """Run pytest with real-time progress reporting.

    Prints a progress line every ~10% of tests. Returns the same dict
    format as check_pytest().
    """
    # First, collect test count
    count_cmd = [VENV_PYTHON, "-m", "pytest", "--collect-only", "-q", str(TESTS_DIR)]
    count_result = _run(count_cmd)
    total_tests = 0
    for line in count_result.stdout.split("\n"):
        m = re.match(r"(\d+) tests? collected", line.strip())
        if m:
            total_tests = int(m.group(1))
            break

    milestone_every = max(1, total_tests // 10) if total_tests > 0 else 50

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(REPO_ROOT),
    )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    passed_count = 0
    failed_count = 0
    error_count = 0
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
        stripped = line.strip()

        # Count test results from -q output lines like ".....F..x.."
        # Each char is a test result: . = pass, F = fail, E = error, s = skip, x = xfail
        if stripped and all(c in ".FEsxX" for c in stripped) and len(stripped) > 0:
            passed_count += stripped.count(".")
            failed_count += stripped.count("F")
            error_count += stripped.count("E")
            current = passed_count + failed_count + error_count
            if total_tests > 0 and current >= last_milestone + milestone_every:
                pct = min(100, int(current / total_tests * 100))
                sys.stderr.write(
                    f"  pytest: {pct}% ({current}/{total_tests})"
                    f" | passed={passed_count} failed={failed_count} errors={error_count}\n"
                )
                sys.stderr.flush()
                last_milestone = current

    proc.wait()
    stderr_thread.join(timeout=5)

    # Final count line
    if total_tests > 0:
        current = passed_count + failed_count + error_count
        sys.stderr.write(
            f"  pytest: done ({current}/{total_tests})"
            f" | passed={passed_count} failed={failed_count} errors={error_count}\n"
        )
        sys.stderr.flush()

    stdout_text = "".join(stdout_lines)
    stderr_text = "".join(stderr_lines)

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
    """Run pytest with 100% coverage check and progress reporting."""
    cmd = [
        VENV_PYTHON, "-m", "pytest", str(TESTS_DIR),
        f"--cov={SRC_DIR}",
        "--cov-report=term-missing",
        "--cov-fail-under=100",
        "-q",
    ]
    return _stream_pytest(cmd)


def check_pytest_files(test_files: list[str]) -> dict:
    """Run pytest on specific test files only (no coverage)."""
    if not test_files:
        return {"name": "pytest (no test files changed)", "passed": True, "detail": ""}

    abs_files = [str(REPO_ROOT / f) for f in test_files]
    cmd = [VENV_PYTHON, "-m", "pytest", *abs_files, "-q"]
    return _stream_pytest(cmd)


def check_ruff_lint(fix: bool = False) -> dict:
    """Run ruff check."""
    sys.stderr.write("  ruff lint: running...\n")
    sys.stderr.flush()
    cmd = [_venv_tool("ruff"), "check", str(SRC_DIR)]
    if fix:
        cmd.append("--fix")
    result = _run(cmd)
    status = "PASS" if result.returncode == 0 else "FAIL"
    sys.stderr.write(f"  ruff lint: {status}\n")
    sys.stderr.flush()
    return {
        "name": "ruff_lint",
        "passed": result.returncode == 0,
        "detail": result.stdout.strip()[-500:] if result.returncode != 0 else "",
    }


def check_ruff_format(fix: bool = False) -> dict:
    """Run ruff format check."""
    sys.stderr.write("  ruff format: running...\n")
    sys.stderr.flush()
    if fix:
        cmd = [_venv_tool("ruff"), "format", str(SRC_DIR)]
    else:
        cmd = [_venv_tool("ruff"), "format", "--check", str(SRC_DIR)]
    result = _run(cmd)
    status = "PASS" if result.returncode == 0 else "FAIL"
    sys.stderr.write(f"  ruff format: {status}\n")
    sys.stderr.flush()
    return {
        "name": "ruff_format",
        "passed": result.returncode == 0,
        "detail": result.stdout.strip()[-500:] if result.returncode != 0 else "",
    }


def check_mypy() -> dict:
    """Run mypy."""
    sys.stderr.write("  mypy: running...\n")
    sys.stderr.flush()
    cmd = [VENV_PYTHON, "-m", "mypy", str(SRC_DIR)]
    result = _run(cmd)
    passed = result.returncode == 0
    status = "PASS" if passed else "FAIL"
    sys.stderr.write(f"  mypy: {status}\n")
    sys.stderr.flush()
    return {
        "name": "mypy",
        "passed": passed,
        "detail": result.stdout.strip()[-500:] if not passed else "",
    }


def check_translations() -> dict:
    """Check if translations need regeneration."""
    sys.stderr.write("  translations: checking...\n")
    sys.stderr.flush()
    if not STRINGS_JSON.exists():
        sys.stderr.write("  translations: SKIP (no strings.json)\n")
        sys.stderr.flush()
        return {"name": "translations", "passed": True, "detail": "no strings.json"}

    gen_script = REPO_ROOT / "scripts" / "generate-translations.sh"
    if not gen_script.exists():
        sys.stderr.write("  translations: SKIP (no generate script)\n")
        sys.stderr.flush()
        return {"name": "translations", "passed": True, "detail": "no generate script"}

    # Capture current en.json content
    en_before = TRANSLATIONS_EN.read_text() if TRANSLATIONS_EN.exists() else ""

    # Run generation — ignore exit code (may fail for unrelated integrations)
    result = _run(["bash", str(gen_script)])

    # Check if our translations file changed (the only thing that matters)
    en_after = TRANSLATIONS_EN.read_text() if TRANSLATIONS_EN.exists() else ""
    if en_before != en_after:
        sys.stderr.write("  translations: FAIL (outdated)\n")
        sys.stderr.flush()
        return {
            "name": "translations",
            "passed": False,
            "detail": "translations/en.json was outdated and has been regenerated. Stage the updated file.",
        }

    sys.stderr.write("  translations: PASS\n")
    sys.stderr.flush()
    return {"name": "translations", "passed": True, "detail": ""}


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
        print(json.dumps({
            "all_passed": all_passed,
            "cached": cached,
            "scope": scope,
            "gates": results,
        }, indent=2))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quality gates")
    parser.add_argument("--fix", action="store_true", help="Auto-fix formatting/lint issues")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--cache", action="store_true", help="Enable result caching based on git state")
    parser.add_argument("--no-cache", action="store_true", help="Force fresh run (overrides --cache)")
    parser.add_argument("--full", action="store_true", help="Force full gate run regardless of scope")
    args = parser.parse_args()

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
    else:
        sys.stderr.write(f"  scope: FULL ({scope_info['reason']})\n")
        sys.stderr.flush()

        results = []
        results.append(check_ruff_format(fix=args.fix))
        results.append(check_ruff_lint(fix=args.fix))
        results.append(check_mypy())
        results.append(check_translations())
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
