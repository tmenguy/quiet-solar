#!/usr/bin/env python3
"""Run all quality gates and report results.

Usage:
    python scripts/qs/quality_gate.py [--fix] [--json] [--cache] [--no-cache]

Options:
    --fix       Auto-fix what can be fixed (ruff format, ruff check --fix)
    --json      Output JSON instead of human-readable text
    --cache     Enable caching — skip gates if git state matches a previous pass
    --no-cache  Force fresh run even when --cache is present

Exit codes:
    0 = all gates pass
    1 = one or more gates failed
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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


def check_pytest() -> dict:
    """Run pytest with 100% coverage check."""
    cmd = [
        VENV_PYTHON, "-m", "pytest", str(TESTS_DIR),
        f"--cov={SRC_DIR}",
        "--cov-report=term-missing",
        "--cov-fail-under=100",
        "-q",
    ]
    result = _run(cmd)
    passed = result.returncode == 0
    # Extract coverage percentage from output
    coverage = None
    missing_lines = []
    for line in result.stdout.split("\n"):
        if "TOTAL" in line and "%" in line:
            parts = line.split()
            for p in parts:
                if p.endswith("%"):
                    coverage = p
        if "FAIL" in line and "%" in line:
            # Lines like: custom_components/quiet_solar/foo.py 50 2 96%  10, 15
            missing_lines.append(line.strip())
    return {
        "name": "pytest",
        "passed": passed,
        "coverage": coverage,
        "missing": missing_lines[:10],  # Cap for brevity
        "detail": result.stdout[-500:] if not passed else "",
        "stderr": result.stderr[-300:] if not passed else "",
    }


def check_ruff_lint(fix: bool = False) -> dict:
    """Run ruff check."""
    cmd = [_venv_tool("ruff"), "check", str(SRC_DIR)]
    if fix:
        cmd.append("--fix")
    result = _run(cmd)
    return {
        "name": "ruff_lint",
        "passed": result.returncode == 0,
        "detail": result.stdout.strip()[-500:] if result.returncode != 0 else "",
    }


def check_ruff_format(fix: bool = False) -> dict:
    """Run ruff format check."""
    if fix:
        cmd = [_venv_tool("ruff"), "format", str(SRC_DIR)]
    else:
        cmd = [_venv_tool("ruff"), "format", "--check", str(SRC_DIR)]
    result = _run(cmd)
    return {
        "name": "ruff_format",
        "passed": result.returncode == 0,
        "detail": result.stdout.strip()[-500:] if result.returncode != 0 else "",
    }


def check_mypy() -> dict:
    """Run mypy."""
    cmd = [VENV_PYTHON, "-m", "mypy", str(SRC_DIR)]
    result = _run(cmd)
    passed = result.returncode == 0
    return {
        "name": "mypy",
        "passed": passed,
        "detail": result.stdout.strip()[-500:] if not passed else "",
    }


def check_translations() -> dict:
    """Check if translations need regeneration."""
    if not STRINGS_JSON.exists():
        return {"name": "translations", "passed": True, "detail": "no strings.json"}

    gen_script = REPO_ROOT / "scripts" / "generate-translations.sh"
    if not gen_script.exists():
        return {"name": "translations", "passed": True, "detail": "no generate script"}

    # Capture current en.json content
    en_before = TRANSLATIONS_EN.read_text() if TRANSLATIONS_EN.exists() else ""

    # Run generation — ignore exit code (may fail for unrelated integrations)
    result = _run(["bash", str(gen_script)])

    # Check if our translations file changed (the only thing that matters)
    en_after = TRANSLATIONS_EN.read_text() if TRANSLATIONS_EN.exists() else ""
    if en_before != en_after:
        return {
            "name": "translations",
            "passed": False,
            "detail": "translations/en.json was outdated and has been regenerated. Stage the updated file.",
        }

    return {"name": "translations", "passed": True, "detail": ""}


def _output_results(
    results: list[dict],
    *,
    all_passed: bool,
    cached: bool,
    json_mode: bool,
) -> None:
    """Print gate results in human-readable or JSON format."""
    if json_mode:
        print(json.dumps({
            "all_passed": all_passed,
            "cached": cached,
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

    # Run all gates fresh
    results = []
    results.append(check_ruff_format(fix=args.fix))
    results.append(check_ruff_lint(fix=args.fix))
    results.append(check_mypy())
    results.append(check_translations())
    results.append(check_pytest())

    all_passed = all(r["passed"] for r in results)

    # Write cache on pass (only when caching is enabled)
    if use_cache and all_passed:
        branch, commit, is_clean = _get_git_state()
        if is_clean:
            _write_cache(branch, commit, results)

    _output_results(results, all_passed=all_passed, cached=False, json_mode=args.json)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
