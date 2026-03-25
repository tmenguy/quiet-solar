#!/usr/bin/env python3
"""Run all quality gates and report results.

Usage:
    python scripts/qs/quality_gate.py [--fix] [--json]

Options:
    --fix   Auto-fix what can be fixed (ruff format, ruff check --fix)
    --json  Output JSON instead of human-readable text

Exit codes:
    0 = all gates pass
    1 = one or more gates failed
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Resolve paths relative to repo root
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "custom_components" / "quiet_solar"
TESTS_DIR = REPO_ROOT / "tests"
VENV_ACTIVATE = REPO_ROOT / "venv" / "bin" / "activate"
STRINGS_JSON = SRC_DIR / "strings.json"
TRANSLATIONS_EN = SRC_DIR / "translations" / "en.json"


def _run(cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or str(REPO_ROOT))


def check_pytest() -> dict:
    """Run pytest with 100% coverage check."""
    cmd = [
        sys.executable, "-m", "pytest", str(TESTS_DIR),
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
    cmd = ["ruff", "check", str(SRC_DIR)]
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
        cmd = ["ruff", "format", str(SRC_DIR)]
    else:
        cmd = ["ruff", "format", "--check", str(SRC_DIR)]
    result = _run(cmd)
    return {
        "name": "ruff_format",
        "passed": result.returncode == 0,
        "detail": result.stdout.strip()[-500:] if result.returncode != 0 else "",
    }


def check_mypy() -> dict:
    """Run mypy."""
    cmd = [sys.executable, "-m", "mypy", str(SRC_DIR)]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quality gates")
    parser.add_argument("--fix", action="store_true", help="Auto-fix formatting/lint issues")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    results = []

    # Run format first if --fix (changes might affect lint)
    results.append(check_ruff_format(fix=args.fix))
    results.append(check_ruff_lint(fix=args.fix))
    results.append(check_mypy())
    results.append(check_translations())
    results.append(check_pytest())

    all_passed = all(r["passed"] for r in results)

    if args.json:
        print(json.dumps({
            "all_passed": all_passed,
            "gates": results,
        }, indent=2))
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

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
