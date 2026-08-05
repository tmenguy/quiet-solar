"""QS-332: pins on the workflow-bot fixes (B4) and the lane-check CI job (B1).

``stale.yml`` / ``issue-triage.yml`` were previously pinned by nothing;
these tests pin the D6 decisions so a later edit is a deliberate act.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def _load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# stale.yml — scale:epic exemption (B4)
# ---------------------------------------------------------------------------


def test_stale_exempts_epics() -> None:
    doc = _load("stale.yml")
    step = doc["jobs"]["stale"]["steps"][0]
    issue_exemptions = step["with"]["exempt-issue-labels"].split(",")
    assert "scale:epic" in issue_exemptions
    # The pre-existing exemptions survive.
    assert {"pinned", "security", "critical"} <= set(issue_exemptions)
    # PR exemptions unchanged — epics have no PRs.
    assert step["with"]["exempt-pr-labels"] == "pinned,security,critical"


# ---------------------------------------------------------------------------
# issue-triage.yml — the D6 per-keyword table (#293)
# ---------------------------------------------------------------------------


@pytest.fixture(name="triage_script")
def triage_script_fixture() -> str:
    doc = _load("issue-triage.yml")
    return doc["jobs"]["triage"]["steps"][0]["with"]["script"]


@pytest.mark.parametrize(
    ("label", "patterns"),
    [
        ("area:solver", [r"/\bsolver\b/", r"/\bconstraints?\b/"]),
        ("area:charger", [r"/\bchargers?\b/", r"/\bocpp\b/", r"/\bwallbox\b/"]),
        ("area:car", [r"/\bcars?\b/", r"/\bvehicles?\b/", r"/\bev\b/"]),
        ("area:battery", [r"/\bbatter(y|ies)\b/", r"/\bstorage\b/"]),
        ("area:person", [r"/\bpersons?\b/", r"/\bpresence\b/"]),
        ("area:ui", [r"/\bdashboards?\b/", r"/\bui\b/"]),
        ("area:config", [r"/\bconfig[ _-]?flow\b/", r"/\bconfig entr(y|ies)\b/"]),
        (
            "area:dev-pipeline",
            [
                r"/\b(harness|pipeline|agents?|workflow|worktree|quality gate|lanes?|launcher|testmon)\b/"
            ],
        ),
    ],
)
def test_triage_implements_the_d6_table(
    triage_script: str, label: str, patterns: list[str]
) -> None:
    assert f"'{label}'" in triage_script
    for pattern in patterns:
        assert pattern in triage_script, f"{label} must match via {pattern}"


def test_triage_has_no_naive_substring_matching(triage_script: str) -> None:
    """The #293 bug class: `includes('ev')` fired on "dev", `includes('ui')`
    on "quiet", `includes('config')`/`includes('setup')` on every factory
    issue. Substring matching is banned wholesale."""
    assert ".includes(" not in triage_script


def test_triage_dropped_keywords_are_gone(triage_script: str) -> None:
    # Bare 'config' and 'setup' dropped; 'configuration' (R2-07: still
    # matched "harness configuration") never reintroduced.
    assert not re.search(r"/\\bconfig\\b/", triage_script)
    assert "setup" not in triage_script
    assert "configuration" not in triage_script


# ---------------------------------------------------------------------------
# pr-quality.yml — the lane-check job (B1, reviews CR-3 + N-1)
# ---------------------------------------------------------------------------


def test_lane_check_job_spec() -> None:
    doc = _load("pr-quality.yml")
    job = doc["jobs"]["lane-check"]
    # Own least-privilege permissions block; the OTHER jobs stay
    # `contents: read`-only — this one needs the labels.
    assert job["permissions"] == {"contents": "read", "issues": "read"}

    checkout = job["steps"][0]
    assert "actions/checkout" in checkout["uses"]
    # The change set is origin/main...HEAD — must reach the merge base.
    assert checkout["with"]["fetch-depth"] == 0

    # Review-fix #01 must-fix: the job must pin the SAME interpreter as
    # every other job — ubuntu-latest's system Python is 3.12 and
    # quality_gate.py uses 3.14-only syntax, so without this step the
    # fail-closed job blocks every PR with a SyntaxError.
    setup_python = next(
        s for s in job["steps"] if "setup-python" in s.get("uses", "")
    )
    assert setup_python["with"]["python-version"] == "3.14"

    run_step = next(s for s in job["steps"] if "run" in s)
    # `gh` reads GH_TOKEN; preinstalled on ubuntu-latest, no install step.
    assert run_step["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert "quality_gate.py --lane-check" in run_step["run"]
    # Detached-HEAD fallback (review N-1): a pull_request checkout has an
    # empty `git branch --show-current`; GITHUB_HEAD_REF is the PR's
    # source branch on every pull_request event. Review-fix #01 must-fix:
    # `head_ref` is attacker-controlled on fork PRs and Actions expands
    # `${{ }}` BEFORE the shell sees the line — script injection. It must
    # reach the shell through an env var, never direct interpolation.
    assert run_step["env"]["HEAD_REF"] == "${{ github.head_ref }}"
    assert '--branch "$HEAD_REF"' in run_step["run"]
    assert "${{" not in run_step["run"]


def test_other_pr_quality_jobs_keep_read_only_tokens() -> None:
    """The lane check must not have been grafted onto an existing job."""
    doc = _load("pr-quality.yml")
    for name, job in doc["jobs"].items():
        if name == "lane-check":
            continue
        assert "issues" not in job.get("permissions", {}), (
            f"job {name} must not hold an issues-scoped token"
        )
        assert "--lane-check" not in str(job)
