"""Pin the static-agent contract for the OpenCode harness.

Three tests:

- ``test_static_opencode_agent_files_present_and_valid`` — every
  phase + sub-agent has a ``.opencode/agents/qs-<name>.md`` file with
  valid YAML frontmatter declaring ``description``, ``mode``,
  ``color``, ``permission``. Sub-agents additionally declare
  ``mode: subagent`` and ``hidden: True``.
- ``test_no_legacy_path_references_in_opencode_agent_bodies`` — no
  body mentions ``_qsprocess_opencode``, ``scripts/qs_opencode``, or
  ``render_agent.py``.
- ``test_dev_only_patterns_contains_legacy_not_qsprocess`` — pins
  the ``_DEV_ONLY_PATTERNS`` swap performed in Task 5.

The bash-allowlist tier per agent (AC2) is NOT pinned by a test —
per-agent variation would make the test brittle; AC2 is the
specification, the agent files are the implementation.

Note: ``yaml`` (PyYAML) is transitively available via
``homeassistant>=2026.4.2``; no extra requirements_test entry needed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
OPENCODE_AGENTS_DIR = REPO_ROOT / ".opencode" / "agents"

# ``@pytest.mark.parametrize`` evaluates its argument list at collection
# time — before the autouse ``_add_scripts_qs_to_syspath`` fixture in
# ``tests/qs/conftest.py`` fires — so we re-pin ``scripts/qs/`` onto
# ``sys.path`` at module load to make ``from launchers.phases import
# PHASE_TO_AGENT`` resolvable inside ``_all_static_agent_names()``. The
# insert is idempotent; the fixture's later run is a no-op.
_SCRIPTS_QS_DIR = str(REPO_ROOT / "scripts" / "qs")
if _SCRIPTS_QS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_QS_DIR)

SUB_AGENT_NAMES: tuple[str, ...] = (
    "qs-plan-critic",
    "qs-plan-concrete-planner",
    "qs-plan-dev-proxy",
    "qs-plan-scope-guardian",
    "qs-review-blind-hunter",
    "qs-review-edge-case-hunter",
    "qs-review-acceptance-auditor",
    "qs-review-coderabbit",
)

# Pattern that catches every "active reference to legacy machinery"
# the migration scrubs. Anchored on three distinct tokens so a future
# rename that affects only one of them still trips the test.
_LEGACY_REFERENCE_RE = re.compile(
    r"_qsprocess_opencode|scripts/qs_opencode|render_agent\.py",
)


def _all_static_agent_names() -> tuple[str, ...]:
    """Phase orchestrators (7) + sub-agents (8) = 15 static agents."""
    from launchers.phases import PHASE_TO_AGENT  # type: ignore[import-not-found]

    return tuple(PHASE_TO_AGENT.values()) + SUB_AGENT_NAMES


def _parse_frontmatter(path: Path) -> tuple[dict, str]:
    """Split an agent file into ``(frontmatter_dict, body_text)``.

    Raises ``AssertionError`` on a malformed file — the parametrized
    happy-path tests are the only callers, and any malformed file is
    a test failure by construction.
    """
    text = path.read_text()
    assert text.startswith("---\n"), f"{path} missing opening ``---``"
    rest = text[len("---\n") :]
    end = rest.index("\n---\n")
    return yaml.safe_load(rest[:end]), rest[end + len("\n---\n") :]


@pytest.mark.parametrize("name", _all_static_agent_names())
def test_static_opencode_agent_files_present_and_valid(name: str) -> None:
    path = OPENCODE_AGENTS_DIR / f"{name}.md"
    assert path.is_file(), f"Missing static agent file: {path}"

    fm, _ = _parse_frontmatter(path)
    assert isinstance(fm, dict), f"{path} frontmatter is not a mapping"
    for key in ("description", "mode", "color", "permission"):
        assert key in fm, f"{path} frontmatter missing {key!r}"

    if name in SUB_AGENT_NAMES:
        assert fm["mode"] == "subagent", f"{path} should be mode: subagent"
        assert fm.get("hidden") is True, f"{path} should be hidden: true"
    else:
        assert fm["mode"] == "primary", f"{path} should be mode: primary"


@pytest.mark.parametrize("name", _all_static_agent_names())
def test_no_legacy_path_references_in_opencode_agent_bodies(name: str) -> None:
    path = OPENCODE_AGENTS_DIR / f"{name}.md"
    _, body = _parse_frontmatter(path)
    match = _LEGACY_REFERENCE_RE.search(body)
    assert match is None, (
        f"{path} body still references the legacy pipeline at {match!r}; "
        "scrub the reference or move the file to legacy/."
    )


def test_dev_only_patterns_contains_legacy_not_qsprocess() -> None:
    from quality_gate import _DEV_ONLY_PATTERNS  # type: ignore[import-not-found]

    assert "legacy/" in _DEV_ONLY_PATTERNS
    assert "_qsprocess_opencode/" not in _DEV_ONLY_PATTERNS
    assert "scripts/qs_opencode/" not in _DEV_ONLY_PATTERNS
