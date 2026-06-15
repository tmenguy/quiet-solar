"""Pin the static-agent contract for the OpenCode harness.

Five tests (three contract assertions + two helper-robustness checks +
one launchers-sync validator):

- ``test_static_opencode_agent_files_present_and_valid`` — every
  phase + sub-agent has a ``.opencode/agents/qs-<name>.md`` file with
  valid YAML frontmatter declaring ``description``, ``mode``,
  ``color``, ``permission``. Sub-agents additionally declare
  ``mode: subagent`` and ``hidden: True``.
- ``test_no_legacy_path_references_in_opencode_agent_bodies`` — no
  body mentions any of the migration's three forbidden tokens
  (``_qsprocess_opencode``, ``scripts/qs_opencode``,
  ``render_agent.py``). The test is strict by design (review-fix #01
  N7): if a future agent legitimately needs to discuss legacy paths
  in prose, the right home for that prose is ``legacy/`` itself or
  ``docs/workflow/``, NOT ``.opencode/agents/``.
- ``test_dev_only_patterns_contains_legacy_not_qsprocess`` — pins
  the ``_DEV_ONLY_PATTERNS`` swap performed in Task 5.
- ``test_phase_agent_names_match_launchers`` — review-fix #01 S5:
  the hardcoded ``PHASE_AGENT_NAMES`` tuple (used at parametrize
  collection time) stays in sync with
  ``launchers.phases.PHASE_TO_AGENT``. Imports inside the test body
  so a launchers refactor surfaces as a single clean test failure
  instead of a collection-time abort.
- ``test_parse_frontmatter_handles_crlf_and_missing_delimiter`` —
  review-fix #01 S3 + S4: ``_parse_frontmatter`` normalizes CRLF
  line endings and raises ``AssertionError`` (not the bare
  ``ValueError`` from ``str.index``) when the closing ``---``
  delimiter is missing.

The bash-allowlist tier per agent (AC2) is NOT pinned by a test —
per-agent variation would make the test brittle; AC2 is the
specification, the agent files are the implementation.

Note: ``yaml`` (PyYAML) is transitively available via
``homeassistant>=2026.4.2``; no extra requirements_test entry needed.

The autouse ``_add_scripts_qs_to_syspath`` fixture in
``tests/qs/conftest.py`` cascades into this subdirectory, so any
``from launchers.* import ...`` inside a test BODY is resolvable
without a module-level ``sys.path`` mutation (review-fix #01 N2 —
removed the previous module-load-time insert).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.qs.agents.const import (
    DEV_ONLY_PATTERN_FORBIDDEN,
    DEV_ONLY_PATTERN_LEGACY,
    LEGACY_REFERENCE_RE,
    REQUIRED_FRONTMATTER_KEYS,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OPENCODE_AGENTS_DIR = REPO_ROOT / ".opencode" / "agents"

# Hardcoded at module level so ``@pytest.mark.parametrize`` works during
# collection even if ``launchers.phases`` is temporarily broken
# (review-fix #01 S5). ``test_phase_agent_names_match_launchers``
# guards against drift between this tuple and the launchers source of
# truth.
PHASE_AGENT_NAMES: tuple[str, ...] = (
    "qs-setup-task",
    "qs-create-plan",
    "qs-implement-task",
    "qs-implement-setup-task",
    "qs-review-task",
    "qs-finish-task",
    "qs-release",
)

SUB_AGENT_NAMES: tuple[str, ...] = (
    "qs-plan-critic",
    "qs-plan-concrete-planner",
    "qs-plan-dev-proxy",
    "qs-plan-scope-guardian",
    "qs-plan-delta-auditor",
    "qs-review-blind-hunter",
    "qs-review-edge-case-hunter",
    "qs-review-acceptance-auditor",
    "qs-review-coderabbit",
)

ALL_STATIC_AGENT_NAMES: tuple[str, ...] = PHASE_AGENT_NAMES + SUB_AGENT_NAMES


def _parse_frontmatter(path: Path) -> tuple[dict, str]:
    """Split an agent file into ``(frontmatter_dict, body_text)``.

    Normalizes CRLF line endings before scanning (review-fix #01 S3) so
    a Windows-saved file parses identically to a Unix-saved one. Raises
    ``AssertionError`` with a useful message on a malformed file
    (review-fix #01 S4); ``str.find`` is used in place of
    ``str.index`` to keep the exception type consistent with the
    docstring contract.
    """
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert text.startswith("---\n"), f"{path}: missing opening ``---`` delimiter"
    rest = text[len("---\n") :]
    end = rest.find("\n---\n")
    assert end != -1, f"{path}: missing closing ``---`` delimiter"
    return yaml.safe_load(rest[:end]), rest[end + len("\n---\n") :]


@pytest.mark.parametrize("name", ALL_STATIC_AGENT_NAMES)
def test_static_opencode_agent_files_present_and_valid(name: str) -> None:
    path = OPENCODE_AGENTS_DIR / f"{name}.md"
    assert path.is_file(), f"Missing static agent file: {path}"

    fm, _ = _parse_frontmatter(path)
    assert isinstance(fm, dict), f"{path} frontmatter is not a mapping"
    for key in REQUIRED_FRONTMATTER_KEYS:
        assert key in fm, f"{path} frontmatter missing {key!r}"

    if name in SUB_AGENT_NAMES:
        assert fm["mode"] == "subagent", f"{path} should be mode: subagent"
        hidden = fm.get("hidden")
        # Strict identity check: ``hidden`` must be the YAML boolean
        # ``true`` after ``yaml.safe_load``. Failure message explains
        # the YAML quirk so a maintainer who wrote ``hidden: yes``
        # (which safe_load returns as ``True``, but ``hidden: "yes"``
        # returns as the string) gets a clear hint (review-fix #01 N6).
        assert hidden is True, (
            f"{path}: 'hidden' must be the YAML boolean ``true`` after "
            f"safe_load; got {type(hidden).__name__}={hidden!r}. If you "
            f"used ``hidden: yes`` quoted, drop the quotes."
        )
    else:
        assert fm["mode"] == "primary", f"{path} should be mode: primary"


@pytest.mark.parametrize("name", ALL_STATIC_AGENT_NAMES)
def test_no_legacy_path_references_in_opencode_agent_bodies(name: str) -> None:
    path = OPENCODE_AGENTS_DIR / f"{name}.md"
    _, body = _parse_frontmatter(path)
    match = LEGACY_REFERENCE_RE.search(body)
    assert match is None, (
        f"{path} body still references the legacy pipeline at {match!r}; "
        "scrub the reference or move the file to legacy/."
    )


def test_dev_only_patterns_contains_legacy_not_qsprocess() -> None:
    from quality_gate import _DEV_ONLY_PATTERNS  # type: ignore[import-not-found]

    assert DEV_ONLY_PATTERN_LEGACY in _DEV_ONLY_PATTERNS
    for forbidden in DEV_ONLY_PATTERN_FORBIDDEN:
        assert forbidden not in _DEV_ONLY_PATTERNS, (
            f"_DEV_ONLY_PATTERNS still contains pre-migration pattern {forbidden!r}"
        )


def test_phase_agent_names_match_launchers() -> None:
    """Pin the hardcoded ``PHASE_AGENT_NAMES`` against the launchers source.

    Imports inside the body so a temporary launchers-package breakage
    fails this single test cleanly rather than aborting collection
    for the entire module (review-fix #01 S5).
    """
    from launchers.phases import PHASE_TO_AGENT  # type: ignore[import-not-found]

    assert tuple(PHASE_TO_AGENT.values()) == PHASE_AGENT_NAMES, (
        "PHASE_AGENT_NAMES drifted from launchers.phases.PHASE_TO_AGENT — "
        "update one to match the other."
    )


def test_parse_frontmatter_handles_crlf_and_missing_delimiter(tmp_path: Path) -> None:
    """``_parse_frontmatter`` is robust to CRLF and unterminated frontmatter.

    Covers review-fix #01 S3 (CRLF normalization) and S4 (missing
    closing delimiter raises ``AssertionError`` with a useful message,
    not the bare ``ValueError`` from ``str.index``).
    """
    # S3: CRLF-encoded file parses identically to LF.
    crlf_file = tmp_path / "crlf.md"
    crlf_file.write_bytes(
        b"---\r\n"
        b"description: crlf agent\r\n"
        b"mode: primary\r\n"
        b"---\r\n"
        b"\r\n"
        b"# body\r\n"
    )
    fm, body = _parse_frontmatter(crlf_file)
    assert fm == {"description": "crlf agent", "mode": "primary"}
    assert body == "\n# body\n"

    # S4: missing closing ``---`` triggers AssertionError (not
    # ValueError) with a path-anchored message.
    truncated = tmp_path / "truncated.md"
    truncated.write_text("---\ndescription: x\nmode: primary\n# body, no closer\n")
    with pytest.raises(AssertionError, match="missing closing"):
        _parse_frontmatter(truncated)

    # Missing OPENING ``---`` triggers AssertionError too — symmetric guard.
    no_open = tmp_path / "no_open.md"
    no_open.write_text("description: x\nmode: primary\n")
    with pytest.raises(AssertionError, match="missing opening"):
        _parse_frontmatter(no_open)
