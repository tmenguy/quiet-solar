"""Pin the QS-434 contract: ``/release`` runs the ``qs-release`` protocol inline.

``.claude/commands/release.md`` must not delegate to the ``qs-release``
subagent: auto mode only accepts the user's *direct* consent for the public
push (main + a ``v*`` tag), so the real run has to happen in the main session,
after the user's own approving message in a later turn.

Authoring rules for ``release.md`` that these tests rely on:

- The pinned phrases are compared lowercased after whitespace normalisation,
  so they must stay un-bolded / un-italicised (no ``*`` inside a phrase).
- ``$ARGUMENTS`` must appear exactly once (the ``User request:`` line):
  Claude Code substitutes every occurrence at invocation, so a placeholder
  inside a rule would be replaced by the user's text.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_MD = REPO_ROOT / ".claude" / "commands" / "release.md"


def _normalize_whitespace(body: str) -> str:
    """Collapse newlines + markdown blockquote markers (``> ``) to single spaces.

    Markdown wraps long lines, so a phrase like ``Claude Desktop`` may appear
    as ``Claude\\n> Desktop`` in the source. We normalize before substring
    checks so the tests are robust to line wrap.
    """
    # Strip ``> `` blockquote prefixes (at line start) and collapse all
    # whitespace runs to a single space.
    stripped = re.sub(r"(?m)^\s*>\s?", "", body)
    return re.sub(r"\s+", " ", stripped)


def _body() -> str:
    return _normalize_whitespace(RELEASE_MD.read_text(encoding="utf-8")).lower()


def test_release_runs_agent_protocol_inline() -> None:
    body = _body()
    assert "use the qs-release subagent" not in body.replace("*", "")
    assert "do not call the `agent` tool" in body
    assert ".claude/agents/qs-release.md" in body
    assert "always run `python scripts/qs/render_agents.py`" in body


def test_release_stops_after_dry_run() -> None:
    body = _body()
    assert "end your turn" in body
    assert "the text typed after /release is never approval" in body
    assert RELEASE_MD.read_text(encoding="utf-8").count("$ARGUMENTS") == 1


def test_release_commands_are_standalone() -> None:
    assert "run the real release as exactly `python scripts/qs/release.py`" in _body()


def test_release_verifies_after_run() -> None:
    body = _body()
    assert "git ls-remote --exit-code origin refs/tags/" in body
    assert "git ls-remote origin refs/heads/main" in body
    assert "gh run list --workflow release.yml --branch" in body
    assert "do not re-run `python scripts/qs/release.py`" in body


def test_release_guards_real_run() -> None:
    """Review fixes #01/#02: no dirty-tree loop, no drift, no unpushed commits."""
    body = _body()
    assert "if the tree is not clean, stop" in body
    assert "git log --oneline origin/main..main` must print nothing" in body
    assert "apply step 2 of this list to it first" in body
    assert "unpushed commits mean stop, whatever else differs" in body


def test_release_reports_partial_push_and_run_status() -> None:
    """Review fixes #01/#02: partial-push evidence and a total run-status mapping."""
    body = _body()
    assert "git tag --list <tag>" in body
    assert "the listed run is not completed yet" in body
    assert "completed with any other conclusion" in body


def test_release_has_no_degraded_fallback_preamble() -> None:
    body = _body()
    assert "degraded fallback" not in body
    assert "claude --agent qs-release" in body


def test_rendered_release_agent_has_referenced_sections() -> None:
    from tests.qs.agents._rendered import agents_dir

    text = (agents_dir("claude") / "qs-release.md").read_text(encoding="utf-8")
    assert "## Phase protocol" in text
    assert "## Hard rules" in text
