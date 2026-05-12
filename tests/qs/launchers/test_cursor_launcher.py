"""Tests for ``scripts/qs/launchers/cursor.py`` ``build_payload``.

The Cursor launcher gets parity with Claude: when ``cursor-agent`` is on
PATH, ``cli_context`` invokes ``cursor-agent --workspace <wd> --agent
qs-<phase>``; otherwise it falls back to the legacy prompt-positional
form.
"""

from __future__ import annotations

import shlex
import shutil

import pytest


def test_cli_context_uses_agent_flag_when_cursor_agent_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """``cursor-agent`` on PATH → cli_context invokes it with --agent."""
    from launchers import cursor as cursor_launcher  # type: ignore[import-not-found]

    def fake_which(name: str) -> str | None:
        if name == "cursor-agent":
            return "/usr/local/bin/cursor-agent"
        return None

    monkeypatch.setattr(shutil, "which", fake_which)

    payload = cursor_launcher.build_payload(
        "/tmp/work",
        42,
        "Fix bug",
        next_cmd="create-plan",
    )

    assert payload["tool"] == "cursor"
    assert payload["agent"] == "qs-create-plan"
    assert payload["same_context"] == "create-plan"
    cli = payload["cli_context"]
    assert "cursor-agent" in cli
    assert "--workspace" in cli
    assert "--agent qs-create-plan" in cli
    # And the work dir is properly quoted.
    assert shlex.quote("/tmp/work") in cli


def test_cli_context_falls_back_when_cursor_agent_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ``cursor-agent`` on PATH → cli_context uses legacy prompt-positional form."""
    from launchers import cursor as cursor_launcher  # type: ignore[import-not-found]

    monkeypatch.setattr(shutil, "which", lambda _name: None)

    payload = cursor_launcher.build_payload(
        "/tmp/work",
        42,
        "Fix bug",
        next_cmd="create-plan",
    )
    cli = payload["cli_context"]
    # Fallback path: no --agent flag, legacy prompt-positional form.
    assert "--agent" not in cli
    assert "cursor-agent --workspace" in cli
    # Slash form gets injected into the prompt body for the fallback.
    assert "create-plan" in cli


def test_same_context_mirrors_slash_form(monkeypatch: pytest.MonkeyPatch) -> None:
    """``same_context`` is the next_cmd verbatim (parity with claude.py)."""
    from launchers import cursor as cursor_launcher  # type: ignore[import-not-found]

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    payload = cursor_launcher.build_payload(
        "/tmp/work", 1, "T", next_cmd="/review-task",
    )
    assert payload["same_context"] == "/review-task"
    assert payload["agent"] == "qs-review-task"


def test_unknown_phase_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown phase propagates as ValueError (matches claude.py behaviour)."""
    from launchers import cursor as cursor_launcher  # type: ignore[import-not-found]

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(ValueError):
        cursor_launcher.build_payload(
            "/tmp/work", 1, "T", next_cmd="bogus",
        )


def test_next_prompt_appended_to_cli_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """``next_prompt`` survives into the cli_context for the fallback path."""
    from launchers import cursor as cursor_launcher  # type: ignore[import-not-found]

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    payload = cursor_launcher.build_payload(
        "/tmp/work",
        1,
        "T",
        next_cmd="create-plan",
        next_prompt="hello world",
    )
    assert "hello world" in payload["cli_context"]
