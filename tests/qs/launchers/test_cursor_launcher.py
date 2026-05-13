"""Tests for ``scripts/qs/launchers/cursor.py`` ``build_payload``.

The Cursor launcher gets parity with Claude: when ``cursor-agent`` is on
PATH, ``cli_context`` invokes ``cursor-agent --workspace <wd> --agent
qs-<phase>``; otherwise it falls back to the legacy prompt-positional
form.
"""

from __future__ import annotations

import shlex
import shutil
from pathlib import Path

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


# --------------------------------------------------------------------------- #
# Fallback prompt must render the SLASH form regardless of input shape — so a
# user landing on the degraded Cursor path can invoke the phase as a slash
# command in chat. Regression catch for review-fix #2.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("next_cmd", ["create-plan", "/create-plan"])
def test_cursor_fallback_cli_uses_slash_form(
    next_cmd: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When cursor-agent is missing, the fallback prompt embeds ``/<phase>``."""
    from launchers import cursor as cursor_launcher  # type: ignore[import-not-found]

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    payload = cursor_launcher.build_payload(
        "/tmp/work", 1, "T", next_cmd=next_cmd,
    )
    cli = payload["cli_context"]
    # Slash form must be present so the user can paste/type it as a slash
    # command once they open Cursor manually.
    assert "/create-plan" in cli, f"expected '/create-plan' in {cli!r}"


@pytest.mark.parametrize("next_cmd", ["create-plan", "/create-plan"])
def test_cursor_fallback_new_context_uses_slash_form(
    next_cmd: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``new_context`` instructions point at the slash form when no IDE on PATH."""
    from launchers import cursor as cursor_launcher  # type: ignore[import-not-found]

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    payload = cursor_launcher.build_payload(
        "/tmp/work", 1, "T", next_cmd=next_cmd,
    )
    new_context = payload["new_context"]
    # The "Then in chat, type:" instruction must surface ``/create-plan``
    # (with leading slash) so the user reaches the slash-command fallback.
    assert "/create-plan" in new_context, (
        f"expected '/create-plan' in new_context, got: {new_context!r}"
    )


# --------------------------------------------------------------------------- #
# IDE-path slash-form coverage — review-fix #04 SF3 + NTH11.
#
# Round 2 SF2 normalized the FALLBACK-path prompt to slash form, but the
# IDE-path banner script kept interpolating the raw ``next_cmd`` — so a
# bare phase was rendered as ``type create-plan`` in the user-facing
# instruction. The IDE path is the more common one (cursor IDE on PATH),
# so this gap was the more user-visible of the two.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("next_cmd", ["create-plan", "/create-plan"])
def test_cursor_ide_banner_uses_slash_form(
    next_cmd: str, monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    """IDE launcher script's user-facing instruction shows ``/<phase>``."""
    from launchers import cursor as cursor_launcher  # type: ignore[import-not-found]

    # Pretend ``cursor`` is on PATH so the IDE path is selected (not the
    # text-instructions fallback). The IDE script lands under
    # tempfile.gettempdir() — read it back to inspect the rendered text.
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/cursor" if name == "cursor" else None)
    payload = cursor_launcher.build_payload(
        "/tmp/work", 42, "T", next_cmd=next_cmd,
    )
    assert payload["new_context"].startswith("sh "), payload["new_context"]
    script_path = Path(payload["new_context"][len("sh "):])
    script_body = script_path.read_text()

    # The "In the chat, type: ..." banner is the user-facing instruction
    # at the top of the IDE script. It must show ``/create-plan`` with a
    # leading slash regardless of the input form.
    assert "/create-plan" in script_body, (
        f"IDE banner missed the slash form. Got script body:\n{script_body!r}"
    )


def test_cursor_same_context_preserves_input_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``same_context`` is the next_cmd verbatim — only the fallback prompt normalizes."""
    from launchers import cursor as cursor_launcher  # type: ignore[import-not-found]

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    bare = cursor_launcher.build_payload("/tmp/w", 1, "T", next_cmd="create-plan")
    slash = cursor_launcher.build_payload("/tmp/w", 1, "T", next_cmd="/create-plan")
    # Verbatim preservation is the back-compat invariant.
    assert bare["same_context"] == "create-plan"
    assert slash["same_context"] == "/create-plan"


# --------------------------------------------------------------------------- #
# Source-grep regression — locks the cursor fallback rationale comment in
# place so a future refactor cannot silently delete the "why we fall back to
# the legacy prompt-positional form" explanation. Review-fix #10.
# --------------------------------------------------------------------------- #


def test_cursor_source_documents_fallback_rationale() -> None:
    """cursor.py must explain why the legacy form is used when cursor-agent is missing."""
    cursor_py = (
        Path(__file__).resolve().parents[3]
        / "scripts" / "qs" / "launchers" / "cursor.py"
    )
    source = cursor_py.read_text()
    # The rationale block lives in _cursor_cli_command's fallback branch.
    # Look for two stable substrings — keep this loose so wording can evolve.
    assert "legacy prompt-positional form" in source, (
        "Fallback rationale comment was removed from cursor.py — restore the "
        "block explaining why older cursor-agent binaries (no --agent support) "
        "fall back to the legacy form."
    )
    assert "manual equivalent" in source.lower() or "type ``/" in source.lower() \
        or "type `/" in source.lower(), (
        "Fallback rationale should document the manual equivalent for the "
        "user (open Cursor → type the slash command in chat)."
    )


# --------------------------------------------------------------------------- #
# slash_form defense-in-depth — review-fix #02 NTH8.
#
# The helper is upstream-protected by ``resolve_agent_for_next_cmd``, so
# the empty / double-slash cases are currently unreachable from
# ``build_payload``. But a future refactor could call ``slash_form``
# outside that validation path. The function should be a leaf with its
# own preconditions encoded.
#
# Round-3 review-fix #03 SF2: the helper was promoted to a public name
# (``slash_form`` without leading underscore) so tests can pin its
# contract directly without reaching for a "private" import.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("input_val", "expected"), [
    ("create-plan", "/create-plan"),
    ("/create-plan", "/create-plan"),
    # ``//create-plan`` is a contract violation but the function's job is
    # to NORMALIZE to slash form — passing through here is acceptable.
    # The upstream resolver is what rejects ``//<anything>``.
    ("//create-plan", "//create-plan"),
])
def test_slash_form_normalizes_to_slash_prefix(input_val: str, expected: str) -> None:
    """``slash_form`` adds a leading slash iff one is missing."""
    from launchers.cursor import slash_form  # type: ignore[import-not-found]

    assert slash_form(input_val) == expected


@pytest.mark.parametrize("bad", ["", " ", "  ", "\t", "\n"])
def test_slash_form_rejects_empty_or_whitespace(bad: str) -> None:
    """Empty/whitespace input is a contract violation — ``slash_form`` raises."""
    from launchers.cursor import slash_form  # type: ignore[import-not-found]

    with pytest.raises(ValueError):
        slash_form(bad)


# --------------------------------------------------------------------------- #
# build_payload-level rejection of invalid next_cmd values — review-fix #02
# NTH10. Symmetric with the claude.py contract.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_next_cmd", ["", "/", "//create-plan", "unknown", "/nope"],
)
def test_cursor_build_payload_rejects_invalid_next_cmd(
    bad_next_cmd: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``build_payload`` raises for invalid next_cmd at the public boundary.

    ``""`` is included for parity with the CLI-layer check
    (review-fix #03 NTH1).
    """
    from launchers import cursor as cursor_launcher  # type: ignore[import-not-found]

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(ValueError):
        cursor_launcher.build_payload(
            "/tmp/work", 1, "Title", next_cmd=bad_next_cmd,
        )
