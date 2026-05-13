"""Tests for ``scripts/qs/launchers/claude.py`` ``build_payload``.

The Claude launcher now invokes ``claude --agent qs-<phase>`` instead of a
bare ``claude`` (the old non-interactive Agent-tool path). Both the slash
form and the bare phase name must resolve to the same agent (back-compat
with callers like ``setup_task.py`` that still pass slash form).
"""

from __future__ import annotations

import stat
import tempfile
from pathlib import Path

import pytest


def _read_script(new_context: str) -> str:
    """``new_context`` is a ``sh /tmp/qs_launch_<N>.sh`` one-liner; read the script."""
    assert new_context.startswith("sh "), f"unexpected new_context: {new_context!r}"
    script_path = new_context[len("sh "):]
    return Path(script_path).read_text()


def test_build_payload_emits_agent_flag_for_bare_phase() -> None:
    """``next_cmd='create-plan'`` produces a script invoking ``claude --agent qs-create-plan``."""
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    payload = claude_launcher.build_payload(
        "/tmp/work",
        42,
        "Fix bug",
        next_cmd="create-plan",
    )

    assert payload["tool"] == "claude-code"
    assert payload["agent"] == "qs-create-plan"
    assert payload["same_context"] == "create-plan"

    script = _read_script(payload["new_context"])
    assert "claude " in script
    assert "--agent qs-create-plan" in script


def test_build_payload_emits_agent_flag_for_slash_phase() -> None:
    """Slash form maps to the same agent (back-compat for older callers)."""
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    payload = claude_launcher.build_payload(
        "/tmp/work",
        42,
        "Fix bug",
        next_cmd="/create-plan",
    )
    assert payload["agent"] == "qs-create-plan"
    # ``same_context`` is preserved verbatim so the fallback path the
    # orchestrator prints stays intact.
    assert payload["same_context"] == "/create-plan"
    script = _read_script(payload["new_context"])
    assert "--agent qs-create-plan" in script


def test_build_payload_back_compat_bare_and_slash_agree_on_agent() -> None:
    """Bare and slash forms resolve to the same agent."""
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    bare = claude_launcher.build_payload(
        "/tmp/work", 42, "Fix bug", next_cmd="create-plan",
    )
    slash = claude_launcher.build_payload(
        "/tmp/work", 42, "Fix bug", next_cmd="/create-plan",
    )
    assert bare["agent"] == slash["agent"] == "qs-create-plan"


def test_build_payload_unknown_phase_raises() -> None:
    """Unknown phase propagates as ValueError (no silent fallback)."""
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    with pytest.raises(ValueError):
        claude_launcher.build_payload(
            "/tmp/work", 42, "Fix bug", next_cmd="bogus",
        )


def test_build_payload_script_is_under_tempdir_and_executable() -> None:
    """Generated script lives under tempfile.gettempdir() and is owner-executable."""
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    payload = claude_launcher.build_payload(
        "/tmp/work", 99, "Some title", next_cmd="implement-task",
    )
    new_context = payload["new_context"]
    assert new_context.startswith("sh ")
    script_path = Path(new_context[len("sh "):])
    # Path must be under the system tempdir
    assert str(script_path).startswith(tempfile.gettempdir())
    # Executable-by-owner is the contract that actually matters for
    # ``sh /tmp/qs_launch_<N>.sh`` to run. We avoid asserting an exact mode
    # (``0o755``) because a developer's umask can shift the group/other
    # bits and break the test without breaking the launcher.
    mode = script_path.stat().st_mode
    assert mode & stat.S_IXUSR, f"script not executable by owner: {oct(mode)}"
    assert mode & 0o700 == 0o700, f"owner must have rwx; got: {oct(mode & 0o777)}"


def test_build_payload_preserves_launch_opts_and_workdir() -> None:
    """The legacy ``CLAUDE_LAUNCH_OPTS`` flags survive the rewrite."""
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    payload = claude_launcher.build_payload(
        "/tmp/work",
        7,
        "Title",
        next_cmd="finish-task",
    )
    script = _read_script(payload["new_context"])
    assert "/tmp/work" in script
    # CLAUDE_LAUNCH_OPTS keeps these defaults — change here is intentional and
    # caught by this assertion.
    assert "--dangerously-skip-permissions" in script
    assert "--model opus" in script
    assert "--effort max" in script
    # Stable layout invariant: ``--agent`` appears after CLAUDE_LAUNCH_OPTS
    # in the rendered command line. (CLI flag ORDER is independent in
    # argparse-style parsers — this is a layout/cosmetic check, not a
    # semantic requirement.)
    assert script.index("--dangerously-skip-permissions") < script.index("--agent")


def test_build_payload_appends_next_prompt_when_provided() -> None:
    """``next_prompt`` is appended as a positional initial prompt."""
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    payload = claude_launcher.build_payload(
        "/tmp/work",
        7,
        "Title",
        next_cmd="finish-task",
        next_prompt="please ship it",
    )
    script = _read_script(payload["new_context"])
    assert "please ship it" in script


# --------------------------------------------------------------------------- #
# Shell-escaping regression — lock that the agent name reaches the script via
# ``shlex.quote`` so a hypothetical agent name with whitespace or quotes
# cannot break the shell command. The current mapping never produces
# metacharacters; this test guards the contract anyway. Review-fix #11.
# --------------------------------------------------------------------------- #


def test_claude_command_quotes_agent_name_with_metacharacters() -> None:
    """``_claude_command`` must shlex.quote the agent name."""
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    # Bypass build_payload (which goes through the validated mapping) and
    # exercise the private helper directly with a synthetic name that has
    # whitespace + a single quote — exactly the chars shlex.quote handles.
    script_invocation = claude_launcher._claude_command(
        "/tmp/work",
        99,
        "Title",
        agent="qs-test agent's-name",
        next_prompt=None,
    )
    script = _read_script(script_invocation)
    # The dangerous chars must be quoted, not spliced raw into the shell line.
    # shlex.quote wraps the whole token in single quotes and escapes inner
    # single quotes — the literal substring is the safe form, not the raw.
    import shlex
    expected_safe = shlex.quote("qs-test agent's-name")
    assert expected_safe in script, (
        f"Expected shlex-quoted agent in script; got: {script!r}"
    )
    # And the raw form (with unescaped space + apostrophe) must NOT appear
    # outside the quoted block.
    assert "qs-test agent's-name " not in script.replace(expected_safe, "")
