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
#
# Round-2 review-fix #02 NTH7: this used to reach into the private
# ``_claude_command`` helper (ruff ``SLF001``). The refactored version
# monkeypatches the resolver so the synthetic agent name flows through
# the public ``build_payload`` API instead.
# --------------------------------------------------------------------------- #


def test_build_payload_shlex_quotes_agent_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """``build_payload`` must shlex.quote whatever the resolver returns."""
    import shlex

    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    # Inject a synthetic agent name via the public path — the resolver is
    # what build_payload consults, so monkeypatching it puts a value with
    # metacharacters into the shell command without reaching for any
    # private helper.
    monkeypatch.setattr(
        claude_launcher,
        "resolve_agent_for_next_cmd",
        lambda _next_cmd: "qs-test agent's-name",
    )
    payload = claude_launcher.build_payload(
        "/tmp/work", 99, "Title", next_cmd="create-plan",
    )
    script = _read_script(payload["new_context"])

    expected_safe = shlex.quote("qs-test agent's-name")
    assert expected_safe in script, (
        f"Expected shlex-quoted agent in script; got: {script!r}"
    )
    # And the raw form (with unescaped space + apostrophe) must NOT appear
    # outside the quoted block.
    assert "qs-test agent's-name " not in script.replace(expected_safe, "")


# --------------------------------------------------------------------------- #
# build_payload-level rejection of invalid next_cmd values — review-fix #02
# NTH10. Each input below should raise ``ValueError`` (or a subclass) so
# the launcher contract stays end-to-end strict.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Review fix plan #01 — should-fix #17: existing_session_prompt key.
# Parallel pin in test_cursor_launcher / test_codex_launcher /
# test_opencode_launcher. The shared helper lives in
# launchers/phases.py::build_existing_session_prompt; each launcher
# threads the kwargs through ``build_payload``.
# --------------------------------------------------------------------------- #


def test_build_payload_includes_existing_session_prompt() -> None:
    """Both ``fix_plan_path`` and ``pr_number`` provided → payload carries the prompt."""
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    payload = claude_launcher.build_payload(
        "/tmp/wt",
        177,
        "Test",
        next_cmd="implement-task",
        fix_plan_path="/tmp/wt/docs/stories/QS-177.story_review_fix_#01.md",
        pr_number=179,
    )
    assert "existing_session_prompt" in payload
    prompt = payload["existing_session_prompt"]
    assert "docs/stories/QS-177.story_review_fix_#01.md" in prompt
    assert "#179" in prompt
    # Worktree-relative path — no absolute prefix leak.
    assert "/tmp/wt/" not in prompt


def test_build_payload_omits_existing_session_prompt_when_inputs_missing() -> None:
    """Neither ``fix_plan_path`` nor ``pr_number`` provided → key omitted."""
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    payload = claude_launcher.build_payload(
        "/tmp/wt", 177, "Test", next_cmd="implement-task",
    )
    assert "existing_session_prompt" not in payload


@pytest.mark.parametrize(
    "bad_next_cmd", ["", "/", "//create-plan", "unknown", "/nope"],
)
def test_claude_build_payload_rejects_invalid_next_cmd(bad_next_cmd: str) -> None:
    """``build_payload`` raises for invalid next_cmd at the public boundary.

    ``""`` is included for parity with the CLI-layer check
    (review-fix #03 NTH1): a direct caller importing ``build_payload``
    must hit the same contract as a user passing ``--next-cmd ""``.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    with pytest.raises(ValueError):
        claude_launcher.build_payload(
            "/tmp/work", 1, "Title", next_cmd=bad_next_cmd,
        )
