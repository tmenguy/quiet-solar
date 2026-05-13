"""Tests for ``scripts/qs/launchers/opencode.py`` ``build_payload``.

The OpenCode launcher has a bifurcated contract — ``caller="next_step"``
(default; intermediate phases) emits a ``python scripts/qs/spawn_session.py``
invocation (HTTP API path), while ``caller="setup_task"`` emits a
``sh /tmp/qs_oc_launch_<N>.sh`` one-liner whose generated script
invokes ``opencode <worktree> --agent <name> --prompt <kickoff>`` (CLI
path, because the new worktree is a cross-workspace launch).

Both paths share the same phase-name resolution as the Claude / Cursor
launchers — unknown phases raise ``UnknownPhaseError`` (caught by
``next_step.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest


# --------------------------------------------------------------------------- #
# Task 6.2 — next_step branch (HTTP API form, default)
# --------------------------------------------------------------------------- #


def test_build_payload_returns_spawn_session_invocation() -> None:
    """``caller='next_step'`` produces a ``python scripts/qs/spawn_session.py …`` command."""
    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

    payload = opencode_launcher.build_payload(
        "/tmp/work",
        42,
        "Fix bug",
        next_cmd="create-plan",
        caller="next_step",
    )

    assert payload["tool"] == "opencode"
    assert payload["agent"] == "qs-create-plan"
    assert payload["same_context"] == "create-plan"
    new_context = payload["new_context"]
    # Single-line shell command — no embedded newlines.
    assert "\n" not in new_context
    assert new_context.startswith(
        "python scripts/qs/spawn_session.py --agent qs-create-plan --directory ",
    )
    # Title is QS_<issue>: <title>
    assert "QS_42: Fix bug" in new_context


# --------------------------------------------------------------------------- #
# Task 6.3 — setup_task branch (CLI form)
# --------------------------------------------------------------------------- #


def test_build_payload_setup_task_uses_cli_form() -> None:
    """``caller='setup_task'`` produces a ``sh /tmp/qs_oc_launch_<N>.sh`` one-liner."""
    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

    payload = opencode_launcher.build_payload(
        "/tmp/work",
        42,
        "Fix bug",
        next_cmd="create-plan",
        caller="setup_task",
    )
    new_context = payload["new_context"]
    assert new_context.startswith("sh "), new_context
    script_path = Path(new_context[len("sh "):])
    assert script_path.name.startswith("qs_oc_launch_42")
    script_body = script_path.read_text()
    # The generated script invokes opencode <worktree> --agent <name> --prompt <kickoff>.
    assert "opencode" in script_body
    assert "/tmp/work" in script_body
    assert "--agent qs-create-plan" in script_body
    # Default kickoff prompt is present.
    assert "Begin your phase protocol." in script_body
    # Agent still surfaced as a top-level payload key.
    assert payload["agent"] == "qs-create-plan"


# --------------------------------------------------------------------------- #
# Task 6.4 — default caller is next_step
# --------------------------------------------------------------------------- #


def test_build_payload_default_caller_is_next_step() -> None:
    """Omitting ``caller=`` is identical to ``caller='next_step'``."""
    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

    default_payload = opencode_launcher.build_payload(
        "/tmp/work", 42, "Fix bug", next_cmd="create-plan",
    )
    explicit_payload = opencode_launcher.build_payload(
        "/tmp/work", 42, "Fix bug", next_cmd="create-plan", caller="next_step",
    )
    assert default_payload["new_context"] == explicit_payload["new_context"]
    assert default_payload["agent"] == explicit_payload["agent"]


# --------------------------------------------------------------------------- #
# Task 6.5 — slash and bare phase agree on agent
# --------------------------------------------------------------------------- #


def test_build_payload_back_compat_bare_and_slash_agree() -> None:
    """Slash and bare phase forms resolve to the same agent."""
    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

    bare = opencode_launcher.build_payload(
        "/tmp/work", 42, "Fix bug", next_cmd="create-plan",
    )
    slash = opencode_launcher.build_payload(
        "/tmp/work", 42, "Fix bug", next_cmd="/create-plan",
    )
    assert bare["agent"] == slash["agent"] == "qs-create-plan"


# --------------------------------------------------------------------------- #
# Task 6.6 — unknown phase raises
# --------------------------------------------------------------------------- #


def test_build_payload_unknown_phase_raises() -> None:
    """Unknown phase propagates as ``UnknownPhaseError`` (caught by next_step.py)."""
    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]
    from launchers.phases import UnknownPhaseError  # type: ignore[import-not-found]

    with pytest.raises(UnknownPhaseError):
        opencode_launcher.build_payload(
            "/tmp/work", 42, "Fix bug", next_cmd="bogus",
        )


# --------------------------------------------------------------------------- #
# Task 6.7 — shell escaping of paths and titles
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("work_dir", "title"),
    [
        ("/tmp/with space/wt", "It's a 'title'"),
        ("/tmp/no-space", "Plain"),
    ],
)
def test_build_payload_shell_escapes_path_with_spaces(
    work_dir: str, title: str,
) -> None:
    """``new_context`` must shlex-quote interpolated values (no shell-injection vectors)."""
    import shlex

    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

    payload = opencode_launcher.build_payload(
        work_dir, 42, title, next_cmd="create-plan", caller="next_step",
    )
    new_context = payload["new_context"]
    assert shlex.quote(work_dir) in new_context, new_context
    expected_title = f"QS_42: {title}"
    assert shlex.quote(expected_title) in new_context, new_context


# --------------------------------------------------------------------------- #
# Task 6.8 — docstring documents the AC #12 limitation
# --------------------------------------------------------------------------- #


def test_build_payload_emits_known_limitation_note() -> None:
    """The module docstring documents the missing ``.opencode/agents/qs-<phase>.md`` limitation."""
    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

    assert opencode_launcher.__doc__ is not None
    doc = opencode_launcher.__doc__
    assert ".opencode/agents/qs-" in doc, (
        "Module docstring must document the AC #12 limitation that "
        "static `.opencode/agents/qs-<phase>.md` files are required "
        "for agent activation to take effect."
    )


# --------------------------------------------------------------------------- #
# Additional coverage — next_prompt forwarding, CLI form payload keys
# --------------------------------------------------------------------------- #


def test_build_payload_next_prompt_forwarded() -> None:
    """When ``next_prompt`` is provided, it lands in the spawn_session ``--prompt``."""
    import shlex

    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

    payload = opencode_launcher.build_payload(
        "/tmp/work",
        42,
        "Fix bug",
        next_cmd="create-plan",
        next_prompt="Custom kickoff",
        caller="next_step",
    )
    assert shlex.quote("Custom kickoff") in payload["new_context"]


def test_build_payload_setup_task_next_prompt_forwarded() -> None:
    """``setup_task`` caller forwards ``next_prompt`` into the CLI script."""
    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

    payload = opencode_launcher.build_payload(
        "/tmp/work",
        42,
        "Fix bug",
        next_cmd="create-plan",
        next_prompt="Custom kickoff",
        caller="setup_task",
    )
    script_path = Path(payload["new_context"][len("sh "):])
    body = script_path.read_text()
    assert "Custom kickoff" in body


def test_build_payload_default_kickoff_used_when_no_next_prompt() -> None:
    """Without ``next_prompt``, the default kickoff prompt is used."""
    import shlex

    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

    payload = opencode_launcher.build_payload(
        "/tmp/work", 42, "Fix bug", next_cmd="create-plan", caller="next_step",
    )
    assert shlex.quote("Begin your phase protocol.") in payload["new_context"]
