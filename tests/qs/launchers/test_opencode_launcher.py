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
from unittest.mock import MagicMock

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


# --------------------------------------------------------------------------- #
# Review fix plan #01 — must-fix #2: shell-injection via `issue` parameter
# --------------------------------------------------------------------------- #


def test_opencode_cli_command_rejects_non_integer_issue() -> None:
    """A non-integer ``issue`` raises ``ValueError`` — closes the shell-injection vector.

    The signature admits ``int | str`` but the body coerces via
    ``int(issue)`` so a payload like ``issue="42; rm -rf $HOME"``
    raises rather than producing a script path with a ``;`` injected.
    """
    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

    with pytest.raises(ValueError):
        opencode_launcher.build_payload(
            "/tmp/work",
            "42; rm -rf $HOME",
            "Fix bug",
            next_cmd="create-plan",
            caller="setup_task",
        )


def test_opencode_cli_command_shell_quotes_script_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A ``TMPDIR`` with whitespace produces a shell-safe ``sh <path>`` return.

    Without ``shlex.quote`` on the script path, the returned
    ``new_context`` would split on the embedded whitespace and the
    shell would execute the first token as the script and treat the
    rest as separate arguments.
    """
    import shlex
    import tempfile as tempfile_module

    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

    weird_tmp = tmp_path / "with space"
    weird_tmp.mkdir()
    monkeypatch.setattr(tempfile_module, "gettempdir", lambda: str(weird_tmp))

    payload = opencode_launcher.build_payload(
        "/tmp/work", 42, "Fix bug", next_cmd="create-plan", caller="setup_task",
    )
    new_context = payload["new_context"]
    # Either the path was written and shlex-quoted, or the write
    # failed and we fell back to the inline command. Both are safe.
    if new_context.startswith("sh "):
        # The script-path argument must be shlex-quoted (whitespace
        # cannot leak unquoted into the shell command).
        assert shlex.quote(str(weird_tmp)) in new_context or "'" in new_context, new_context


# --------------------------------------------------------------------------- #
# Review fix plan #01 — should-fix #4: unique tempfile paths per invocation
# --------------------------------------------------------------------------- #


def test_opencode_cli_command_uses_unique_paths() -> None:
    """Two ``build_payload`` calls with the same issue produce different script paths.

    The legacy implementation wrote to a deterministic
    ``/tmp/qs_oc_launch_<N>.sh`` path keyed only on issue — concurrent
    setup-task runs raced on the file and a malicious symlink could
    redirect the write. ``tempfile.NamedTemporaryFile`` generates
    unique names so each invocation lands on its own path.
    """
    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

    p1 = opencode_launcher.build_payload(
        "/tmp/work", 42, "Fix bug", next_cmd="create-plan", caller="setup_task",
    )["new_context"]
    p2 = opencode_launcher.build_payload(
        "/tmp/work", 42, "Fix bug", next_cmd="create-plan", caller="setup_task",
    )["new_context"]
    assert p1 != p2, f"both runs produced the same path: {p1!r}"


# --------------------------------------------------------------------------- #
# Review fix plan #01 — should-fix #5: agent name shlex-quoted inside the echo
# --------------------------------------------------------------------------- #


def test_generated_script_handles_quotes_in_agent_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A synthetic agent name containing a single quote produces a syntactically valid script.

    The legacy ``echo '── Activating agent: {agent} ──'`` line broke
    on a single quote inside ``agent``. The fix wraps the entire echo
    argument in ``shlex.quote``.
    """
    import subprocess

    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

    monkeypatch.setattr(
        opencode_launcher,
        "resolve_agent_for_next_cmd",
        lambda _next_cmd: "qs-with'quote",
    )

    payload = opencode_launcher.build_payload(
        "/tmp/work", 42, "Fix bug", next_cmd="create-plan", caller="setup_task",
    )
    new_context = payload["new_context"]
    if new_context.startswith("sh "):
        # The path is shlex-quoted in the return value; strip the
        # surrounding quotes (if any) to get the raw path.
        import shlex as _shlex
        parts = _shlex.split(new_context)
        script_path = parts[1]
        result = subprocess.run(
            ["bash", "-n", script_path],
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"bash syntax check failed: {result.stderr.decode()}"
        )


# --------------------------------------------------------------------------- #
# Review fix plan #01 — should-fix #6: tempfile write/chmod failure → inline fallback
# --------------------------------------------------------------------------- #


def test_opencode_cli_command_falls_back_to_inline_on_write_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``NamedTemporaryFile`` raises ``OSError`` (read-only fs, etc.), fall back to inline command.

    Without this fallback, ``build_payload`` would propagate the
    ``OSError`` uncaught all the way out to ``setup_task.main()``
    with no JSON error payload — the user gets a Python traceback
    instead of a working command.
    """
    import shlex
    import tempfile as tempfile_module

    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

    monkeypatch.setattr(
        tempfile_module,
        "NamedTemporaryFile",
        MagicMock(side_effect=OSError("read-only filesystem")),
    )

    payload = opencode_launcher.build_payload(
        "/tmp/work", 42, "Fix bug", next_cmd="create-plan", caller="setup_task",
    )
    new_context = payload["new_context"]
    # No script-file wrapper — the bare ``opencode …`` command is
    # returned directly.
    assert not new_context.startswith("sh "), new_context
    assert new_context.startswith("opencode "), new_context
    assert shlex.quote("/tmp/work") in new_context
    assert "--agent qs-create-plan" in new_context


# --------------------------------------------------------------------------- #
# Review fix plan #01 — should-fix #13: harness.md table-row text pinned
# --------------------------------------------------------------------------- #


def test_harness_md_contains_new_opencode_row() -> None:
    """``docs/workflow/harness.md`` contains the exact AC #11 row text.

    AC #11 mandates the specific string in the new table row's
    "Session spawn" column.
    """
    harness_md = (
        Path(__file__).resolve().parents[3]
        / "docs" / "workflow" / "harness.md"
    ).read_text()
    assert (
        "HTTP API: POST /session + POST /session/<id>/prompt_async (no reload)"
        in harness_md
    ), "harness.md is missing the new OpenCode pipeline row text"


# --------------------------------------------------------------------------- #
# Review fix plan #01 — should-fix #14: harness.md known-limitation note pinned
# --------------------------------------------------------------------------- #


def test_harness_md_contains_known_limitation_note() -> None:
    """``docs/workflow/harness.md`` carries the AC #12 ``.opencode/agents/`` limitation note.

    Parallel to the launcher docstring pin in
    ``test_build_payload_emits_known_limitation_note``.
    """
    harness_md = (
        Path(__file__).resolve().parents[3]
        / "docs" / "workflow" / "harness.md"
    ).read_text()
    assert ".opencode/agents/qs-<phase>.md" in harness_md, (
        "harness.md is missing the AC #12 known-limitation note"
    )


# --------------------------------------------------------------------------- #
# Review fix plan #01 — nice-to-have #21: DEFAULT_KICKOFF parity pin
# --------------------------------------------------------------------------- #


def test_default_kickoff_constants_match() -> None:
    """``DEFAULT_KICKOFF`` is identical in the launcher and ``spawn_session``.

    The duplication is intentional (keeps the launcher
    self-contained), but a regression test pins the values to prevent
    silent drift.
    """
    import spawn_session  # type: ignore[import-not-found]

    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

    assert opencode_launcher.DEFAULT_KICKOFF == spawn_session.DEFAULT_KICKOFF
