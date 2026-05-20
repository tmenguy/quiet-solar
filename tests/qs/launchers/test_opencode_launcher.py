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
    # Structural pin via ``shlex.split`` so a future quote-aware
    # variation of ``sh`` invocation still yields the script path
    # correctly (review fix #03 should-fix #11).
    import shlex as _shlex
    parts = _shlex.split(new_context)
    assert parts[0] == "sh"
    script_path = Path(parts[1])
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
# QS-190 Task 8.1 — docstring records the closed-limitation note
# --------------------------------------------------------------------------- #


def test_build_payload_documents_closed_limitation() -> None:
    """The module docstring records that QS-190 closed the QS-177 AC #12 limitation.

    Renamed from ``test_build_payload_emits_known_limitation_note`` —
    the QS-177 AC #12 known limitation (HTTP API silently lands the
    session on the default agent when ``.opencode/agents/<agent>.md``
    is missing) is now closed by QS-190's pre-flight guard in
    ``spawn_session.py``. The docstring text is rephrased to a
    closed-historical note and the pinning substring is updated to a
    phrase unique to the new wording.
    """
    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

    assert opencode_launcher.__doc__ is not None
    doc = opencode_launcher.__doc__
    assert "spawn_session.py performs a pre-flight check" in doc, (
        "Module docstring must record the closed-limitation note "
        "introduced by QS-190 — the pre-flight in spawn_session.py "
        "now guards against the missing-agent-file failure mode."
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
    # Structural pin via ``shlex.split`` (review fix #03 should-fix #11).
    import shlex as _shlex
    parts = _shlex.split(payload["new_context"])
    assert parts[0] == "sh"
    script_path = Path(parts[1])
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


@pytest.mark.parametrize(
    "bad_issue",
    [
        "42; rm -rf $HOME",  # shell-injection vector
        "abc",               # plain non-numeric string
        None,                # programmer error — int(None) raises TypeError
    ],
)
def test_opencode_cli_command_rejects_non_integer_issue(bad_issue: object) -> None:
    """A non-integer ``issue`` raises ``ValueError`` — closes the shell-injection vector.

    The signature admits ``int | str`` but the body coerces via
    ``int(issue)`` so a payload like ``issue="42; rm -rf $HOME"``
    raises rather than producing a script path with a ``;`` injected.

    Review fix #02 should-fix #10: widen the exception handling to
    convert both ``TypeError`` (from ``int(None)``) and ``ValueError``
    (from ``int("abc")``) into a single ``ValueError`` so the public
    contract is uniform.
    """
    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

    with pytest.raises(ValueError):
        opencode_launcher.build_payload(
            "/tmp/work",
            bad_issue,  # type: ignore[arg-type]
            "Fix bug",
            next_cmd="create-plan",
            caller="setup_task",
        )


def test_opencode_cli_command_rejects_bool_issue() -> None:
    """Bool subclass of int is explicitly rejected.

    ``int(True) == 1`` / ``int(False) == 0`` would silently coerce —
    almost certainly a programmer error. Reject upfront so the type
    contract reads "actual integers only" (review fix #02 should-fix
    #10).
    """
    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

    with pytest.raises(ValueError):
        opencode_launcher.build_payload(
            "/tmp/work",
            True,  # type: ignore[arg-type]
            "Fix bug",
            next_cmd="create-plan",
            caller="setup_task",
        )


@pytest.mark.parametrize("issue", [42, "42"])
def test_build_payload_returns_spawn_session_invocation_int_str_parity(
    issue: int | str,
) -> None:
    """``_spawn_session_command`` accepts ``int`` and ``str`` issue equivalently.

    Parity gap with ``_opencode_cli_command`` (which already coerces
    via ``int(issue)``). Review fix #02 should-fix #3.
    """
    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

    payload = opencode_launcher.build_payload(
        "/tmp/work", issue, "Fix bug", next_cmd="create-plan", caller="next_step",
    )
    # Title uses the post-coercion integer form.
    assert "QS_42: Fix bug" in payload["new_context"]


def test_spawn_session_command_rejects_non_integer_issue() -> None:
    """``_spawn_session_command`` also rejects shell-injection-style strings.

    Parity with ``_opencode_cli_command``'s upfront ``int(issue)``
    coercion (review fix #02 should-fix #3). Without this, a future
    refactor that drops ``shlex.quote`` on the title interpolation
    would reopen the shell-injection vector via the next_step path.
    """
    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

    with pytest.raises(ValueError):
        opencode_launcher.build_payload(
            "/tmp/work",
            "42; rm -rf",
            "Fix bug",
            next_cmd="create-plan",
            caller="next_step",
        )


def test_opencode_cli_command_shell_quotes_script_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A ``TMPDIR`` with whitespace produces a shell-safe ``sh <path>`` return.

    Without ``shlex.quote`` on the script path, the returned
    ``new_context`` would split on the embedded whitespace and the
    shell would execute the first token as the script and treat the
    rest as separate arguments.

    Structural pin (review fix #02 should-fix #4): use ``shlex.split``
    to assert the EXACT tokenization rather than a substring-or-quote
    OR-disjunction that any unrelated single quote could satisfy.
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
        # Structural assertion: after ``shlex.split`` the second token
        # must be EXACTLY the absolute script path under ``weird_tmp``.
        # A regression that drops the path quoting would split the
        # path on the embedded space and fail this pin.
        parts = shlex.split(new_context)
        assert parts[0] == "sh", parts
        assert parts[1].startswith(str(weird_tmp)), parts


# --------------------------------------------------------------------------- #
# Review fix plan #01 — should-fix #4: unique tempfile paths per invocation
# --------------------------------------------------------------------------- #


def test_opencode_cli_command_uses_unique_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Two ``build_payload`` calls with the same issue produce different script paths.

    The legacy implementation wrote to a deterministic
    ``/tmp/qs_oc_launch_<N>.sh`` path keyed only on issue — concurrent
    setup-task runs raced on the file and a malicious symlink could
    redirect the write. ``tempfile.NamedTemporaryFile`` generates
    unique names so each invocation lands on its own path.

    Tempfile sandbox via ``tmp_path`` (review fix #03 nice-to-have
    #17): pytest auto-cleans the directory so we don't accumulate
    artefacts in the real ``/tmp`` across CI runs.
    """
    import tempfile as tempfile_module

    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

    monkeypatch.setattr(tempfile_module, "gettempdir", lambda: str(tmp_path))

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


@pytest.mark.parametrize("weird_work_dir", [
    "/tmp/work",                # baseline
    "/tmp/with space/wt",       # whitespace
    "/tmp/with'quote/wt",       # quote
])
def test_opencode_cli_command_falls_back_to_inline_on_write_failure(
    monkeypatch: pytest.MonkeyPatch, weird_work_dir: str,
) -> None:
    """When ``NamedTemporaryFile`` raises ``OSError`` (read-only fs, etc.), fall back to inline command.

    Without this fallback, ``build_payload`` would propagate the
    ``OSError`` uncaught all the way out to ``setup_task.main()``
    with no JSON error payload — the user gets a Python traceback
    instead of a working command.

    Shell-safety pin (review fix #03 should-fix #6): use
    ``shlex.split`` so a regression producing an unquoted directory
    with spaces / quotes is caught by tokenisation, not just by
    substring presence.
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
        weird_work_dir, 42, "Fix bug", next_cmd="create-plan", caller="setup_task",
    )
    new_context = payload["new_context"]
    # No script-file wrapper — the bare ``opencode …`` command is
    # returned directly.
    assert not new_context.startswith("sh "), new_context

    # Structural pin: tokenise and check exact layout.
    parts = shlex.split(new_context)
    assert parts[0] == "opencode", parts
    # parts[1] is the worktree path — equal to the input post-unquoting.
    assert parts[1] == weird_work_dir, parts
    assert "--agent" in parts
    assert "qs-create-plan" in parts
    assert "--prompt" in parts


def test_opencode_cli_command_cleans_temp_file_on_write_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``f.write`` raises mid-stream → temp file unlinked, inline fallback returned.

    The legacy flow was:

        with NamedTemporaryFile(..., delete=False) as f:
            f.write(...)                  # ← raises here
            script_path = Path(f.name)    # ← never assigned

    If ``f.write`` raised (disk full, quota exceeded, broken pipe),
    the file was already created on disk but ``script_path`` was
    still ``None``, and the cleanup branch did nothing. Review fix
    #03 should-fix #7 captures the path BEFORE writing so cleanup
    always knows what to unlink.
    """
    import tempfile as tempfile_module

    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

    created_paths: list[str] = []
    original_named_temp = tempfile_module.NamedTemporaryFile

    class _WrappedFile:
        """File wrapper that raises ``OSError`` from ``write``."""

        def __init__(self, real: object) -> None:
            self._real = real
            self.name = real.name  # type: ignore[attr-defined]

        def write(self, _data: str) -> int:
            raise OSError("disk full")

        def __enter__(self) -> _WrappedFile:
            return self

        def __exit__(self, *_args: object) -> None:
            self._real.__exit__(None, None, None)  # type: ignore[attr-defined]

    def _wrapping_temp(*args: object, **kwargs: object) -> _WrappedFile:
        real = original_named_temp(*args, **kwargs)  # type: ignore[arg-type]
        created_paths.append(real.name)
        return _WrappedFile(real)

    monkeypatch.setattr(tempfile_module, "NamedTemporaryFile", _wrapping_temp)

    payload = opencode_launcher.build_payload(
        "/tmp/work", 42, "Fix bug", next_cmd="create-plan", caller="setup_task",
    )
    # Inline fallback was taken.
    assert not payload["new_context"].startswith("sh ")
    # The temp file was created (so we know the write path ran) but
    # cleanup unlinked it (review fix #03 should-fix #7).
    assert len(created_paths) == 1
    assert not Path(created_paths[0]).exists(), (
        f"temp file should have been cleaned up: {created_paths[0]}"
    )


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
# QS-190 Task 8.2 — harness.md records the closed-limitation note
# --------------------------------------------------------------------------- #


def test_harness_md_documents_closed_limitation() -> None:
    """``docs/workflow/harness.md`` records the QS-190 closed-limitation note.

    Renamed from ``test_harness_md_contains_known_limitation_note`` —
    parallel to the launcher docstring pin in
    ``test_build_payload_documents_closed_limitation``. The QS-177
    AC #12 known limitation is now closed; the harness.md paragraph
    points to QS-190's pre-flight guard.
    """
    harness_md = (
        Path(__file__).resolve().parents[3]
        / "docs" / "workflow" / "harness.md"
    ).read_text()
    assert "spawn_session.py performs a pre-flight check" in harness_md, (
        "harness.md must record the closed-limitation note introduced "
        "by QS-190 — a missing OpenCode agent file now produces a "
        "clean `agent_file_missing` exit shape via the pre-flight."
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


# --------------------------------------------------------------------------- #
# Review fix plan #02 — should-fix #7: inline fallback degradation documented
# --------------------------------------------------------------------------- #


def test_opencode_cli_command_inline_fallback_documented() -> None:
    """``_opencode_cli_command`` docstring documents the degraded inline-fallback path.

    The fallback drops the tab-title banner + activate/preload echoes
    — a degraded UX. The docstring must call this out so a future
    reader doesn't assume the fallback is feature-parity with the
    script path.
    """
    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

    doc = opencode_launcher._opencode_cli_command.__doc__
    assert doc is not None
    doc_lower = doc.lower()
    assert "degraded" in doc_lower or "inline fallback drops" in doc_lower, (
        "_opencode_cli_command docstring must document the inline-fallback "
        "degradation (banner UX dropped on temp filesystem write failure). "
        "See review fix #02 should-fix #7."
    )


# --------------------------------------------------------------------------- #
# Review fix plan #02 — should-fix #9: chmod failure cleans up temp file
# --------------------------------------------------------------------------- #


def test_opencode_cli_command_cleans_temp_file_on_chmod_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``chmod`` raises ``OSError``, the orphaned temp file is unlinked.

    On filesystems that don't support POSIX permissions (FAT32, some
    NFS mounts), ``tempfile.NamedTemporaryFile`` creates the file
    successfully but the subsequent ``chmod(0o700)`` raises
    ``OSError``. Without cleanup, every failed run leaks a file
    in the temp directory.
    """
    import tempfile as tempfile_module

    from launchers import opencode as opencode_launcher  # type: ignore[import-not-found]

    created_paths: list[str] = []
    original_named_temp = tempfile_module.NamedTemporaryFile

    def _tracking(*args: object, **kwargs: object) -> object:
        f = original_named_temp(*args, **kwargs)  # type: ignore[arg-type]
        created_paths.append(f.name)
        return f

    monkeypatch.setattr(tempfile_module, "NamedTemporaryFile", _tracking)

    # Force chmod to fail.
    def _failing_chmod(self: Path, mode: int) -> None:
        del self, mode
        raise OSError("FAT32 perm not supported")

    monkeypatch.setattr(Path, "chmod", _failing_chmod)

    payload = opencode_launcher.build_payload(
        "/tmp/work", 42, "Fix bug", next_cmd="create-plan", caller="setup_task",
    )
    # Inline fallback was taken (no script wrapper).
    assert not payload["new_context"].startswith("sh ")
    # The temp file was created but cleaned up.
    assert len(created_paths) == 1, created_paths
    assert not Path(created_paths[0]).exists(), (
        f"temp file should have been cleaned up: {created_paths[0]}"
    )
