"""Tests for ``scripts/qs/launchers/claude.py`` ``build_payload``.

The Claude launcher now invokes ``claude --agent qs-<phase>`` instead of a
bare ``claude`` (the old non-interactive Agent-tool path). Both the slash
form and the bare phase name must resolve to the same agent (back-compat
with callers like ``setup_task.py`` that still pass slash form).
"""

from __future__ import annotations

import json
import os
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


# --------------------------------------------------------------------------- #
# QS-311 AC2 — the per-worktree GUI phase pin.
#
# ``build_payload`` writes ``{"agent": "qs-<phase>"}`` into
# ``<work_dir>/.claude/settings.local.json`` so a Claude Code **GUI**
# session opened on the worktree boots as the phase orchestrator (the GUI
# has no ``--agent`` flag). The write is guarded, in this order:
#
#   1. the phase agent file exists at
#      ``<work_dir>/.claude/agents/<agent>.md``  (pure filesystem check —
#      short-circuits before any subprocess)
#   2. ``work_dir`` is a **linked git worktree**: its ``.git`` must be a
#      *file* (``gitdir: …``), and it must not be the main checkout
#
# **Isolation invariant** (AC4): each guard *independently* rejects the
# throwaway paths the tests above pass (``/tmp/work``, ``/tmp/wt``), so no
# existing test writes into a real directory. Review-fix #01 S4: this was
# only true of guard 1 while guard 2 was a bare ``utils.is_worktree``
# call, which means "is not the main checkout" and therefore accepts any
# throwaway path. The ``.git``-is-a-file containment check restores the
# independence, and it is pinned by ``test_no_write_for_non_worktree_path``
# plus ``test_second_clone_is_not_pinned``.
# --------------------------------------------------------------------------- #


SETTINGS_REL = Path(".claude") / "settings.local.json"


def _fake_worktree(tmp_path: Path, agent: str = "qs-create-plan") -> Path:
    """Return ``tmp_path`` prepared as a linked worktree holding ``<agent>.md``.

    A *linked* git worktree's ``.git`` is a FILE containing a ``gitdir:``
    pointer (the main checkout's — and any second clone's — is a
    directory). Guard 2 checks exactly that, so the stub has to carry it
    (review-fix #01 S4).
    """
    agents_dir = tmp_path / ".claude" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / f"{agent}.md").write_text("# stub agent body\n")
    (tmp_path / ".git").write_text(f"gitdir: {tmp_path}/../.git/worktrees/stub\n")
    return tmp_path


def _tmp_siblings(work_dir: Path) -> list[Path]:
    """Return any surviving atomic-write temp siblings of the settings file."""
    return sorted((work_dir / ".claude").glob("settings.local.json.*.tmp"))


def _settings(work_dir: Path) -> dict:
    return json.loads((work_dir / SETTINGS_REL).read_text(encoding="utf-8"))


def test_writes_agent_key_into_new_settings_file(tmp_path: Path) -> None:
    """No settings file yet → one is created carrying just the ``agent`` key."""
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    payload = claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert payload["agent"] == "qs-create-plan"
    assert _settings(work_dir) == {"agent": "qs-create-plan"}
    # The atomic-write temp sibling must not survive the call. The name
    # carries the writer's PID (review-fix #01 N5), so glob for it.
    assert _tmp_siblings(work_dir) == []


def test_merges_and_preserves_existing_keys(tmp_path: Path) -> None:
    """Every other top-level key in the user's local settings survives."""
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    (work_dir / SETTINGS_REL).write_text(
        json.dumps({"permissions": {"allow": ["Bash(git status)"]}, "model": "opus"}),
        encoding="utf-8",
    )

    claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert _settings(work_dir) == {
        "permissions": {"allow": ["Bash(git status)"]},
        "model": "opus",
        "agent": "qs-create-plan",
    }


def test_replaces_pre_existing_agent_value(tmp_path: Path) -> None:
    """A stale pin from the previous phase is replaced, not merged."""
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    (work_dir / SETTINGS_REL).write_text(
        json.dumps({"agent": "qs-review-task"}), encoding="utf-8",
    )

    claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert _settings(work_dir) == {"agent": "qs-create-plan"}


def test_skips_when_destination_is_main_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard 2: the main checkout is never pinned (``--no-worktree``, release)."""
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    # Patch the name imported INTO the launcher module, not utils'.
    monkeypatch.setattr(claude_launcher, "is_worktree", lambda _work_dir: False)

    payload = claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert payload["agent"] == "qs-create-plan"
    assert not (work_dir / SETTINGS_REL).exists()


def test_skips_when_agent_file_absent(tmp_path: Path) -> None:
    """Guard 1: no ``<agent>.md`` in the destination → nothing is written.

    A pin naming an agent that doesn't exist there would fall back to the
    default agent *silently* (finding F3) and the GUI shows no agent name
    (F6) — worse than no pin at all.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    payload = claude_launcher.build_payload(
        str(tmp_path), 311, "Title", next_cmd="create-plan",
    )

    assert payload["agent"] == "qs-create-plan"
    assert not (tmp_path / SETTINGS_REL).exists()


def test_skips_without_invoking_git_when_agent_file_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard ORDER: the cheap filesystem check runs before any subprocess.

    ``git`` is made to raise ``AssertionError`` — deliberately *not* one of
    the exceptions ``utils.is_worktree`` swallows — so a reversed guard
    order surfaces as a failure here instead of silently depending on git
    succeeding in the pytest CWD.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    import utils  # type: ignore[import-not-found]

    def _no_git(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("git must not be invoked when the agent file is absent")

    monkeypatch.setattr(utils, "run_git", _no_git)

    payload = claude_launcher.build_payload(
        str(tmp_path), 311, "Title", next_cmd="create-plan",
    )

    assert payload["agent"] == "qs-create-plan"
    assert not (tmp_path / SETTINGS_REL).exists()


def test_overwrites_corrupt_existing_json_with_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Unparseable settings are rebuilt from scratch, with a warning.

    Skipping would leave the phase unbound, and F3+F6 make that invisible
    in the GUI. Review-fix #01 M1: rebuilding is **not** free — Claude Code
    persists per-user permission decisions in this file — so it is confined
    to genuinely unparseable content (a *read* failure must not rebuild,
    see ``test_read_oserror_does_not_write``) and the old bytes are kept in
    a ``.bak`` sibling (``test_corrupt_file_is_backed_up_before_rebuild``).
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    (work_dir / SETTINGS_REL).write_text("{not json at all", encoding="utf-8")

    claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert _settings(work_dir) == {"agent": "qs-create-plan"}
    assert "warning:" in capsys.readouterr().err


def test_overwrites_non_object_existing_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Valid JSON that isn't an object is corrupt too.

    Without this branch the shallow merge would raise ``TypeError`` /
    ``AttributeError`` *outside* the read/parse guard.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    (work_dir / SETTINGS_REL).write_text("[1, 2]", encoding="utf-8")

    claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert _settings(work_dir) == {"agent": "qs-create-plan"}
    assert "warning:" in capsys.readouterr().err


def test_warns_on_stderr_not_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Warnings must never touch stdout — it carries the JSON payload.

    ``next_step.py`` prints the payload to stdout and
    ``test_next_step_unit.py`` parses it, so a stray ``print()`` here
    would corrupt a machine-read stream.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    (work_dir / SETTINGS_REL).write_text("{corrupt", encoding="utf-8")

    claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    captured = capsys.readouterr()
    assert captured.out == "", f"unexpected stdout output: {captured.out!r}"
    assert "warning:" in captured.err


def test_second_write_is_byte_identical(tmp_path: Path) -> None:
    """Re-pinning the same phase is a no-op change (repeat-safe + stable format)."""
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )
    first = (work_dir / SETTINGS_REL).read_bytes()
    claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="/create-plan",
    )
    second = (work_dir / SETTINGS_REL).read_bytes()

    assert first == second
    # Pin the on-disk format: 2-space indent + trailing newline.
    assert first.decode("utf-8") == (
        json.dumps({"agent": "qs-create-plan"}, indent=2) + "\n"
    )


def test_no_write_for_non_worktree_path() -> None:
    """The isolation invariant: throwaway paths never gain a settings file.

    ``/tmp/work`` is the path the pre-QS-311 tests in this module pass.
    Both guards reject it — there is no
    ``.claude/agents/qs-create-plan.md`` there (guard 1) and no ``.git``
    *file* either (guard 2) — which is why AC4 holds with zero edits to
    those tests.

    Review-fix #01 S4: guard 2 used to be a bare ``utils.is_worktree``
    call, which is a *not-the-main-checkout* test and therefore returned
    ``True`` for ``/tmp/work``. The two guards were consequently NOT
    independent; the ``.git``-is-a-file containment check makes them so.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    claude_launcher.build_payload(
        "/tmp/work", 311, "Title", next_cmd="create-plan",
    )

    assert not (Path("/tmp/work") / SETTINGS_REL).exists()


# --------------------------------------------------------------------------- #
# Review fix plan #01 — hardening of the pin writer.
#
# M1  a READ failure must never rebuild the file (it holds user-local
#     permission state); a genuine rebuild backs the old bytes up
# M3  the write result is surfaced as ``phase_agent_pinned`` and every
#     skip path warns on stderr
# S1  the temp unlink is best-effort — it must not break the handoff
# S2  a re-read immediately before the publish shrinks the race against a
#     live Claude Code session rewriting the same file
# S4  guard 2 is a real containment check, not ``!= main checkout``
# S5  a literal ``null`` body is a non-object, and must warn
# S6  the write-failure branch is tested
# N5  the temp name carries the PID
# N6  the target's permission bits survive the replace
# --------------------------------------------------------------------------- #


def test_payload_reports_pin_true_on_success(tmp_path: Path) -> None:
    """M3: a successful write is reported as ``phase_agent_pinned: True``."""
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    payload = claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert payload["phase_agent_pinned"] is True


def test_payload_reports_pin_false_when_skipped(tmp_path: Path) -> None:
    """M3: a skipped write is reported, so the handoff can stop asserting the pin.

    Every GUI handoff block claims the worktree "should now be pinned".
    Before this key the orchestrator had no signal at all — the claim was
    unfalsifiable at the moment it was printed, and *deterministically
    false* for ``setup_task.py --no-worktree`` (which passes the main
    checkout as ``work_dir``).
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    payload = claude_launcher.build_payload(
        str(tmp_path), 311, "Title", next_cmd="create-plan",
    )

    assert payload["phase_agent_pinned"] is False


def test_warns_on_stderr_when_agent_file_absent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """M3: the guard-1 skip is observable — it was the one silent path."""
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    claude_launcher.build_payload(
        str(tmp_path), 311, "Title", next_cmd="create-plan",
    )

    captured = capsys.readouterr()
    assert captured.out == "", f"unexpected stdout output: {captured.out!r}"
    assert "warning:" in captured.err
    assert "qs-create-plan" in captured.err


def test_read_oserror_does_not_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """M1: a file we cannot READ is a file we must not replace.

    ``EACCES`` / ``EINTR`` / a lock / ``EIO`` on a network mount are
    transient conditions; treating them like corrupt JSON would replace a
    still-valid file — and this file carries the user's ``permissions``
    decisions (see ``test_merges_and_preserves_existing_keys``).

    ``read_bytes`` is the patch target because the reader takes bytes and
    decodes separately: ``UnicodeDecodeError`` is a ``ValueError``, not an
    ``OSError``, and must keep routing to the rebuild path.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    original = json.dumps({"permissions": {"allow": ["Bash(git status)"]}})
    (work_dir / SETTINGS_REL).write_text(original, encoding="utf-8")

    real_read_bytes = Path.read_bytes

    def _boom(self: Path) -> bytes:
        if self.name == "settings.local.json":
            raise PermissionError(13, "Permission denied")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _boom)

    payload = claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert payload["phase_agent_pinned"] is False
    assert (work_dir / SETTINGS_REL).read_text(encoding="utf-8") == original
    assert "warning:" in capsys.readouterr().err


def test_corrupt_file_is_backed_up_before_rebuild(tmp_path: Path) -> None:
    """M1: a rebuild is recoverable — the old bytes land in a ``.bak`` sibling."""
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    corrupt = '{"permissions": {"allow": ["Bash(git status)"]}'  # truncated
    (work_dir / SETTINGS_REL).write_text(corrupt, encoding="utf-8")

    claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert _settings(work_dir) == {"agent": "qs-create-plan"}
    backup = work_dir / ".claude" / "settings.local.json.bak"
    assert backup.is_file(), "corrupt settings were rebuilt with no backup"
    assert backup.read_text(encoding="utf-8") == corrupt


def test_null_json_is_overwritten_with_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """S5: ``null`` parses fine but is not an object — warn, then rebuild.

    The reader used to use ``loaded is not None`` as its parse-failed
    sentinel, so ``json.loads("null") -> None`` took neither the corrupt
    branch nor the non-object branch: the file was rewritten with an empty
    stderr. AC2 requires a non-``dict`` parse to be treated as corrupt,
    i.e. overwritten *with a warning*.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    (work_dir / SETTINGS_REL).write_text("null", encoding="utf-8")

    claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert _settings(work_dir) == {"agent": "qs-create-plan"}
    assert "warning:" in capsys.readouterr().err


def test_unlink_failure_does_not_break_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S1: the temp unlink is best-effort — ``EACCES`` there is not fatal.

    The unlink sits in a bare ``finally`` outside any ``except``, so an
    ``OSError`` from it used to propagate and break the handoff — the exact
    outcome the "must never break a handoff" contract forbids.
    ``missing_ok=True`` only suppresses ``FileNotFoundError``.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    real_unlink = Path.unlink

    def _boom(self: Path, *args: object, **kwargs: object) -> None:
        if self.name.endswith(".tmp"):
            raise PermissionError(13, "Permission denied")
        real_unlink(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "unlink", _boom)

    payload = claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert payload["phase_agent_pinned"] is True
    assert _settings(work_dir) == {"agent": "qs-create-plan"}


def test_write_failure_reports_unpinned_and_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """S6: the ``except OSError`` write branch warns, returns, and is reported."""
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    real_write_text = Path.write_text

    def _boom(self: Path, *args: object, **kwargs: object) -> int:
        if self.name.startswith("settings.local.json"):
            raise OSError(28, "No space left on device")
        return real_write_text(  # type: ignore[return-value]
            self, *args, **kwargs,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(Path, "write_text", _boom)

    payload = claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    # The handoff itself survives — the launcher one-liner is still built.
    assert payload["phase_agent_pinned"] is False
    assert payload["new_context"].startswith("sh ")
    assert not (work_dir / SETTINGS_REL).exists()
    assert "warning:" in capsys.readouterr().err


def test_reread_before_replace_preserves_late_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S2: a key that appears after the first read still survives the publish.

    The handoff normally runs from *inside* a live Claude Code session on
    the same worktree, and the app owns this file too. A read-modify-write
    with no re-read loses whatever the app wrote in between. The re-read
    immediately before ``os.replace`` shrinks — it does not close — that
    window; the residual race is documented in ``harness.md``'s Traps.

    The simulation hooks the temp write, which happens between the first
    read and the publish.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    target = work_dir / SETTINGS_REL
    target.write_text(json.dumps({"model": "opus"}), encoding="utf-8")

    real_write_text = Path.write_text
    fired: list[str] = []

    def _late_writer(self: Path, *args: object, **kwargs: object) -> int:
        if self.name.endswith(".tmp") and not fired:
            fired.append(self.name)
            # The live session lands a permission decision right here.
            real_write_text(
                target,
                json.dumps({"model": "opus", "permissions": {"allow": ["Bash(ls)"]}}),
                encoding="utf-8",
            )
        return real_write_text(  # type: ignore[return-value]
            self, *args, **kwargs,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(Path, "write_text", _late_writer)

    claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert fired, "the temp write never happened — the simulation is vacuous"
    assert _settings(work_dir) == {
        "model": "opus",
        "permissions": {"allow": ["Bash(ls)"]},
        "agent": "qs-create-plan",
    }


def test_temp_sibling_name_carries_the_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N5: a fixed temp name lets two concurrent handoffs clobber each other.

    With a shared name, A's ``os.replace`` can publish over B, or A's
    unlink can remove B's temp so B's ``os.replace`` raises
    ``FileNotFoundError`` and the *earlier* phase wins.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    seen: list[str] = []
    real_write_text = Path.write_text

    def _spy(self: Path, *args: object, **kwargs: object) -> int:
        if self.name.endswith(".tmp"):
            seen.append(self.name)
        return real_write_text(  # type: ignore[return-value]
            self, *args, **kwargs,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(Path, "write_text", _spy)

    claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert seen, "no temp sibling was written"
    assert all(str(os.getpid()) in name for name in seen), (
        f"temp name must carry the writer's PID; got {seen!r}"
    )


def test_file_mode_is_preserved_across_replace(tmp_path: Path) -> None:
    """N6: ``chmod 600`` survives the publish.

    ``write_text`` creates the temp at ``0o666 & ~umask`` and
    ``os.replace`` keeps the *temp's* mode, so a user who tightened the
    file (it can carry an ``env`` block with a token) had it silently
    republished world-readable.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    target = work_dir / SETTINGS_REL
    target.write_text(json.dumps({"model": "opus"}), encoding="utf-8")
    target.chmod(0o600)

    claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_second_clone_is_not_pinned(tmp_path: Path) -> None:
    """S4: a second full clone is a main checkout — guard 2 must reject it.

    ``utils.is_worktree`` cannot see this: ``get_main_worktree()`` takes no
    ``cwd``, so run from a cwd inside a different repo it reports the other
    repo's root, and the clone (or even quiet-solar's own main checkout)
    compares unequal → ``True``. A clone's ``.git`` is a *directory*, which
    is what guard 2 now checks.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    (work_dir / ".git").unlink()
    (work_dir / ".git").mkdir()

    payload = claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert payload["phase_agent_pinned"] is False
    assert not (work_dir / SETTINGS_REL).exists()


def test_is_worktree_semantics_are_not_containment(tmp_path: Path) -> None:
    """S4: ``utils.is_worktree`` means "not the main checkout" — nothing more.

    Documented rather than fixed: the launcher is its only caller, and the
    containment property the pin needs is supplied by guard 2's
    ``.git``-is-a-file check. Anything that reads the name as "is inside a
    worktree of this repo" is reading a claim the function never made.
    """
    import utils  # type: ignore[import-not-found]

    main = utils.get_main_worktree()
    assert utils.is_worktree(tmp_path) is True
    assert utils.is_worktree("/tmp/work") is True
    # Review-fix #02 N-e: the main-checkout case used to be spelled
    # ``is_worktree(get_main_worktree())``, i.e. structurally
    # ``p.resolve() != p.resolve()`` — true by construction, and equally
    # ``False`` if the call had raised. Pass the resolved path as a plain
    # string so the comparison is against a value, not against the
    # function's own output.
    assert utils.is_worktree(str(main.resolve())) is False
    assert str(main.resolve()) == str(main.resolve()), "sanity: path is stable"


# --------------------------------------------------------------------------- #
# Review fix plan #02 — the rebuild safety net and the publish sequence.
#
# A    the `.bak` copy must not leak what `_preserve_mode` protects
# B    `_backup`'s failure arm was the only untested new statement
# C    an empty body must not clobber a good `.bak`
# D    `os.replace` must not silently convert a symlinked target
# N-a  a FRESH settings file must not be born world-readable
# N-c  a destructive rebuild must be visible in the payload
# N-f  guard 2 must be shown doing work on its own
# --------------------------------------------------------------------------- #


def _corrupt(work_dir: Path, body: str = '{"env": {"TOKEN": "s3cr"}, ') -> Path:
    """Write an unparseable settings body and return the target path."""
    target = work_dir / SETTINGS_REL
    target.write_text(body, encoding="utf-8")
    return target


def test_backup_preserves_restrictive_mode(tmp_path: Path) -> None:
    """A: the `.bak` must carry the target's mode, not ``0o666 & ~umask``.

    `_backup` and `_preserve_mode` shipped in the same commit and cancelled
    each other out: the rebuild path is the one place the bytes are
    *copied* rather than replaced, so a `chmod 600` file holding an ``env``
    token was republished at ``0o644`` under a ``.bak`` name — exactly what
    the mode-preservation fix existed to prevent.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    target = _corrupt(work_dir)
    target.chmod(0o600)

    claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    backup = work_dir / ".claude" / "settings.local.json.bak"
    assert backup.is_file(), "the corrupt body was not backed up at all"
    assert "s3cr" in backup.read_text(encoding="utf-8")
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600, (
        f"the backup leaked the token at "
        f"{oct(stat.S_IMODE(backup.stat().st_mode))}"
    )


def test_backup_failure_warns_and_still_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """B: a `.bak` write failure warns and never breaks the handoff.

    The backup is a safety net, not a precondition: losing it is worth a
    warning, but the pin must still land or the phase goes unbound.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    _corrupt(work_dir)
    real_write_bytes = Path.write_bytes

    def _boom(self: Path, data: bytes) -> int:
        if self.name.endswith(".bak"):
            raise PermissionError(13, "Permission denied")
        return real_write_bytes(self, data)

    monkeypatch.setattr(Path, "write_bytes", _boom)

    payload = claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert payload["phase_agent_pinned"] is True
    assert _settings(work_dir) == {"agent": "qs-create-plan"}
    assert "warning:" in capsys.readouterr().err


def test_empty_body_does_not_clobber_existing_backup(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """C: a 0-byte target must not overwrite a good `.bak` with nothing.

    Reachable without contrivance: a `SIGKILL` or a full disk while Claude
    Code rewrites the file non-atomically leaves it empty, and the
    unconditional copy then destroyed the one recoverable copy of the
    user's approvals. The warning must say so distinctly — the generic
    "could not parse … rebuilding it" line does not.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    good = json.dumps({"permissions": {"allow": ["Bash(git status)"]}})
    backup = work_dir / ".claude" / "settings.local.json.bak"
    backup.write_text(good, encoding="utf-8")
    (work_dir / SETTINGS_REL).write_text("", encoding="utf-8")

    payload = claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert payload["phase_agent_pinned"] is True
    assert backup.read_text(encoding="utf-8") == good, (
        "the empty body clobbered a good backup"
    )
    err = capsys.readouterr().err
    assert "empty" in err, f"no distinct empty-body warning; got: {err!r}"


def test_symlinked_target_stays_a_symlink(tmp_path: Path) -> None:
    """D: the write must follow a symlink, not replace it with a file.

    Symlinking the pin at a shared file is the natural workaround for
    re-approving permissions in every fresh per-task worktree. ``os.replace``
    on the link silently converted it to a regular file: content looked
    right at handoff time, and the shared file simply never updated again —
    surfacing much later as "my approvals stopped syncing".
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    shared = tmp_path / "shared-settings.json"
    shared.write_text(json.dumps({"model": "opus"}), encoding="utf-8")
    link = work_dir / SETTINGS_REL
    link.symlink_to(shared)

    payload = claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert payload["phase_agent_pinned"] is True
    assert link.is_symlink(), "the handoff converted the symlink to a file"
    assert json.loads(shared.read_text(encoding="utf-8")) == {
        "model": "opus",
        "agent": "qs-create-plan",
    }
    assert not list(work_dir.glob(".claude/*.tmp")), "temp left in .claude"


def test_fresh_settings_file_is_created_private(tmp_path: Path) -> None:
    """N-a: an absent target must not yield a world-readable pin file.

    `_preserve_mode` returned early when the target was absent, and
    `worktree-setup.sh` never seeds a settings file — so *every* worktree
    took that path and the mode-preservation fix was dead code in practice.
    Worse, a later writer that preserves the existing mode then keeps
    ``0o644`` forever. Fail closed at ``0o600``.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    assert not (work_dir / SETTINGS_REL).exists(), "precondition: no target"

    claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    mode = stat.S_IMODE((work_dir / SETTINGS_REL).stat().st_mode)
    assert mode == 0o600, f"fresh pin file created at {oct(mode)}"


def test_payload_reports_settings_rebuilt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """N-c: a destructive rebuild must be visible in the payload.

    `phase_agent_pinned` is ``True`` on a run that just discarded the
    user's entire local settings, and the only other signal is a stderr
    line no orchestrator is told to relay. A hand-edited file with a
    trailing comma is enough to trigger it.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    _corrupt(work_dir, '{"model": "opus",}')

    payload = claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert payload["phase_agent_pinned"] is True
    assert payload["settings_rebuilt"] is True
    capsys.readouterr()


def test_payload_reports_settings_not_rebuilt_on_clean_merge(
    tmp_path: Path,
) -> None:
    """N-c: the flag is ``False`` for the ordinary merge path."""
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    (work_dir / SETTINGS_REL).write_text(
        json.dumps({"model": "opus"}), encoding="utf-8",
    )

    payload = claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert payload["settings_rebuilt"] is False
    # ... and on a skipped write there is nothing to rebuild.
    assert claude_launcher.build_payload(
        str(tmp_path / "nope"), 311, "Title", next_cmd="create-plan",
    )["settings_rebuilt"] is False


def test_guard_two_rejects_when_agent_file_present_but_no_git(
    tmp_path: Path,
) -> None:
    """N-f: guard 2 doing work on its own, with guard 1 satisfied.

    ``test_no_write_for_non_worktree_path`` passes ``/tmp/work``, which
    trips guard 1 first, so it can never show guard 2 rejecting anything.
    Here the agent file is present and only the ``.git`` pointer is
    missing — the isolated guard-2 case.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    (work_dir / ".git").unlink()
    assert (work_dir / ".claude" / "agents" / "qs-create-plan.md").is_file()

    payload = claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert payload["phase_agent_pinned"] is False
    assert not (work_dir / SETTINGS_REL).exists()
