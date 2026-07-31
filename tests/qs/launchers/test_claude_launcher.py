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


# --------------------------------------------------------------------------- #
# Review fix plan #03 B1 — a settings file we cannot parse is LEFT ALONE.
#
# Rounds 1 and 2 rebuilt it and kept a `.bak`. Three must-fix findings in a
# row came out of that safety net (a mode leak, a recoverability claim that
# could be false, and a read-only target producing an EMPTY backup while the
# original was discarded). Option B removes the destructive path outright:
# warn, write nothing, report `phase_agent_pinned: False`. The only cost is
# an unpinned GUI session on a worktree whose settings file is corrupt —
# `--agent` already covers that, and it is strictly better than destroying
# the user's `permissions.allow` list.
#
# Parametrized over every shape that reaches the "not a JSON object"
# conclusion, including the ones that motivated the deleted machinery.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "body"),
    [
        ("corrupt", "{not json at all"),
        ("truncated_with_token", '{"env": {"TOKEN": "s3cr"}, '),
        ("non_object_list", "[1, 2]"),
        ("non_object_string", '"x"'),
        ("null", "null"),
        ("empty", ""),
        ("nul_bytes", "\x00\x00\x00"),
    ],
)
def test_unparseable_settings_are_left_untouched_and_pin_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], name: str, body: str,
) -> None:
    """Every unusable body: bytes preserved, warning, no pin, no `.bak`."""
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    target = work_dir / SETTINGS_REL
    target.write_text(body, encoding="utf-8")
    before = target.read_bytes()

    payload = claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert payload["phase_agent_pinned"] is False, name
    assert target.read_bytes() == before, f"{name}: the user's bytes changed"
    assert "warning:" in capsys.readouterr().err, name
    assert not list((work_dir / ".claude").glob("*.bak")), (
        f"{name}: a .bak was created — Option B removed the backup path, and "
        f"a backup implies a rebuild that no longer happens"
    )


def test_bom_prefixed_settings_are_parsed_and_merged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """D1: a UTF-8 BOM is not corruption — parse it and merge normally.

    A BOM'd `settings.local.json` is perfectly valid JSON to most editors,
    and several write one by default on Windows. Decoding as plain
    ``utf-8`` leaves the BOM in the string, ``json.loads`` rejects it, and
    under the review-fix #03 refusal that means the worktree is skipped
    **permanently** — every future handoff re-reads the same file and
    re-refuses. ``utf-8-sig`` strips a leading BOM and is a no-op without
    one, so it is the correct codec for both.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    (work_dir / SETTINGS_REL).write_text(
        '\ufeff{"model": "opus"}', encoding="utf-8",
    )

    payload = claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert payload["phase_agent_pinned"] is True
    assert _settings(work_dir) == {"model": "opus", "agent": "qs-create-plan"}
    assert "warning:" not in capsys.readouterr().err


def test_corrupt_json_is_left_untouched_and_pin_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """AC2's named case: unparseable JSON. Named for the acceptance auditor."""
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    (work_dir / SETTINGS_REL).write_text("{not json at all", encoding="utf-8")

    payload = claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert payload["phase_agent_pinned"] is False
    assert (work_dir / SETTINGS_REL).read_text(encoding="utf-8") == (
        "{not json at all"
    )
    assert "warning:" in capsys.readouterr().err


def test_non_object_json_is_left_untouched_and_pin_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """AC2's named case: valid JSON of the wrong shape.

    Without this branch the shallow merge would raise ``TypeError`` /
    ``AttributeError`` *outside* the read/parse guard.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    (work_dir / SETTINGS_REL).write_text("[1, 2]", encoding="utf-8")

    payload = claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert payload["phase_agent_pinned"] is False
    assert (work_dir / SETTINGS_REL).read_text(encoding="utf-8") == "[1, 2]"
    assert "warning:" in capsys.readouterr().err


def test_null_json_is_left_untouched_and_pin_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """AC2's named case: ``null`` parses, is not an object, and must not write."""
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    (work_dir / SETTINGS_REL).write_text("null", encoding="utf-8")

    payload = claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert payload["phase_agent_pinned"] is False
    assert (work_dir / SETTINGS_REL).read_text(encoding="utf-8") == "null"
    assert "warning:" in capsys.readouterr().err


def test_no_settings_rebuilt_key_in_payload(tmp_path: Path) -> None:
    """B1: the key is gone. Nothing may claim a rebuild happened.

    It existed only to report the destructive path. With that path deleted,
    a surviving key would be an unread payload field — the exact thing
    Decision 8 declined `gui_context` for.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    payload = claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert "settings_rebuilt" not in payload


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


def test_no_write_for_non_worktree_path(tmp_path: Path) -> None:
    """The isolation invariant: throwaway paths never gain a settings file.

    ``/tmp/work`` is the path the pre-QS-311 tests in this module pass.
    Both guards reject it — there is no
    ``.claude/agents/qs-create-plan.md`` there (guard 1) and no ``.git``
    *file* either (guard 2) — which is why AC4 holds with zero edits to
    those tests.

    Review-fix #03 C5: the assertion runs against a ``tmp_path`` stand-in
    rather than the real global ``/tmp/work``. A leftover
    ``/tmp/work/.claude/agents/qs-create-plan.md`` on any machine — this
    module's own earlier runs could not create one, but a developer
    experimenting could — would have satisfied guard 1 and failed the test
    for a reason that has nothing to do with the invariant.

    Review-fix #01 S4: guard 2 used to be a bare ``utils.is_worktree``
    call, which is a *not-the-main-checkout* test and therefore returned
    ``True`` for ``/tmp/work``. The two guards were consequently NOT
    independent; the ``.git``-is-a-file containment check makes them so.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    throwaway = tmp_path / "work"
    throwaway.mkdir()

    claude_launcher.build_payload(
        str(throwaway), 311, "Title", next_cmd="create-plan",
    )

    assert not (throwaway / SETTINGS_REL).exists()


# --------------------------------------------------------------------------- #
# The pin writer's failure behaviour.
#
# What the writer guarantees, and what each test below holds it to:
#
# * A file it cannot read or cannot parse as a JSON object is **left
#   byte-identical** and the pin is skipped, with a warning naming the file
#   and the remedy. There is no rebuild and no backup.
# * A symlink at the pin file, at ``.claude``, or at the temp sibling is
#   refused outright. Those are the three paths the writer touches, so with
#   all three refused the writes stay inside ``work_dir/.claude``.
# * The result is reported as ``phase_agent_pinned``; every skip path warns
#   on stderr, and none of them can break the handoff.
# * The publish is atomic (temp sibling + ``os.replace``): the temp name
#   carries the PID, its cleanup is suppressed, the target's mode is
#   carried over, and a fresh file is created ``0o600``.
# * A re-read immediately before the publish shrinks the race against a
#   live Claude Code session rewriting the same file.
#
# Written as a statement of the contract rather than a changelog of the
# five review rounds that arrived at it (review-fix #04 D4 / D12) — the
# rounds are recorded in the story. Earlier revisions of this header
# advertised a `.bak` rebuild path that no longer exists, which is the
# first thing a reviewer reads.
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
    ``OSError``. Both now end in the same place — leave the file alone and
    skip the pin — but only ``OSError`` is the transient case, so the two
    keep distinct warning text. (Review-fix #04 D5: this said "must keep
    routing to the rebuild path"; there is no rebuild path any more.)
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
            # The live session lands a permission decision right here —
            # after our first read, before the pre-publish re-read.
            real_write_text(
                target,
                json.dumps(
                    {"model": "opus", "permissions": {"allow": ["Bash(ls)"]}},
                ),
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

    ``Path.write_text`` creates the temp at ``0o666 & ~umask`` and
    ``os.replace`` keeps the *temp's* mode, so without an explicit
    ``shutil.copymode`` a user who tightened the file (it can carry an
    ``env`` block with a token) had it silently republished world-readable.
    Verified rather than assumed (review-fix #04 D5): after C1 the writer
    is back to ``write_text``, so this description is accurate again.
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
    import subprocess

    import utils  # type: ignore[import-not-found]

    # Review-fix #03 C2: derive the main checkout WITHOUT calling the
    # function under test. Round 2's attempt still passed
    # ``get_main_worktree()``'s own output back in, so the assertion stayed
    # ``p.resolve() != p.resolve()`` — true by construction — and the
    # "comparison is against a value" comment was simply wrong. ``git
    # rev-parse --git-common-dir`` names the shared git dir; its parent is
    # the main checkout, computed here from the test file's location.
    #
    # Review-fix #04 M1: this used to also assert ``main_checkout != repo``
    # as a precondition. That is true in a linked worktree and FALSE in a
    # plain clone, so it broke CI — which is the only place this assertion
    # runs against a plain clone, and therefore the last place it should
    # break. The property below needs no such precondition, and skipping
    # instead of deleting would have hidden it exactly where it matters.
    repo = Path(__file__).resolve().parents[3]
    common = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--path-format=absolute",
         "--git-common-dir"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    main_checkout = Path(common).parent.resolve()
    assert (main_checkout / ".git").is_dir(), (
        f"derived main checkout {main_checkout} has no .git directory — the "
        f"derivation, not is_worktree, is broken"
    )
    assert utils.is_worktree(tmp_path) is True
    assert utils.is_worktree(str(main_checkout)) is False


# --------------------------------------------------------------------------- #
# Refusals and permissions.
#
# The two shapes the writer refuses (a symlinked pin file, a symlinked
# ``.claude`` directory) and the two permission properties it maintains (a
# fresh file is ``0o600``; an existing mode is carried over, including onto
# the temp before it is published). Plus guard 2 shown rejecting on its own,
# with guard 1 satisfied.
# --------------------------------------------------------------------------- #


def test_symlinked_target_is_refused_not_followed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """B2: a symlinked pin file is refused, not resolved.

    Round 2 followed the link with ``target.resolve()`` and never re-applied
    the containment guard, so every write destination became arbitrary: the
    pin, its temp and its `.bak` could land in ``~/.claude/settings.json``
    (user scope — every project, every headless run) or in the **main
    checkout**, which `harness.md` simultaneously promises is never pinned.
    Refusing keeps guard 2's invariant unconditional: every destination is a
    literal path inside ``work_dir/.claude``.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    shared = tmp_path / "shared-settings.json"
    original = json.dumps({"model": "opus"})
    shared.write_text(original, encoding="utf-8")
    link = work_dir / SETTINGS_REL
    link.symlink_to(shared)

    payload = claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert payload["phase_agent_pinned"] is False
    assert link.is_symlink(), "the link itself must be left alone"
    assert shared.read_text(encoding="utf-8") == original, (
        "the link's target was written through"
    )
    err = capsys.readouterr().err
    assert "warning:" in err and "symlink" in err, (
        f"the refusal must be observable on stderr; got: {err!r}"
    )
    # Nothing may be written beside EITHER end of the link.
    assert not list((work_dir / ".claude").glob("settings.local.json.*"))
    assert not list(tmp_path.glob("shared-settings.json.*"))


def test_symlinked_temp_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """M2: a symlink at the **temp** name is refused too.

    The refusal covered ``.claude`` and the pin file but not the temp, and
    ``Path.write_text`` follows symlinks. Reachable without contrivance:
    ``.gitignore`` itself documents that a ``SIGKILL`` or OOM kill can leave
    a temp behind, and the name is PID-derived, so a reused PID lands on it.

    The consequence was the worst of the round: the merged settings — an
    ``env`` token included — were written **outside the worktree**,
    ``os.replace`` then renamed the *link* onto ``settings.local.json``, the
    call reported ``phase_agent_pinned: True``, and every later handoff hit
    the pin-file refusal, leaving the worktree permanently unpinnable.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    target = work_dir / SETTINGS_REL
    target.write_text(
        json.dumps({"env": {"TOKEN": "s3cr"}}), encoding="utf-8",
    )
    outside = tmp_path / "outside-the-worktree.json"
    outside.write_text("untouched\n", encoding="utf-8")
    # The temp name the writer will choose: PID-derived, same as the code.
    planted = work_dir / ".claude" / f"settings.local.json.{os.getpid()}.tmp"
    planted.symlink_to(outside)

    payload = claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert payload["phase_agent_pinned"] is False
    assert outside.read_text(encoding="utf-8") == "untouched\n", (
        "the merged settings were written outside the worktree"
    )
    assert not target.is_symlink(), "os.replace renamed the link onto the pin"
    assert "s3cr" in target.read_text(encoding="utf-8"), (
        "the original settings should be untouched"
    )
    err = capsys.readouterr().err
    assert "warning:" in err and "symlink" in err


def test_symlinked_claude_dir_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """C2: a symlinked ``.claude`` **directory** is refused too.

    The round-3 refusal checked only the pin file. With ``.claude`` itself a
    link — pointing at the main checkout's ``.claude``, which is a plausible
    way to share one agents dir — guard 1 passes (the agent file is there),
    ``target.is_symlink()`` is ``False`` because the *file* is not a link,
    and the write lands in the **main checkout** while reporting
    ``phase_agent_pinned: True``. That directly falsifies `harness.md` →
    Traps, which promises the main checkout is never pinned. The same shape
    reaches ``~/.claude``.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    # A stand-in "main checkout" .claude, complete with the agent file so
    # guard 1 is satisfied through the link.
    shared = tmp_path / "main-checkout" / ".claude"
    (shared / "agents").mkdir(parents=True)
    (shared / "agents" / "qs-create-plan.md").write_text("# stub\n")
    existing = json.dumps({"permissions": {"allow": ["Bash(git status)"]}})
    (shared / "settings.local.json").write_text(existing, encoding="utf-8")

    work_dir = tmp_path / "wt"
    work_dir.mkdir()
    (work_dir / ".git").write_text("gitdir: /nowhere/.git/worktrees/wt\n")
    (work_dir / ".claude").symlink_to(shared)

    payload = claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert payload["phase_agent_pinned"] is False
    assert (shared / "settings.local.json").read_text(encoding="utf-8") == (
        existing
    ), "the shared/main-checkout settings file was written through the link"
    assert not list(shared.glob("settings.local.json.*")), "temp leaked"
    err = capsys.readouterr().err
    assert "warning:" in err and "symlink" in err, (
        f"the refusal must be observable on stderr; got: {err!r}"
    )


def test_published_pin_carries_the_target_mode(tmp_path: Path) -> None:
    """The published pin keeps the mode the user chose — ``copymode``'s job.

    Renamed and restored in review-fix #05 (S1 + S5). Two things were wrong
    with the previous version:

    * It was named ``test_temp_is_never_world_readable`` and asserted a
      pre-publish privacy window. Review-fix #04's C1 deliberately gave that
      window up: the temp is created by ``write_text`` at
      ``0o666 & ~umask`` **already containing** the merged content, and
      narrowed afterwards. The name promised a property the code no longer
      has, and #05's M1 deletes the matching doc sentence.
    * It had been narrowed until it could not fail — the fixture mode was
      changed to ``0o600`` and the published-mode assertion dropped — which
      is why nothing caught that stale claim.

    No monkeypatching: the property is observable on the published file
    after ``build_payload`` returns. The previous version hooked
    ``os.replace`` through ``claude_launcher.os``, which is the **real**
    ``os`` module (the launcher keeps no alias), so the syscall was globally
    replaced for the test's duration.

    The fixture is ``0o664`` and the **umask is pinned to 022** for the call
    (review-fix #06 F3). Both halves matter:

    * ``0o644`` — what the fix plan originally named — is exactly what
      ``write_text`` produces under ``umask 022``, so the assertion would
      pass whether or not ``copymode`` ran at all.
    * ``0o664`` alone is not enough either: under ``umask 002``, the default
      for regular users on Debian/Ubuntu-family images, ``write_text``
      produces ``0o664`` and the assertion goes vacuous again. Verified
      empirically by the reviewer — with ``copymode`` removed and
      ``umask 002``, this test passed while its ``0o600`` neighbour failed.

    With the umask pinned, ``0o664`` is reliably *wider* than the temp's
    default, so this covers ``copymode``'s **widening** direction, which no
    other test does; ``test_file_mode_is_preserved_across_replace`` covers
    narrowing (``0o600``). ``os.umask`` is process-global, and pytest runs
    tests sequentially within an xdist worker, so setting and restoring it
    around the call is safe.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    target = work_dir / SETTINGS_REL
    target.write_text(
        json.dumps({"env": {"TOKEN": "s3cr"}}), encoding="utf-8",
    )
    target.chmod(0o664)

    previous_umask = os.umask(0o022)
    try:
        payload = claude_launcher.build_payload(
            str(work_dir), 311, "Title", next_cmd="create-plan",
        )
    finally:
        os.umask(previous_umask)

    assert payload["phase_agent_pinned"] is True
    assert stat.S_IMODE(target.stat().st_mode) == 0o664, (
        f"the target's mode was not carried onto the published file; got "
        f"{oct(stat.S_IMODE(target.stat().st_mode))}"
    )
    # And the merge really happened, so the mode above is the mode of the
    # NEW file rather than of an untouched original.
    assert _settings(work_dir) == {
        "env": {"TOKEN": "s3cr"},
        "agent": "qs-create-plan",
    }


def test_mode_failure_still_publishes_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """S2: a ``copymode`` failure degrades the mode, it does not refuse the pin.

    Review-fix #04's C1 dropped the ``contextlib.suppress(OSError)`` that the
    deleted ``_preserve_mode`` had around the mode call, so an ``EPERM`` from
    ``shutil.copymode`` fell into the outer handler and the pin was skipped —
    even though the content was already written and ``os.replace`` needs only
    directory permission. On a chmod-hostile mount (exFAT, CIFS ``noperm``,
    some Docker volumes) or a settings file owned by another uid, that makes
    the worktree unpinnable *forever*, and per `harness.md` Traps the
    resulting stale pin looks intentional.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    target = work_dir / SETTINGS_REL
    target.write_text(json.dumps({"model": "opus"}), encoding="utf-8")

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(claude_launcher.shutil, "copymode", _boom)

    payload = claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert payload["phase_agent_pinned"] is True
    assert _settings(work_dir) == {"model": "opus", "agent": "qs-create-plan"}
    # Review-fix #06 F2: the degrade must be OBSERVABLE. Silence here is
    # sticky — one failure publishes at ``0o666 & ~umask``, Claude Code may
    # then persist an ``env`` token into that same file, and every later
    # handoff copies the widened mode forward, so a transient failure would
    # leave a secrets-bearing file world-readable for good while reporting
    # success.
    err = capsys.readouterr().err
    assert "warning:" in err, f"the mode failure was silent; stderr: {err!r}"
    assert "mode" in err


def test_late_non_object_keeps_the_first_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S3: the late read yielding a non-object keeps the first render.

    ``_late_render``'s ``isinstance`` arm was uncovered for two rounds. It is
    **reachable** — verified by this test: every *first*-read non-object is
    short-circuited in ``_read_settings``, but an external writer replacing
    the file between the two reads produces exactly this case. So the arm is
    covered rather than deleted (the fix plan's two options), because
    deleting it would let a shallow merge run against a list.
    """
    from launchers import claude as claude_launcher  # type: ignore[import-not-found]

    work_dir = _fake_worktree(tmp_path)
    target = work_dir / SETTINGS_REL
    target.write_text(json.dumps({"model": "opus"}), encoding="utf-8")

    real_write_text = Path.write_text
    fired: list[str] = []

    def _late_non_object(self: Path, *args: object, **kwargs: object) -> int:
        if self.name.endswith(".tmp") and not fired:
            fired.append(self.name)
            real_write_text(target, "[1, 2]", encoding="utf-8")
        return real_write_text(  # type: ignore[return-value]
            self, *args, **kwargs,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(Path, "write_text", _late_non_object)

    payload = claude_launcher.build_payload(
        str(work_dir), 311, "Title", next_cmd="create-plan",
    )

    assert fired, "the late window never opened — the simulation is vacuous"
    assert payload["phase_agent_pinned"] is True
    # The first render stands: the list is discarded, not merged onto.
    assert _settings(work_dir) == {"model": "opus", "agent": "qs-create-plan"}


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
