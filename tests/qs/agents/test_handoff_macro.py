"""QS-372 — the handoff is written once (``_macros.md.j2::handoff``) and the
implement variant is routed by the declared target (``implement_routing``).

- AC2: every Claude handoff site prints the launcher's ``handoff_text``
  verbatim, passes the right ``--next-cmd``, and carries the generated
  "Before running" sentence; the GUI prose no longer lives in the agents.
- AC3: every OpenCode handoff site carries the canonical auto-execute
  contract, clause by clause and in order.
- AC4: create-plan / review-task route the implement variant from
  ``target`` and STOP with a stable message when it is unusable.
- AC5: no path-list routing phrase survives in any rendered orchestrator
  (unbound + bound renders, both harnesses) or in ``.claude/commands``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.qs.agents._rendered import REPO_ROOT, agents_dir

import render_agents  # type: ignore[import-not-found]  # isort: skip

_HARNESSES = ("claude", "opencode")

# The closed list of 8 handoff sites (story §3), in file order:
# (next_cmd, fix_plan, rerun).
_SITES: dict[str, list[tuple[str, bool, str]]] = {
    "qs-create-plan.md": [("{{NEXT_PHASE}}", False, "")],
    "qs-diagnose-task.md": [("{{NEXT_PHASE}}", False, "")],
    "qs-implement-task.md": [("{{NEXT_PHASE}}", False, "")],
    "qs-implement-setup-task.md": [("review-task", False, "")],
    "qs-review-task.md": [
        ("finish-task", False, ""),
        ("{{next_implement}}", True, "review-task"),
    ],
    "qs-verify-task.md": [
        ("finish-task", False, ""),
        ("implement-task", True, "verify-task"),
    ],
}

_VERBATIM = "`handoff_text` **verbatim**"
_BEFORE_RUNNING = "**Before running** — substitute"
_NEXT_CMD_RE = re.compile(r'--next-cmd "([^"]+)"')
_BASH_FENCE_RE = re.compile(r"^```bash\n(.*?)\n^```$", re.DOTALL | re.MULTILINE)


def _norm(text: str) -> str:
    return " ".join(text.split())


def _body(harness: str, filename: str, mode: str = "unbound") -> str:
    return (agents_dir(harness, mode) / filename).read_text(encoding="utf-8")


def _paragraphs(body: str) -> list[str]:
    return re.split(r"\n[ \t]*\n", body)


def _before_running(body: str) -> list[str]:
    return [p for p in _paragraphs(body) if p.startswith(_BEFORE_RUNNING)]


# --------------------------------------------------------------------------- #
# AC2
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("filename", sorted(_SITES))
def test_claude_sites_print_handoff_text_verbatim(filename: str) -> None:
    """One "print ``handoff_text`` verbatim" instruction per handoff site."""
    body = _body("claude", filename)
    assert body.count(_VERBATIM) == len(_SITES[filename])


@pytest.mark.parametrize("harness", _HARNESSES)
@pytest.mark.parametrize("filename", sorted(_SITES))
def test_sites_pass_the_expected_next_cmd(harness: str, filename: str) -> None:
    """Each site's ``--next-cmd`` matches §3, and every callsite fence is at column 0."""
    body = _body(harness, filename)
    fences = [f for f in _BASH_FENCE_RE.findall(body) if "scripts/qs/next_step.py" in f]
    got = [m for f in fences for m in _NEXT_CMD_RE.findall(f)]
    assert got == [site[0] for site in _SITES[filename]]
    # No indented (list-nested) next_step.py invocation left behind.
    indented = [
        line for line in body.splitlines()
        if "python scripts/qs/next_step.py" in line and line[:1].isspace()
    ]
    assert not indented, f"{filename}: indented next_step.py callsite(s): {indented}"


@pytest.mark.parametrize("harness", _HARNESSES)
@pytest.mark.parametrize("filename", sorted(_SITES))
def test_sites_carry_the_generated_before_running_sentence(
    harness: str, filename: str,
) -> None:
    """The macro-generated sentence names every placeholder of its site."""
    paras = _before_running(_body(harness, filename))
    assert len(paras) == len(_SITES[filename])
    for para, (next_cmd, fix_plan, _rerun) in zip(paras, _SITES[filename], strict=True):
        text = _norm(para)
        for placeholder in ("{{worktree}}", "{{issue}}", "{{title}}"):
            assert f"`{placeholder}`" in text
        for placeholder in ("{{fix_plan_path}}", "{{pr_number}}"):
            assert (f"`{placeholder}`" in text) is fix_plan
        clause = f"substitute `{next_cmd}` with the value you resolved above"
        assert (clause in text) is next_cmd.startswith("{{"), text


@pytest.mark.parametrize("filename", sorted(_SITES))
def test_claude_sites_carry_no_gui_prose(filename: str) -> None:
    """The GUI block and its stale-pin rule now live in ``launchers/claude.py``."""
    body = _body("claude", filename)
    assert "[Claude Code GUI]" not in body
    assert "On `false` the worktree may still carry" not in _norm(body)


@pytest.mark.parametrize("filename", sorted(_SITES))
def test_claude_sites_stop_on_launcher_error(filename: str) -> None:
    """AC6 (4) + S1: a non-zero exit, no JSON, an ``error`` key, or a missing
    ``handoff_text`` all make the Claude agent STOP (OpenCode parity)."""
    text = _norm(_body("claude", filename))
    expected = (
        "If the command exits non-zero or prints no JSON, if the JSON "
        "contains an `error` key, or if it has no `handoff_text`, STOP and "
        "print the raw output verbatim."
    )
    assert text.count(expected) == len(_SITES[filename])
    # S1: the error-key STOP is still semantically present at every site.
    assert text.count("contains an `error` key") == len(_SITES[filename])
    # S1: every Claude site carries the ``handoff_text``-missing STOP.
    assert text.count("or if it has no `handoff_text`, STOP") == len(_SITES[filename])


def test_claude_fix_plan_sites_carry_the_rerun_trailer() -> None:
    """Sites 6 and 8 end their report with the re-run trailer."""
    for filename, rerun in (("qs-review-task.md", "review-task"), ("qs-verify-task.md", "verify-task")):
        text = _norm(_body("claude", filename))
        assert (
            f"Then re-run /{rerun} (or open a fresh `claude --agent qs-{rerun}` "
            f"session) to verify." in text
        )


def test_review_task_captures_worktree() -> None:
    """The generated sentence names ``{{worktree}}``, so review-task must capture it."""
    for harness in _HARNESSES:
        paras = [p for p in _paragraphs(_body(harness, "qs-review-task.md")) if p.startswith("Capture `issue`")]
        assert len(paras) == 1
        assert "`worktree`" in paras[0]


# --------------------------------------------------------------------------- #
# AC3
# --------------------------------------------------------------------------- #

_FAILURE_MODES = (
    "``agent_file_missing``",
    "``agent_file_unreadable``",
    "``agent_file_empty``",
    "``worktree_invalid``",
    "``fallback_cli``",
    "``fallback_unavailable``",
    "``session_orphaned``",
)
_SECTION_END = "``scripts/qs/spawn_session.py``)."


def _opencode_sections(filename: str) -> list[str]:
    """Slice each OpenCode handoff section: "Before running" → the failure-mode list."""
    body = _body("opencode", filename)
    starts = [m.start() for m in re.finditer(re.escape(_BEFORE_RUNNING), body)]
    sections = []
    for start in starts:
        end = body.index(_SECTION_END, start) + len(_SECTION_END)
        sections.append(body[start:end])
    return sections


def _clauses(next_cmd: str, fix_plan: bool, rerun: str) -> list[str]:
    clauses = [
        _BEFORE_RUNNING,
        "--harness opencode",
        "Parse the JSON output of ``next_step.py``.",
        "**If `next_step.py` exits non-zero or prints no JSON, or its JSON "
        "has no `new_context`**, STOP",
        "**If the `next_step.py` JSON contains an `error` key**, STOP",
        "Otherwise capture the ``new_context`` string",
    ]
    if fix_plan:
        clauses += [
            "it is NOT a session-spawn command. Do NOT execute it.",
            "omit the \"Already running an implementation session?\" block",
        ]
    clauses += [
        "**Run `new_context` via the Bash tool**",
        "python scripts/qs/spawn_session.py --agent qs-<phase>",
        "Do NOT extract only the prompt",
        "Do NOT strip ``--agent qs-<phase>``",
        "POST /session/<id>/prompt_async",
        "**If the Bash tool returns an error before producing any JSON output**",
        "The success contract is **binary**:",
        '``status == "session_created"``',
        "equals `qs-` followed by the phase name passed to `--next-cmd` "
        f"(here: `qs-{next_cmd}`)",
        f"[OK] Next phase session created: qs-{next_cmd}",
        "(visible in the OpenCode session list on the left)",
    ]
    if fix_plan:
        clauses += [
            "Already running an implementation session?",
            "Paste this prompt into it:",
            "{{existing_session_prompt}}",
        ]
    if rerun:
        clauses.append(f"Then re-activate `qs-{rerun}` (or open a fresh session bound to it) to verify.")
    if fix_plan:
        clauses.append(
            "(Omit the \"Already running an implementation session?\" block when "
            "`existing_session_prompt` is missing or null.)"
        )
    clauses += ["**Anything else**", *_FAILURE_MODES, _SECTION_END]
    return clauses


@pytest.mark.parametrize("filename", sorted(_SITES))
def test_opencode_sites_carry_the_contract_in_order(filename: str) -> None:
    """Every §2 OpenCode clause appears, in order, in each handoff section."""
    sections = _opencode_sections(filename)
    assert len(sections) == len(_SITES[filename])
    for section, site in zip(sections, _SITES[filename], strict=True):
        text = _norm(section)
        pos = 0
        for clause in _clauses(*site):
            found = text.find(_norm(clause), pos)
            assert found >= 0, f"{filename} site {site[0]}: missing/out of order: {clause!r}"
            pos = found + 1


@pytest.mark.parametrize(("filename", "index"), [("qs-review-task.md", 1), ("qs-verify-task.md", 1)])
def test_opencode_omit_note_sits_after_the_report_fence(filename: str, index: int) -> None:
    """The omit-when-null note is prose after the fence, never inside the report."""
    section = _opencode_sections(filename)[index]
    assert "  ```\n\n  (Omit the \"Already running an implementation session?\" block" in section


@pytest.mark.parametrize("filename", sorted(_SITES))
def test_opencode_banner_lines_are_indented_inside_the_report(filename: str) -> None:
    """``[OK]`` banner lines sit inside the list-item report (2-space indent)."""
    body = _body("opencode", filename)
    ok_lines = [line for line in body.splitlines() if "[OK]" in line]
    for line in ok_lines:
        if line.lstrip().startswith("[OK]"):
            assert line.startswith("  [OK]"), line


@pytest.mark.parametrize("filename", sorted(_SITES))
def test_opencode_lsp_section_survives(filename: str) -> None:
    """The OpenCode LSP section is present once, outside any handoff section."""
    body = _body("opencode", filename)
    assert body.count("## Code intelligence (LSP)") == 1
    last_end = body.rindex(_SECTION_END)
    assert body.index("## Code intelligence (LSP)") > last_end


# --------------------------------------------------------------------------- #
# AC4
# --------------------------------------------------------------------------- #

_STOP_PREFIX = "Cannot pick the implement variant:"


@pytest.mark.parametrize("harness", _HARNESSES)
@pytest.mark.parametrize(
    ("filename", "var"),
    [("qs-create-plan.md", "NEXT_PHASE"), ("qs-review-task.md", "{{next_implement}}")],
)
def test_implement_variant_routes_by_declared_target(
    harness: str, filename: str, var: str,
) -> None:
    raw = _body(harness, filename)
    text = _norm(raw)
    assert "Route by the **declared target** (QS-321" in text
    assert "re-run `python scripts/qs/context.py` and take `target`" in text
    assert f"`factory` → `{var} = implement-setup-task`" in text
    assert f"`product` → `{var} = implement-task`" in text
    # S2: the new STOP message literal (still carrying the stable prefix).
    assert (
        f"{_STOP_PREFIX} issue #{{{{issue}}}} has no single target:* label. "
        "Label it with exactly one of target:factory or target:product "
        "(see the commands below), or tell me which variant to run."
    ) in text
    # M1: the agent prints the commands to the user but must not run them —
    # picking a label itself is the QS-321-forbidden inference.
    assert "Do NOT run any of them yourself" in text
    assert "Then apply exactly one" not in text
    # S2/M1: two separate, concrete, runnable add-label commands, each in its
    # own bash fence — copying one fence never applies both labels.
    assert "gh issue edit {{issue}} --add-label target:factory" in text
    assert "gh issue edit {{issue}} --add-label target:product" in text
    assert " or target:product)" not in text
    add_fences = [f for f in _BASH_FENCE_RE.findall(raw) if "--add-label" in f]
    assert len(add_fences) == 2
    for fence in add_fences:
        assert not ("target:factory" in fence and "target:product" in fence)
    # M1: both --remove-label remedies are concrete labels; the non-pasteable
    # `target:<wrong>` placeholder (a shell redirect) is gone.
    assert "gh issue edit {{issue}} --remove-label target:factory" in text
    assert "gh issue edit {{issue}} --remove-label target:product" in text
    assert "target:<wrong>" not in text
    # M1: a hard context.py failure (non-zero exit / no JSON) is now covered.
    assert "If `context.py` exits non-zero / prints no JSON" in text
    assert "the `gh` lookup" in text
    # S2: only the two implement variants are accepted, not any phase name.
    assert (
        "if they name `implement-task` or `implement-setup-task`, use it; "
        "any other answer → ask again"
    ) in text
    assert "if they report adding a label, re-run `context.py` and route again" in text
    # Bare phase names only — never a slash-form assignment.
    assert f"`{var} = /implement" not in text


@pytest.mark.parametrize(
    ("harness", "expected"),
    [
        ("claude", "review-task → `/{{next_implement}}` is the"),
        ("opencode", "review-task → `{{next_implement}}` is the"),
    ],
)
def test_review_task_loop_prose_uses_slash_on_claude_only(
    harness: str, expected: str,
) -> None:
    """N3: the loop-describing prose renders the slash form on Claude only —
    it is prose, not a `= implement-…` assignment, so AC4's no-slash rule
    (which targets assignments) still holds."""
    assert expected in _body(harness, "qs-review-task.md")


# --------------------------------------------------------------------------- #
# AC5
# --------------------------------------------------------------------------- #

_PATH_ROUTING_PHRASES = (
    "If **all** are in",
    "if all touched files are in",
    "If **every** touched file is under",
    "chosen by file scope",
    "chosen by the file scope",
    "based on the file paths in its task breakdown",
    "Inspect the file paths your task breakdown will touch",
)

# ``_rendered`` has put ``scripts/qs`` on ``sys.path``.
_ORCHESTRATOR_FILES = sorted(f"{stem}.md" for stem in render_agents.ORCHESTRATORS)


def test_orchestrator_file_list_is_populated() -> None:
    assert len(_ORCHESTRATOR_FILES) == 9


@pytest.mark.parametrize("mode", ["unbound", "bound"])
@pytest.mark.parametrize("harness", _HARNESSES)
@pytest.mark.parametrize("filename", _ORCHESTRATOR_FILES)
def test_no_path_list_routing_in_rendered_orchestrators(
    mode: str, harness: str, filename: str,
) -> None:
    text = _norm(_body(harness, filename, mode))
    hits = [p for p in _PATH_ROUTING_PHRASES if p in text]
    assert not hits, f"{harness}/{mode}/{filename}: path-list routing phrase(s) {hits}"


@pytest.mark.parametrize(
    "command", sorted((REPO_ROOT / ".claude" / "commands").glob("*.md")), ids=lambda p: p.name,
)
def test_no_path_list_routing_in_slash_commands(command: Path) -> None:
    text = _norm(command.read_text(encoding="utf-8"))
    hits = [p for p in _PATH_ROUTING_PHRASES if p in text]
    assert not hits, f"{command.name}: path-list routing phrase(s) {hits}"
