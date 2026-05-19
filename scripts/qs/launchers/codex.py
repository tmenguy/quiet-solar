"""Launcher payload for OpenAI Codex.

Stub — Codex's CLI / agent-spawn surface is in flux. When we settle on
the actual command (e.g. ``codex --agent <name> --cwd <dir>``), replace
the contents of ``build_payload`` below.
"""

from __future__ import annotations

from typing import Literal

from launchers.phases import (  # type: ignore[import-not-found]
    build_existing_session_prompt,
)

# ``caller`` literal — reserved for harness-specific bifurcation
# (the OpenCode launcher uses it to switch between HTTP-API and
# CLI-form payloads; Codex is a stub today). Kept as a no-op kwarg so
# all launchers can be dispatched uniformly from ``setup_task.py`` and
# ``next_step.py``.
Caller = Literal["setup_task", "next_step"]


def build_payload(
    work_dir: str,
    issue: int | str,
    title: str,
    *,
    next_cmd: str,
    next_prompt: str | None = None,
    caller: Caller = "next_step",
    fix_plan_path: str | None = None,
    pr_number: int | None = None,
) -> dict:
    """Return a placeholder launcher payload for Codex.

    The ``caller`` kwarg is reserved for harness-specific bifurcation
    (used by the OpenCode launcher). Codex is a stub today and
    accepts/ignores it.

    ``fix_plan_path`` + ``pr_number`` populate an
    ``existing_session_prompt`` field when both are provided — the
    paste-into-existing-session prompt for the review-task →
    implement-task common loop. See
    ``launchers/phases.py::build_existing_session_prompt``.
    """
    del title  # unused
    del caller  # reserved for harness-specific bifurcation
    payload: dict = {
        "tool": "codex",
        "same_context": next_cmd,
        "new_context": (
            f"[Codex launcher not implemented yet]\n"
            f"Manually open a Codex session in {work_dir} and run {next_cmd!r}."
            + (f"\nInitial prompt: {next_prompt}" if next_prompt else "")
        ),
        "issue": issue,
        "work_dir": work_dir,
    }
    existing_prompt = build_existing_session_prompt(
        work_dir, fix_plan_path, pr_number,
    )
    if existing_prompt is not None:
        payload["existing_session_prompt"] = existing_prompt
    return payload
