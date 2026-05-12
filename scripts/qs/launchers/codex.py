"""Launcher payload for OpenAI Codex.

Stub — Codex's CLI / agent-spawn surface is in flux. When we settle on
the actual command (e.g. ``codex --agent <name> --cwd <dir>``), replace
the contents of ``build_payload`` below.
"""

from __future__ import annotations


def build_payload(
    work_dir: str,
    issue: int | str,
    title: str,
    *,
    next_cmd: str,
    next_prompt: str | None = None,
) -> dict:
    """Return a placeholder launcher payload for Codex."""
    del title  # unused
    return {
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
