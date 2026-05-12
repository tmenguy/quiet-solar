"""Launcher payload for OpenCode.

OpenCode has its own canonical handoff implementation under
``scripts/qs_opencode/`` (HTTP API session spawn + ``/instance/reload``).
This launcher emits a minimal payload pointing the user at that pipeline
— **the legacy OpenCode workflow remains the source of truth for
OpenCode sessions** and is not touched by the harness-agnostic
``scripts/qs/`` pipeline.

If you run the harness-agnostic pipeline under ``QS_HARNESS=opencode``,
expect to see this hand-off message; switch to the legacy OpenCode flow
(`.opencode/agents/qs-setup-task` etc.) for the real launcher.
"""

from __future__ import annotations

import shlex


def build_payload(
    work_dir: str,
    issue: int | str,
    title: str,
    *,
    next_cmd: str,
    next_prompt: str | None = None,
) -> dict:
    """Return a delegation payload pointing at the legacy OpenCode pipeline."""
    del title  # unused
    safe_dir = shlex.quote(work_dir)
    cmd = (
        f"opencode {safe_dir}"
        if not next_prompt
        else f"opencode {safe_dir} --prompt {shlex.quote(next_prompt)}"
    )

    return {
        "tool": "opencode",
        "same_context": next_cmd,
        "new_context": cmd,
        "note": (
            "OpenCode's canonical handoff is the legacy pipeline under "
            "scripts/qs_opencode/ (per-task rendered agents). This payload is "
            "a minimal fallback; for full feature parity launch OpenCode "
            "through the legacy /setup-task agent instead."
        ),
        "issue": issue,
        "work_dir": work_dir,
    }
