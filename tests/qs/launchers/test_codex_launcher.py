"""Tests for ``scripts/qs/launchers/codex.py`` ``build_payload``."""

from __future__ import annotations


def test_lane_kwarg_is_accepted_and_ignored() -> None:
    """QS-358: ``lane=`` is reserved for the Claude pin; codex ignores it."""
    from launchers import codex as codex_launcher  # type: ignore[import-not-found]

    kw = {"next_cmd": "create-plan", "next_prompt": "go"}
    assert codex_launcher.build_payload("/tmp/work", 42, "T", **kw) == codex_launcher.build_payload(
        "/tmp/work", 42, "T", **kw, lane="feature-factory",
    )
