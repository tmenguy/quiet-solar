"""Session-cached rendering of the agent files for the contract tests (QS-357).

The harness agent files under ``.claude/agents/`` and ``.opencode/agents/``
are gitignored, rendered outputs (QS-357). The contract tests therefore
must not read the working-tree copies — on a fresh clone with no render
they do not exist. This helper renders the tracked templates into a
session-scoped temp dir and exposes the per-harness agents dir, so every
agent test reads rendered output and the full gate is green on a fresh
clone (AC11).

Content pins run on the **unbound** render (story §7): the bound render
adds a header comment and inlines a lane file with its own H1, which a
body-shape pin would trip on.
"""

from __future__ import annotations

import atexit
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
_TEMPLATES = REPO_ROOT / "scripts" / "qs" / "agent_templates"
_LANES = REPO_ROOT / "docs" / "workflow" / "lanes"

# ``render_agents`` is a top-level module under ``scripts/qs/`` — put it on
# ``sys.path`` here (idempotent with the autouse fixture) so this module can
# import it at import time, before the fixture fires.
sys.path.insert(0, str(REPO_ROOT / "scripts" / "qs"))

import render_agents as _r  # type: ignore[import-not-found]  # noqa: E402

_cache: dict[str, Path] = {}


def rendered_root(mode: str = "unbound") -> Path:
    """Render all templates once per ``mode`` into a cached temp dir."""
    if mode not in _cache:
        tmp = Path(tempfile.mkdtemp(prefix=f"qs_rendered_{mode}_"))
        atexit.register(shutil.rmtree, tmp, ignore_errors=True)
        if mode == "bound":
            ctx = _r.build_render_context(
                tmp, bound=True, issue=42, title="A title",
                labels=["kind:feature", "target:factory", "scale:task"],
                fetch=False, lanes_dir=_LANES,
            )
        else:
            ctx = _r.build_render_context(tmp, bound=False, fetch=False, lanes_dir=_LANES)
        _r.render_all(tmp, context=ctx, out_root=tmp, templates_dir=_TEMPLATES)
        _cache[mode] = tmp
    return _cache[mode]


def agents_dir(harness: str = "claude", mode: str = "unbound") -> Path:
    """Return the rendered ``<harness>/agents`` dir for ``mode``."""
    sub = ".claude" if harness in ("claude", "claude-code") else ".opencode"
    return rendered_root(mode) / sub / "agents"
