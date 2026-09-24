#!/usr/bin/env python3
"""Per-phase model policy by lane (QS-358).

The single source of truth for which model — and how much thinking
effort — each pipeline agent runs on. The renderer
(``render_agents.py``) resolves every agent's frontmatter from here;
the Claude launcher's GUI pin writes the phase's ``effortLevel`` from
here. This module imports nothing from the renderer (one-way
dependency).

Vocabulary (D18): the policy speaks five harness-agnostic **classes**;
each harness owns one complete ``class → model`` row in
``HARNESS_MODELS``, in its own vocabulary. Both rows name exact
versions — no alias floats anywhere in the policy (D14/D20).

The rendered frontmatter is the visible resolved table (D11)::

    grep -h '^model:' .claude/agents/*.md .opencode/agents/*.md
"""

from __future__ import annotations

import targets  # type: ignore[import-not-found]

# --- classes (D18): the policy's own, harness-agnostic vocabulary ----------
# build    — runs the implementer's model: contained, small-footprint code changes (QS-367 E1)
# deep     — the best code-grounded analyst: critique, concrete planning, hunting, root cause
# frontier — the best general reasoner: planning conversation, consolidation, judgment
# light    — checklists: delta-auditor, setup-task
# fast     — mechanical: finish, CodeRabbit wrapper, release
# ("inherit" is a renderer/template concern and never appears here.)
CLASSES: frozenset[str] = frozenset({"build", "deep", "frontier", "light", "fast"})

# Every lane, derived from targets.py (tuples → stable order).
LANES: tuple[str, ...] = tuple(f"{k}-{t}" for k in targets.KINDS for t in targets.TARGETS) + tuple(
    f"epic-{t}" for t in targets.TARGETS
)

# --- one complete class → model row per harness (D4/D17/D18/D20) ------------
# Keys mirror render_agents._HARNESSES (test-enforced). Claude Code takes
# full model IDs in frontmatter (honoured by the CLI and by sub-agents —
# aliases would float to the provider default because the settings ``env``
# pin is not applied). ``claude-opus-5-5`` requires Claude Code ≥ 2.1.280
# (QS-367 E8). OpenCode takes provider/model literals from ``opencode models``.
# Bumping ``deep``? Also bump ``opencode.json`` (the lockstep test names it;
# ``build`` does not affect ``opencode.json``).
HARNESS_MODELS: dict[str, dict[str, str]] = {
    "claude": {  # Claude Code ≥ 2.1.280 (required by ``claude-opus-5-5``), first-party API
        "build": "claude-opus-4-8",  # QS-367 E1: footprint — D3 reason still standing
        "deep": "claude-opus-5-5",  # QS-367 E2: analysis/review
        "frontier": "claude-fable-5-1",
        "light": "claude-sonnet-5",
        "fast": "claude-haiku-4-5",
    },
    "opencode": {  # OpenCode v2.0.14, github-copilot
        "build": "github-copilot/claude-opus-4.8",  # QS-367 E1
        "deep": "github-copilot/claude-opus-5.5",  # QS-367 E2
        "frontier": "github-copilot/gpt-6-astra",  # no Claude Fable on this provider (D17)
        "light": "github-copilot/claude-sonnet-5",
        "fast": "github-copilot/claude-haiku-4.5",
    },
}

# --- Claude Code CLI floor (QS-367 S4) -------------------------------------
# ``HARNESS_MODELS["claude"]["deep"] == "claude-opus-5-5"`` needs Claude Code
# ≥ 2.1.280 (QS-367 E8); older builds 400 on it. Named here, next to the
# ``claude`` row it constrains, and consumed by
# ``launchers.claude.check_cli_floor`` so the version and the model that
# requires it cannot drift apart.
CLAUDE_CLI_FLOOR: tuple[int, int, int] = (2, 1, 280)

# --- thinking effort per class (D19) ---------------------------------------
# Claude Code frontmatter ``effort`` / settings ``effortLevel`` vocabulary
# (low | medium | high | xhigh | max). None = do not set (Haiku 4.5 does
# not take the parameter). Claude-only: OpenCode documents no per-agent effort.
CLASS_EFFORT: dict[str, str | None] = {
    "build": "high",
    "deep": "high",
    "frontier": "high",
    "light": "medium",
    "fast": None,
}

# --- the table -------------------------------------------------------------
# The planning orchestrators are the only lane-dependent rows (D1).
_PLANNING: frozenset[str] = frozenset({"qs-create-plan", "qs-diagnose-task"})

# Lane-invariant rows (D2: reviewers deliberately mixed across classes).
_FLAT: dict[str, str] = {
    "qs-plan-critic": "deep",
    "qs-plan-concrete-planner": "deep",
    "qs-plan-dev-proxy": "build",
    "qs-plan-scope-guardian": "frontier",
    "qs-plan-delta-auditor": "light",
    "qs-diag-root-cause-skeptic": "deep",
    "qs-diag-fix-minimalist": "frontier",
    "qs-implement-task": "build",
    "qs-implement-setup-task": "build",
    "qs-review-task": "frontier",
    "qs-verify-task": "frontier",
    "qs-review-blind-hunter": "deep",
    "qs-review-edge-case-hunter": "deep",
    "qs-review-acceptance-auditor": "frontier",
    "qs-review-regression-proof": "frontier",
    "qs-review-coderabbit": "fast",
    "qs-finish-task": "fast",
    "qs-setup-task": "light",
    "qs-release": "fast",
}

STEMS: frozenset[str] = frozenset(_FLAT) | _PLANNING


class ModelPolicyError(ValueError):
    """Raised by :func:`resolve` for a stem with no policy row.

    ``stem`` carries the offending agent so callers can report it
    structurally (``next_step``'s JSON error payload).
    """

    def __init__(self, stem: str) -> None:
        super().__init__(f"no model policy row for agent {stem!r}")
        self.stem = stem


def effort_for(cls: str) -> str | None:
    """``CLASS_EFFORT[cls]`` — ``KeyError`` on an unknown class (programmer error)."""
    return CLASS_EFFORT[cls]


def _planning_class(lane: str | None) -> str:
    """Class of a planning orchestrator under ``lane``.

    Exact membership in :data:`LANES`: a ``bug-*`` lane → ``deep`` (causal
    chain); any other lane → ``frontier`` (conversation); anything else
    (``None``, ``""``, an unknown label) → ``deep``. A lane label is data
    and never raises (same stance as ``targets.parse_axes``).
    """
    if lane is None or lane not in LANES:
        return "deep"
    return "deep" if lane.startswith("bug-") else "frontier"


def resolve(lane: str | None, stem: str) -> str:
    """Class for ``stem`` under ``lane`` — always a member of :data:`CLASSES`.

    Raises:
        ModelPolicyError: for an unknown ``stem`` (a stem is code, not data).
    """
    if stem in _PLANNING:
        return _planning_class(lane)
    try:
        return _FLAT[stem]
    except KeyError:
        raise ModelPolicyError(stem) from None


def model_for(harness: str, cls: str) -> str:
    """``HARNESS_MODELS[harness][cls]`` — ``KeyError`` on an unknown harness or class."""
    return HARNESS_MODELS[harness][cls]


def class_of(value: str) -> str | None:
    """``value`` if it is a class, else ``None``.

    A literal, ``"inherit"`` or a bare alias is not a class; the renderer
    emits those verbatim.
    """
    return value if value in CLASSES else None
