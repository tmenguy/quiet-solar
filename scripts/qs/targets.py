#!/usr/bin/env python3
"""Lane domain module (QS-332): axes, declaration validity, path targets.

Single machine-readable source of truth for the 6-lane model of epic
QS-321 — 6 lanes = {bug, feature, epic} x {product, factory}. Three
consumers in-repo: the quality gate (lane check), ``fetch_issue.py``
and ``setup_task.py`` (declaration validation). The declaration-validity
rule lives HERE and only here (one truth table).

Deliberately NOT ``quality_gate._is_dev_only``'s complement and not a
replacement for it: ``_is_dev_only`` drives scope *detection* (which
suite to run), this module drives lane *enforcement*. The factory
basename list below is a copy, not a reference, so this module stays
import-independent of the gate.

Ad-hoc classification query (AC-10)::

    python scripts/qs/targets.py <path> [<path> ...]
"""

from __future__ import annotations

import re
import sys

KINDS = ("bug", "feature")
TARGETS = ("product", "factory")
SCALES = ("task", "epic")

# --- classify(): fully enumerated classification (story D5) ---------------

# Factory carve-outs by the PURPOSE rule: these pin factory code even
# though their path prefix says otherwise (review CR-1).
_FACTORY_CARVE_OUTS = frozenset({"tests/test_quality_gate.py"})

# ``tests/qs/`` must be checked before the product ``tests/`` prefix.
_FACTORY_PREFIXES = (
    "scripts/",
    "tests/qs/",
    ".claude/",
    ".cursor/",
    ".opencode/",
    ".github/",
    "legacy/",
    "docs/workflow/",
)

# Copied from ``quality_gate._is_dev_only``'s basename list — see the
# module docstring for why this is a copy, not an import.
_FACTORY_BASENAMES = frozenset(
    {
        "CLAUDE.md",
        "AGENTS.md",
        ".cursorrules",
        ".gitignore",
        "pyproject.toml",
        "setup.cfg",
        "requirements.txt",
        "requirements_test.txt",
    }
)

# ``docs/agents/`` is product BY EPIC RULING — the drift checker ties it
# to product source.
_PRODUCT_PREFIXES = (
    "custom_components/",
    "docs/agents/",
    "docs/product/",
    "tests/",
)

# Enumerated neutrality (silent): every task commits its story;
# ``docs/epics/`` is target-neutral by epic rule.
_NEUTRAL_PREFIXES = ("docs/stories/", "docs/epics/")
_NEUTRAL_BASENAMES = frozenset({"README.md"})


def classify(path: str) -> str:
    """Classify a repo-relative path: product | factory | neutral | unknown.

    ``"neutral"`` is *enumerated* neutrality (silent); ``"unknown"`` is
    fallthrough (nothing matched), which the gate prints as an FYI line
    so the fail-open set stays visible (review N-5/PC-09).
    """
    if path in _FACTORY_CARVE_OUTS:
        return "factory"
    if path.startswith(_FACTORY_PREFIXES):
        return "factory"
    if path.startswith(_NEUTRAL_PREFIXES):
        return "neutral"
    if path.startswith(_PRODUCT_PREFIXES):
        return "product"
    if "/" not in path:
        if path in _FACTORY_BASENAMES:
            return "factory"
        if path in _NEUTRAL_BASENAMES:
            return "neutral"
    return "unknown"


# --- parse_axes() ----------------------------------------------------------


def _single_axis_value(labels: list[str], axis: str, values: tuple[str, ...]) -> str:
    """Return the axis value iff exactly one label of that axis is present."""
    found = [v for v in values if f"{axis}:{v}" in labels]
    return found[0] if len(found) == 1 else ""


def parse_axes(labels: list[str]) -> dict[str, str]:
    """Parse the three axes (+ derived ``lane``) out of raw label names.

    Best-effort and never-raising: an absent or ambiguous (multi-label)
    axis yields ``""``. ``lane`` is ``{kind}-{target}`` for tasks,
    ``epic-{target}`` for epics, ``""`` when undeclared/incomplete.
    """
    kind = _single_axis_value(labels, "kind", KINDS)
    target = _single_axis_value(labels, "target", TARGETS)
    scale = _single_axis_value(labels, "scale", SCALES)
    if scale == "epic" and target:
        lane = f"epic-{target}"
    elif scale == "task" and kind and target:
        lane = f"{kind}-{target}"
    else:
        lane = ""
    return {"kind": kind, "target": target, "scale": scale, "lane": lane}


# --- validate_declaration(): THE truth table (story D1) --------------------


def validate_declaration(labels: list[str]) -> tuple[bool, list[str], str]:
    """Validate an issue's lane declaration; return ``(ok, missing, message)``.

    Rules (exactly one place — three consumers):

    - exactly one ``target:*``;
    - ``scale:epic``  ⇒ no ``kind:*`` at all;
    - ``scale:task``  ⇒ exactly one ``kind:*`` (task is also the shape
      assumed when ``scale`` is missing/ambiguous).

    ``missing`` lists the invalid-or-missing axis names. ``message`` is
    shape-aware (task vs epic) and carries the exact backfill command
    with an ``<N>`` placeholder for the issue number — callers substitute
    it (``message.replace("<N>", str(issue))``).
    """
    kind_labels = [lb for lb in labels if lb.startswith("kind:")]
    axes = parse_axes(labels)
    is_epic_shape = "scale:epic" in labels and "scale:task" not in labels

    missing: list[str] = []
    problems: list[str] = []
    suggestions: list[str] = []

    if not axes["target"]:
        missing.append("target")
        problems.append("exactly one target:* label is required")
        suggestions.append("target:product|target:factory")

    if not axes["scale"]:
        missing.append("scale")
        problems.append("exactly one scale:* label is required")
        if not is_epic_shape:
            suggestions.append("scale:task")

    if is_epic_shape:
        if kind_labels:
            missing.append("kind")
            problems.append(
                "an epic carries no kind:* label — remove "
                + ", ".join(sorted(kind_labels))
            )
    elif not axes["kind"]:
        missing.append("kind")
        problems.append("exactly one kind:* label is required")
        suggestions.append("kind:bug|kind:feature")

    if not missing:
        return True, [], ""

    shape = "epic" if is_epic_shape else "task"
    message = f"incomplete lane declaration ({shape}): " + "; ".join(problems)
    if suggestions:
        message += (
            "\nbackfill with: gh issue edit <N> --add-label "
            f'"{",".join(suggestions)}"'
        )
    return False, missing, message


# --- parse_parent_epic() (story D4) ----------------------------------------

_PARENT_EPIC_SECTION = re.compile(
    r"^###\s+Parent epic\s*$", re.IGNORECASE | re.MULTILINE
)
_REFS_RE = re.compile(r"\bRefs #(\d+)")


def parse_parent_epic(body: str) -> int | None:
    """Resolve the declared parent epic from an issue body, or ``None``.

    The structured issue-form section (a ``### Parent epic`` heading whose
    following non-empty line is ``#?(\\d+)``) wins over free-text
    cross-references (review PC-05); fallback is the first ``Refs #N``.
    Explicit, never guessed.
    """
    section = _PARENT_EPIC_SECTION.search(body)
    if section:
        for line in body[section.end() :].splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            match = re.fullmatch(r"#?(\d+)", stripped)
            if match:
                return int(match.group(1))
            break  # first non-empty line is not a number — fall back
    refs = _REFS_RE.search(body)
    return int(refs.group(1)) if refs else None


# --- ad-hoc classification query (AC-10) -----------------------------------


def main(argv: list[str]) -> None:
    """Print ``<classification>\\t<path>`` for each path argument."""
    if not argv:
        sys.stderr.write("usage: targets.py <path> [<path> ...]\n")
        raise SystemExit(2)
    for path in argv:
        print(f"{classify(path)}\t{path}")


if __name__ == "__main__":
    main(sys.argv[1:])
