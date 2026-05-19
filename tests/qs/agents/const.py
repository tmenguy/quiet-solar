"""Shared constants for OpenCode static-agent contract tests.

Extracted per QS-184 review-fix #01 S10 so future agent tests can reuse
the same legacy-token list, frontmatter key set, and dev-only pattern
without drifting.
"""

from __future__ import annotations

import re

# Tokens that mark "active reference to legacy machinery". Anchored on
# three distinct tokens so a future rename touching only one of them
# still trips downstream tests.
LEGACY_REFERENCE_TOKENS: tuple[str, ...] = (
    "_qsprocess_opencode",
    "scripts/qs_opencode",
    "render_agent.py",
)

LEGACY_REFERENCE_RE: re.Pattern[str] = re.compile("|".join(LEGACY_REFERENCE_TOKENS))

# OpenCode frontmatter keys an agent file MUST declare.
REQUIRED_FRONTMATTER_KEYS: tuple[str, ...] = (
    "description",
    "mode",
    "color",
    "permission",
)

# The dev-only-patterns invariant: ``legacy/`` must be present in
# ``scripts/qs/quality_gate.py:_DEV_ONLY_PATTERNS``; the pre-migration
# tokens must not.
DEV_ONLY_PATTERN_LEGACY: str = "legacy/"
DEV_ONLY_PATTERN_FORBIDDEN: tuple[str, ...] = (
    "_qsprocess_opencode/",
    "scripts/qs_opencode/",
)
