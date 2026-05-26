"""QS-199 — Card source union helper for cross-cutting test patterns.

After the QS-199 refactor, methods like `_escapeHtml`, `_safeNumber`,
`_wireTargetHandle`, etc. moved from each `qs-*-card.js` into shared
modules under `ui/resources/shared/`. Many existing tests in
`tests/test_dashboard_rendering.py` grep the card source text for these
patterns; `card_source_union(card_filename)` returns the concatenated
text of the card file and every shared module it transitively imports,
so the existing grep-based tests keep working without per-test rewrite.

The mapping is a **static dict** (`CARD_TO_SHARED_FILES`) rather than
being parsed from `import` statements in the card source. This lets the
Phase C migration land per-card incrementally — a card that has not yet
been migrated to the shared base still maps to an empty list and the
helper returns only the card text (plan-critic R6 / dev-proxy redesign).

Per-card palette/constant pinning tests should continue to grep the
specific card file directly (`read_text(...)` on the card path); only
*cross-cutting* assertions that target patterns now living in a shared
module should use `card_source_union`.
"""

from __future__ import annotations

from pathlib import Path

_RESOURCES_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "custom_components"
    / "quiet_solar"
    / "ui"
    / "resources"
)

# Each card filename maps to the list of shared filenames it consumes.
# Update this dict whenever a card adds or removes a `from './shared/*'`
# import — the helper does NOT parse imports at runtime.
CARD_TO_SHARED_FILES: dict[str, list[str]] = {
    "qs-on-off-duration-card.js": [
        "qs-card-styles.js",
        "qs-card-base.js",
        "qs-ring-duration-base.js",
    ],
    "qs-radiator-card.js": [
        "qs-card-styles.js",
        "qs-card-base.js",
        "qs-ring-duration-base.js",
        "qs-anim-flame.js",
    ],
    "qs-pool-card.js": [
        "qs-card-styles.js",
        "qs-card-base.js",
        "qs-ring-duration-base.js",
        "qs-anim-wave.js",
    ],
    "qs-water-boiler-card.js": [
        "qs-card-styles.js",
        "qs-card-base.js",
        "qs-ring-duration-base.js",
        "qs-anim-wave.js",
    ],
    "qs-climate-card.js": [
        "qs-card-styles.js",
        "qs-card-base.js",
        "qs-ring-duration-base.js",
        "qs-anim-flame.js",
        "qs-anim-wave.js",
    ],
    "qs-car-card.js": [
        "qs-card-styles.js",
        "qs-card-base.js",
    ],
}


def card_source_union(card_filename: str) -> str:
    """Return the card source concatenated with every shared module it imports.

    The returned string is the literal concatenation of:

      1. The card file at ``ui/resources/<card_filename>``;
      2. Each shared file listed in ``CARD_TO_SHARED_FILES[card_filename]``
         (resolved against ``ui/resources/shared/``).

    Files are separated by ``\\n\\n``. The ``?qs_tag=<tag>`` cache-buster
    query strings (which only ever appear in copied destination files,
    not source files) are not stripped — they shouldn't exist in source.

    If ``card_filename`` is not in ``CARD_TO_SHARED_FILES``, only the
    card text is returned (lets the helper work on unmigrated cards).
    """
    card_path = _RESOURCES_ROOT / card_filename
    parts: list[str] = [card_path.read_text(encoding="utf-8")]

    for shared_name in CARD_TO_SHARED_FILES.get(card_filename, []):
        shared_path = _RESOURCES_ROOT / "shared" / shared_name
        if shared_path.exists():
            parts.append(shared_path.read_text(encoding="utf-8"))

    return "\n\n".join(parts)
