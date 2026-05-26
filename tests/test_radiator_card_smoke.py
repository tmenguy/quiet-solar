"""QS-204 AC4 — JS smoke test for the radiator-card flame redesign.

The radiator card's flame backdrop used to share its sine-wave-tongue
shape with ``qs-pool-card.js``'s water waves, which made the radiator
look like greyed water (idle) or warm-tinted water (running) rather
than flames.

QS-204 replaces the sine-wave path generator with a piecewise-quadratic
peaked-teeth path generator, and removes the global ``translateX``
horizontal scroll in favour of per-tooth tip-flicker. This file
pins the structural markers of that redesign:

* the new path-generator identifier is present;
* ``LAYER_BASE_HEIGHTS`` has a back-layer value ≥ 100 px so the back
  flame reaches roughly half the clip radius;
* idle frames are motionless — either ``STILL_AMP`` is ≤ 0.5 OR the
  card declares a ``STATIC_PEAK_HEIGHT`` constant ≥ 30 px so the idle
  silhouette still shows visible peaks without animation;
* the original palette tables (``FLAME_FILLS`` / ``FLAME_GREY_FILLS``)
  are still referenced (no accidental palette deletion).

JS test harness intentionally omitted — the project has no Jest /
Playwright config. String-grep on the source file is the only
mechanical option, and the actual visual outcome is validated by the
user during PR review (per the project's "JS-card visual review"
workflow).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.utils.card_sources import card_source_union

CARD_PATH = (
    Path(__file__).resolve().parent.parent
    / "custom_components"
    / "quiet_solar"
    / "ui"
    / "resources"
    / "qs-radiator-card.js"
)


@pytest.fixture(scope="module")
def card_source() -> str:
    """QS-199 — radiator-card source unioned with its shared modules.

    After QS-199 the flame engine (`_generateFlameTeethPath`,
    `PHASE_REGEN_MIN_DT`, `sinceLastRegen`, `isIdle` gates, the
    length-equality `console.assert`) lives in
    ``shared/qs-anim-flame.js``. `card_source_union` returns the
    concatenated card text + every shared module the card imports, so
    the existing `string in card_source` assertions keep working without
    per-test rewrites.

    Per-card palette / per-card layer constants (`LAYER_BASE_HEIGHTS`,
    `FLAME_FILLS`, `FLAME_GREY_FILLS`) still live in the card file —
    those test assertions are satisfied by the card section of the
    union directly.
    """
    return card_source_union("qs-radiator-card.js")


def test_radiator_card_declares_flame_teeth_path_generator(card_source: str) -> None:
    """AC4 — the new peaked-teeth path generator identifier is declared."""
    assert "_generateFlameTeethPath" in card_source, (
        "QS-204 redesign marker missing: expected `_generateFlameTeethPath` "
        "to be declared somewhere in qs-radiator-card.js"
    )


def test_radiator_card_back_layer_reaches_clip_radius_half(card_source: str) -> None:
    """AC4 — the back-layer base height is ≥ 100 px (≈ half ``CLIP_R``).

    The back layer's height drives the perceived flame elevation. With
    CLIP_R = 120 px, a back-layer base of ≥ 100 px puts the flame tip
    near the top of the clip circle (≈ half the circle's diameter),
    which is what makes the silhouette read as flames rather than a
    surface ripple.
    """
    match = re.search(
        r"LAYER_BASE_HEIGHTS\s*=\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
        card_source,
    )
    assert match is not None, (
        "Could not find `LAYER_BASE_HEIGHTS = [back, mid, front]` literal "
        "in qs-radiator-card.js — the redesign must declare it"
    )
    back, mid, front = (int(g) for g in match.groups())
    assert back >= 100, (
        f"QS-204 AC4 fail: back-layer base height must be ≥ 100 px so the "
        f"flame reaches ≈ half the clip radius; got LAYER_BASE_HEIGHTS = "
        f"[{back}, {mid}, {front}]"
    )


def test_radiator_card_idle_silhouette_motionless(card_source: str) -> None:
    """AC4 — idle frames are motionless OR there's an explicit static peak.

    QS-204 freezes the idle silhouette: either ``STILL_AMP`` is small
    enough that the tip wobble is sub-pixel (≤ 0.5), OR the card
    declares a ``STATIC_PEAK_HEIGHT`` constant ≥ 30 px that gives the
    idle silhouette visible peaks without animation. Either resolution
    satisfies the AC.

    QS-199 — the constants moved to ``FLAME_CONSTANTS`` in
    ``shared/qs-anim-flame.js``. Accepts both the legacy ``= N`` form
    (top-level `const`) and the new ``: N`` form (inside the
    ``FLAME_CONSTANTS`` object literal).
    """
    still_match = re.search(r"STILL_AMP\s*[=:]\s*([0-9]*\.?[0-9]+)", card_source)
    assert still_match is not None, (
        "Could not find `STILL_AMP = N` or `STILL_AMP: N` constant in the "
        "card+shared union"
    )
    still_amp = float(still_match.group(1))

    static_peak_match = re.search(
        r"STATIC_PEAK_HEIGHT\s*[=:]\s*([0-9]*\.?[0-9]+)", card_source
    )
    has_static_peak_ge_30 = (
        static_peak_match is not None and float(static_peak_match.group(1)) >= 30
    )

    assert still_amp <= 0.5 or has_static_peak_ge_30, (
        f"QS-204 AC4 fail: idle silhouette must be motionless. Either "
        f"`STILL_AMP` (got {still_amp}) must be ≤ 0.5, or a "
        f"`STATIC_PEAK_HEIGHT` constant ≥ 30 px must exist (got "
        f"{static_peak_match.group(1) if static_peak_match else 'none'})."
    )


def test_radiator_card_palettes_still_referenced(card_source: str) -> None:
    """AC4 — the warm/grey colour tables are still referenced.

    Guards against the redesign accidentally dropping the palettes; a
    no-palette card would render as plain SVG paths with no fill.
    """
    assert "FLAME_FILLS" in card_source, (
        "QS-204 palette regression: `FLAME_FILLS` (running orange palette) "
        "was dropped from qs-radiator-card.js"
    )
    assert "FLAME_GREY_FILLS" in card_source, (
        "QS-204 palette regression: `FLAME_GREY_FILLS` (idle grey palette) "
        "was dropped from qs-radiator-card.js"
    )


def test_radiator_card_idle_peak_boost_gated_on_is_idle(card_source: str) -> None:
    """Review-fix #02 G5 — idle peak-boost gated on explicit ``isIdle``.

    Review-fix #01 F5 swapped the strict-equality ``tipAmp === 0``
    check for an epsilon comparison ``tipAmp < 0.05`` so the
    STATIC_PEAK_HEIGHT boost still fired after the asymptotic lerp
    toward STILL_AMP=0. But the epsilon also fired transiently at the
    start of a running ramp (when ``_currentFlameAmp`` was still
    lerping 0→DANCE_AMP), producing a momentary visible "pop" of
    taller flames before they settled.

    Review-fix #02 G5 replaces the epsilon with the explicit
    ``isIdle`` flag passed by the caller; the boost is now driven by
    state, not amplitude residue.
    """
    assert "tipAmp === 0" not in card_source, (
        "review-fix #01 F5 regression: `tipAmp === 0` strict-equality "
        "must not return — the per-tooth lerp residue prevents it from "
        "firing after running→idle transitions."
    )
    assert "tipAmp < 0.05" not in card_source, (
        "review-fix #02 G5 regression: `tipAmp < 0.05` epsilon was "
        "removed from `_generateFlameTeethPath` because it briefly "
        "fired the STATIC_PEAK_HEIGHT boost on idle→running lerps "
        "(visible flame-pop). Use the `isIdle` flag instead."
    )
    assert "isIdle ? STATIC_PEAK_HEIGHT" in card_source, (
        "review-fix #02 G5: expected `isIdle ? STATIC_PEAK_HEIGHT : 0` "
        "to gate the idle peak boost on the explicit isIdle flag "
        "(not on transient amplitude residue)."
    )


def test_radiator_card_regen_is_time_throttled(card_source: str) -> None:
    """Review-fix #03 H2 — running-mode regen is time-throttled.

    Review-fix #02 G4's phase-delta throttle (``PHASE_REGEN_THRESHOLD =
    0.08 rad``) was effectively a no-op: phase advances ~0.84 rad per
    frame at 60 FPS with ``LAYER_TIP_FLICKER_HZ ≈ 8``, so the gate
    cleared every frame and regen still ran 60 times/sec. Review-fix
    #03 H2 swaps the gate for a time-based one
    (``PHASE_REGEN_MIN_DT = 0.20 s`` → ~5 fps cap), and drops the
    stale ``maxPhaseDelta`` and ``PHASE_REGEN_THRESHOLD`` symbols.
    """
    assert "PHASE_REGEN_MIN_DT" in card_source, (
        "review-fix #03 H2: expected `PHASE_REGEN_MIN_DT` constant "
        "to time-throttle the fireOn regen path"
    )
    assert "sinceLastRegen" in card_source, (
        "review-fix #03 H2: expected `sinceLastRegen` time-delta "
        "computation to drive the running-mode throttle"
    )
    assert "PHASE_REGEN_THRESHOLD" not in card_source, (
        "review-fix #03 H2 regression: `PHASE_REGEN_THRESHOLD` was "
        "removed (it was a no-op at 60 FPS) — must not reappear"
    )
    assert "maxPhaseDelta" not in card_source, (
        "review-fix #03 H2 regression: `maxPhaseDelta` was removed — "
        "the time-based throttle replaces the per-tooth phase check"
    )


def test_radiator_card_idle_wobble_gated_on_is_idle(card_source: str) -> None:
    """Review-fix #03 H1 — frozen-mid-wobble idle silhouette guard.

    After a running→idle transition, ``_currentFlameAmp`` is frozen at
    ~DANCE_AMP (the RAF lerp loop is cancelled by ``_stopAnimation()``
    before it converges to STILL_AMP=0). G5 gated the +30 px peak
    boost on ``isIdle`` but the per-tooth wobble term
    ``tipAmp * Math.sin(phase)`` was NOT gated, so each idle re-
    render baked in a frozen-mid-wobble silhouette inconsistent with
    cold-boot idle. H1 gates the wobble on ``!isIdle`` (mirrors the
    G5 peak-boost gate).
    """
    assert "isIdle ? 0 : tipAmp * Math.sin" in card_source, (
        "review-fix #03 H1: expected the wobble term to be gated on "
        "`isIdle` (`isIdle ? 0 : tipAmp * Math.sin(phase)`) so the "
        "idle silhouette doesn't inherit a frozen mid-wobble"
    )


def test_radiator_card_layer_constants_length_assertion(card_source: str) -> None:
    """Review-fix #02 G7 + #03 H7 — module-load length-equality guard.

    The four per-layer arrays (``LAYER_BASE_HEIGHTS``,
    ``LAYER_TIP_AMP_MULTS``, ``LAYER_TEETH_COUNTS``,
    ``LAYER_TIP_FLICKER_HZ``) must remain the same length. A future
    PR that extends one without the others would silently produce
    ``NaN`` paths from out-of-bounds reads. A ``console.assert`` at
    module load catches the mismatch at boot.

    Review-fix #03 H7 — pin all three pair-equality checks so a
    future PR cannot weaken the assertion to only one pair.
    """
    assert "console.assert" in card_source, (
        "review-fix #02 G7: expected a `console.assert` length-equality "
        "guard on the LAYER_* constants at module load"
    )
    assert "LAYER_TEETH_COUNTS.length === LAYER_TIP_FLICKER_HZ.length" in card_source, (
        "review-fix #02 G7: expected the length-equality guard to "
        "check LAYER_TEETH_COUNTS.length === LAYER_TIP_FLICKER_HZ.length"
    )
    assert "LAYER_TEETH_COUNTS.length === LAYER_BASE_HEIGHTS.length" in card_source, (
        "review-fix #03 H7: expected the length-equality guard to "
        "also check LAYER_TEETH_COUNTS.length === LAYER_BASE_HEIGHTS.length"
    )
    assert "LAYER_TEETH_COUNTS.length === LAYER_TIP_AMP_MULTS.length" in card_source, (
        "review-fix #03 H7: expected the length-equality guard to "
        "also check LAYER_TEETH_COUNTS.length === LAYER_TIP_AMP_MULTS.length"
    )
