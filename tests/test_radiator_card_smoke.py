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
    """Read the radiator-card source once for the module."""
    return CARD_PATH.read_text(encoding="utf-8")


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
    """
    still_match = re.search(r"STILL_AMP\s*=\s*([0-9]*\.?[0-9]+)", card_source)
    assert still_match is not None, (
        "Could not find `STILL_AMP = ...` constant in qs-radiator-card.js"
    )
    still_amp = float(still_match.group(1))

    static_peak_match = re.search(
        r"STATIC_PEAK_HEIGHT\s*=\s*([0-9]*\.?[0-9]+)", card_source
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


def test_radiator_card_phase_delta_throttle(card_source: str) -> None:
    """Review-fix #02 G4 — running-mode regen path is phase-delta throttled.

    Pre-fix, ``shouldRegen = hasValidBase && (fireOn || …)`` short-
    circuited true every RAF tick when fire was on, forcing three
    ``setAttribute('d', …)`` writes per frame at ~60Hz. Review-fix
    #02 G4 introduces a ``PHASE_REGEN_THRESHOLD`` constant and a
    ``maxPhaseDelta`` accumulator so regen only fires when the largest
    per-tooth phase has drifted more than ~4.6° since the last regen.
    """
    assert "PHASE_REGEN_THRESHOLD" in card_source, (
        "review-fix #02 G4: expected `PHASE_REGEN_THRESHOLD` constant "
        "to throttle the fireOn regen path"
    )
    assert "maxPhaseDelta" in card_source, (
        "review-fix #02 G4: expected `maxPhaseDelta` computation to "
        "drive the per-tooth throttle decision"
    )


def test_radiator_card_layer_constants_length_assertion(card_source: str) -> None:
    """Review-fix #02 G7 — module-load length-equality guard.

    The four per-layer arrays (``LAYER_BASE_HEIGHTS``,
    ``LAYER_TIP_AMP_MULTS``, ``LAYER_TEETH_COUNTS``,
    ``LAYER_TIP_FLICKER_HZ``) must remain the same length. A future
    PR that extends one without the others would silently produce
    ``NaN`` paths from out-of-bounds reads. A ``console.assert`` at
    module load catches the mismatch at boot.
    """
    assert "console.assert" in card_source, (
        "review-fix #02 G7: expected a `console.assert` length-equality "
        "guard on the LAYER_* constants at module load"
    )
    assert "LAYER_TEETH_COUNTS.length === LAYER_TIP_FLICKER_HZ.length" in card_source, (
        "review-fix #02 G7: expected the length-equality guard to "
        "check LAYER_TEETH_COUNTS.length === LAYER_TIP_FLICKER_HZ.length"
    )
