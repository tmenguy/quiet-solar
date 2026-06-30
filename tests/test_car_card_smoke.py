"""QS-274 — JS smoke test for the car-card origin Mode icon + context row.

The project has no Jest / Playwright harness, so the structural markers of
the card change are pinned via string-grep on the source (the actual visual
outcome is validated by the user during PR review, per the project's
"JS-card visual review" workflow). ``card_source_union`` returns the card
text concatenated with every shared module it imports.
"""

from __future__ import annotations

import pytest

from tests.utils.card_sources import card_source_union


@pytest.fixture(scope="module")
def card_source() -> str:
    return card_source_union("qs-car-card.js")


def test_origin_icon_entries_present(card_source: str) -> None:
    """The origin-aware Mode icons are mapped in ``carChargeTypeIcons``."""
    assert '"Manual": "mdi:hand-back-right"' in card_source
    assert '"Calendar": "mdi:calendar"' in card_source
    assert '"As Fast As Possible": "mdi:rabbit"' in card_source
    assert '"Manual As Fast As Possible": "mdi:rabbit"' in card_source
    assert '"Person Automated": "mdi:account-clock"' in card_source


def test_dropped_icons_absent(card_source: str) -> None:
    """``mdi:auto-fix`` and the removed ``Scheduled`` key are gone."""
    assert "mdi:auto-fix" not in card_source
    assert '"Scheduled"' not in card_source


def test_charge_origin_wired_into_context_row(card_source: str) -> None:
    """The ``charge_origin`` entity feeds the ``.forecast-row`` string."""
    assert "this._entity(e.charge_origin)" in card_source
    assert "chargeOriginStr" in card_source
    # The row is driven by the origin string (``forecastDisplay``). The
    # leading Mode icon was intentionally removed (main commit "removed
    # icon on car js"), so the row no longer carries a ``${chargeIcon`` prefix.
    assert 'class="forecast-row">${forecastDisplay}' in card_source
    assert 'class="forecast-row">${chargeIcon' not in card_source


def test_forecast_prefix_dropped(card_source: str) -> None:
    """The old ``Forecast:`` prefix is no longer rendered."""
    assert "Forecast: ${forecastDisplay}" not in card_source


def test_is_as_fast_state_predicate_defined(card_source: str) -> None:
    """QS-280: the as-fast predicate checks the charge-type strings directly.

    It is backed by an explicit ``AS_FAST_STATES`` set of both as-fast
    charge-type strings, so the icon mapping can change independently without
    silently breaking the rabbit button. The single-quoted literals here are
    distinct from the double-quoted ``carChargeTypeIcons`` keys, so this is
    not satisfied by the icon map alone.
    """
    assert "const isAsFastState" in card_source
    assert "const AS_FAST_STATES" in card_source
    assert "'As Fast As Possible'" in card_source
    assert "'Manual As Fast As Possible'" in card_source


def test_rabbit_lit_class_uses_predicate(card_source: str) -> None:
    """QS-280: the rabbit lit ``on`` class is gated by ``isAsFastState``."""
    assert (
        "class=\"rabbit-btn ${isAsFastState(sChargeType?.state) ? 'on' : ''}\""
        in card_source
    )


def test_is_already_forcing_uses_predicate(card_source: str) -> None:
    """QS-280: the Stop-vs-Start gate is driven by ``isAsFastState``."""
    assert (
        "const isAlreadyForcing = isAsFastState(sChargeType?.state);"
        in card_source
    )


def test_bare_as_fast_comparison_no_longer_gates(card_source: str) -> None:
    """QS-280: the bare display-literal comparison is gone from both sites.

    Both the rabbit lit-class site and the ``isAlreadyForcing`` gate
    previously compared ``sChargeType?.state === 'As Fast As Possible'``,
    which missed the ``"Manual As Fast As Possible"`` user-force variant.
    """
    assert "sChargeType?.state === 'As Fast As Possible'" not in card_source
