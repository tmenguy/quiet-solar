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
    """The new ``charge_origin`` entity feeds the ``.forecast-row`` string."""
    assert "this._entity(e.charge_origin)" in card_source
    assert "chargeOriginStr" in card_source
    # the row is driven by the origin string and a leading Mode icon
    assert 'class="forecast-row">${chargeIcon' in card_source


def test_forecast_prefix_dropped(card_source: str) -> None:
    """The old ``Forecast:`` prefix is no longer rendered."""
    assert "Forecast: ${forecastDisplay}" not in card_source
