"""QS-204 AC2 — radiator bistate-mode literals are namespaced.

Bug #2 (Force ON not toggling the switch) had two causes:

1. The constraints.py green-energy short-circuit bailed out on a fresh
   load with no command history (fixed pre-branch — see T6).
2. The radiator's ``_bistate_mode_on`` / ``_bistate_mode_off`` were
   the bare literals ``"on"`` / ``"off"``, sharing the namespace with
   raw HA switch states. The `radiator_mode` select translation also
   carried a stale ``"heat"`` entry left over from earlier scaffolding.

This test pins the new contract:

* ``radiator._bistate_mode_on == "radiator_mode_on"`` (NOT ``"on"``).
* ``radiator._bistate_mode_off == "radiator_mode_off"``.
* ``get_bistate_modes()`` returns exactly the three shared
  ``bistate_mode_*`` plus the two namespaced ``radiator_mode_*``.
* No raw HA state strings appear in the bistate modes set.

The test uses direct kwarg instantiation with the existing fixtures
``hass`` / ``radiator_config_entry`` / ``radiator_home`` patterned
after ``tests/test_ha_radiator.py`` (no factory addition needed per
the plan's scope-guardian note).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import (
    CONF_CLIMATE,
    CONF_CLIMATE_HVAC_MODE_OFF,
    CONF_CLIMATE_HVAC_MODE_ON,
    CONF_POWER,
    CONF_SWITCH,
    DATA_HANDLER,
    DOMAIN,
)
from custom_components.quiet_solar.ha_model.radiator import QSRadiator
from tests.factories import create_minimal_home_model


@pytest.fixture
def radiator_config_entry() -> MockConfigEntry:
    """Mock config entry used as the host for radiator construction."""
    return MockConfigEntry(
        domain=DOMAIN,
        entry_id="qs204_namespaced_bistate_entry",
        data={CONF_NAME: "QS-204 Namespaced Bistate"},
        title="QS-204 Namespaced Bistate",
    )


@pytest.fixture
def radiator_home(hass: HomeAssistant):
    """Minimal home wired into ``hass.data[DOMAIN]`` for radiator setup."""
    home = create_minimal_home_model()
    home.is_off_grid = MagicMock(return_value=False)
    data_handler = MagicMock()
    data_handler.home = home
    hass.data.setdefault(DOMAIN, {})[DATA_HANDLER] = data_handler
    return home


def _build_switch_radiator(hass, entry, home) -> QSRadiator:
    return QSRadiator(
        hass=hass,
        config_entry=entry,
        home=home,
        **{
            CONF_NAME: "Switch Radiator",
            CONF_SWITCH: "switch.qs204_namespaced",
            CONF_POWER: 1500,
        },
    )


def _build_climate_radiator(hass, entry, home) -> QSRadiator:
    return QSRadiator(
        hass=hass,
        config_entry=entry,
        home=home,
        **{
            CONF_NAME: "Climate Radiator",
            CONF_CLIMATE: "climate.qs204_namespaced",
            CONF_CLIMATE_HVAC_MODE_ON: "heat",
            CONF_CLIMATE_HVAC_MODE_OFF: "off",
            CONF_POWER: 1500,
        },
    )


def test_radiator_bistate_mode_on_is_namespaced_switch_backing(
    hass: HomeAssistant, radiator_config_entry, radiator_home
) -> None:
    """AC2 — switch-backed radiator uses ``radiator_mode_on`` not ``"on"``."""
    radiator = _build_switch_radiator(hass, radiator_config_entry, radiator_home)

    assert radiator._bistate_mode_on == "radiator_mode_on"
    assert radiator._bistate_mode_off == "radiator_mode_off"


def test_radiator_bistate_mode_on_is_namespaced_climate_backing(
    hass: HomeAssistant, radiator_config_entry, radiator_home
) -> None:
    """AC2 — climate-backed radiator also uses the namespaced literals.

    The literals are independent of which backing the radiator picks;
    locking parity here prevents a future split.
    """
    radiator = _build_climate_radiator(hass, radiator_config_entry, radiator_home)

    assert radiator._bistate_mode_on == "radiator_mode_on"
    assert radiator._bistate_mode_off == "radiator_mode_off"


def test_radiator_bistate_modes_pinned_set_namespaced(
    hass: HomeAssistant, radiator_config_entry, radiator_home
) -> None:
    """AC2 — ``get_bistate_modes()`` returns exactly the five namespaced modes.

    Set-equality catches both missing modes (regression) and unexpected
    modes (e.g. the stale ``"heat"`` key leaking back in).
    """
    radiator = _build_switch_radiator(hass, radiator_config_entry, radiator_home)

    modes = radiator.get_bistate_modes()

    assert set(modes) == {
        "bistate_mode_auto",
        "bistate_mode_exact_calendar",
        "bistate_mode_default",
        "radiator_mode_on",
        "radiator_mode_off",
    }
    # Raw HA state strings must no longer appear in the bistate modes
    # — they collide with the underlying entity states and break the
    # `radiator_mode` translation lookup.
    assert "on" not in modes
    assert "off" not in modes
    assert "heat" not in modes


def test_radiator_bistate_modes_order_matches_concatenation(
    hass: HomeAssistant, radiator_config_entry, radiator_home
) -> None:
    """AC2 — order is the existing ``bistate_modes + [on, off]`` concatenation."""
    radiator = _build_switch_radiator(hass, radiator_config_entry, radiator_home)

    modes = radiator.get_bistate_modes()

    assert modes == [
        "bistate_mode_auto",
        "bistate_mode_exact_calendar",
        "bistate_mode_default",
        "radiator_mode_on",
        "radiator_mode_off",
    ]
