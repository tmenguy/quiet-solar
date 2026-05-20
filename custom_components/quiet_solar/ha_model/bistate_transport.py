"""Per-backing transport strategy for bistate-duration devices.

A bistate-duration device (`QSBiStateDuration` and its subclasses) is a
load with two states (on/off) that runs for a target duration. Two
concrete subclasses differ only in **which** HA entity they observe and
**which** HA service they call: `QSOnOffDuration` flips a `switch.*`
entity via `switch.turn_on` / `switch.turn_off`, while
`QSClimateDuration` toggles a `climate.*` entity via
`climate.set_hvac_mode`.

This module extracts that per-backing difference into a small strategy
object so future variants (e.g. the heating-only `QSRadiator` that can
sit on either a switch OR a climate) can pick a transport at
construction time without duplicating the shared bistate logic.

The transport owns:
  - `entity` — the HA entity_id observed and controlled
  - `default_state_on()` / `default_state_off()` — the initial state
    strings the host should adopt
  - `mode_options(hass)` — the list of valid states for selectors
  - `power_from_state(...)` — map a raw HA state to load power
  - `execute(...)` — issue the underlying HA service call

The host (`QSBiStateDuration`) keeps owning the bistate-mode signals
(`_state_on`, `_state_off`, `_bistate_mode_on`, `_bistate_mode_off`),
the override logic, and `expected_state_from_command`. The transport
receives primitives, not the host — that keeps the contract narrow and
makes the transports trivially testable without spinning up a full
load.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from homeassistant.components import climate
from homeassistant.components.climate import HVACMode
from homeassistant.const import (
    ATTR_ENTITY_ID,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    Platform,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from ..home_model.commands import CMD_IDLE, CMD_ON, LoadCommand

_LOGGER = logging.getLogger(__name__)


def get_hvac_modes(hass: HomeAssistant, entity_id: str) -> list[str]:
    """Return the HVAC modes advertised by a climate entity.

    Falls back to `[AUTO, OFF]` when the entity has no `hvac_modes`
    capability (mirrors the legacy helper in `climate_controller.py`,
    which is re-exported here for backwards compatibility).
    """
    registry = er.async_get(hass)
    entry = registry.async_get(entity_id)
    return entry.capabilities.get("hvac_modes", [HVACMode.AUTO.value, HVACMode.OFF.value])


class BistateTransport(ABC):
    """Strategy for the per-backing concerns of a bistate-duration load.

    Concrete subclasses (`SwitchTransport`, `ClimateTransport`) hold the
    underlying HA entity_id and translate `LoadCommand` plus optional
    override-state into the right service call. The host
    (`QSBiStateDuration` and its subclasses) is responsible for the
    bistate-mode signalling and override state machine — the transport
    receives primitives, not the host.
    """

    entity: str

    @abstractmethod
    def default_state_on(self) -> str:
        """Initial value the host should use for its `_state_on`."""

    @abstractmethod
    def default_state_off(self) -> str:
        """Initial value the host should use for its `_state_off`."""

    @abstractmethod
    def mode_options(self, hass: HomeAssistant) -> list[str]:
        """Valid raw-state values the device exposes (for selector UIs)."""

    def power_from_state(self, state: str | None, power_use: float) -> float | None:
        """Map a raw HA state to load power.

        Returns `None` when the state is missing or unavailable so the
        caller can decide whether to fall back to defaults. Returns
        `power_use` when the state matches the on-state, `0.0` for
        anything else (including the explicit off-state).
        """
        if state is None or state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            return None
        if state == self.default_state_on():
            return power_use
        return 0.0

    @abstractmethod
    async def execute(
        self,
        hass: HomeAssistant,
        command: LoadCommand,
        override_state: str | None,
        state_on: str,
        state_off: str,
    ) -> bool | None:
        """Issue the service call that flips the underlying device.

        - ``override_state`` — when set, becomes the target state directly
          (the user override path).
        - ``state_on`` / ``state_off`` — host-owned values used to map a
          ``LoadCommand`` to a concrete service call.

        Returns ``False`` after a successful service call (mirroring the
        existing ``execute_command_system`` contract: the command has been
        dispatched but the device's confirmation is still pending).

        Raises:
            ValueError: when ``command`` is neither on, off, nor idle and
                no override state is provided.
        """


class SwitchTransport(BistateTransport):
    """Transport for switch-backed bistate loads (`switch.turn_on/off`)."""

    def __init__(self, switch_entity: str) -> None:
        """Hold the underlying `switch.*` entity_id."""
        self.entity = switch_entity

    def default_state_on(self) -> str:
        return "on"

    def default_state_off(self) -> str:
        return "off"

    def mode_options(self, hass: HomeAssistant) -> list[str]:
        # `hass` unused but kept for parity with `ClimateTransport`.
        del hass
        return ["on", "off"]

    async def execute(
        self,
        hass: HomeAssistant,
        command: LoadCommand,
        override_state: str | None,
        state_on: str,
        state_off: str,
    ) -> bool | None:
        # The host's `state_on` / `state_off` map override states 1:1 to
        # the switch's "on" / "off" service. Unused locals are silenced
        # so the parameter list stays symmetric with `ClimateTransport`.
        del state_off

        if override_state is not None:
            if override_state == state_on:
                action = SERVICE_TURN_ON
            else:
                action = SERVICE_TURN_OFF
        else:
            if command.is_like(CMD_ON):
                action = SERVICE_TURN_ON
            elif command.is_off_or_idle():
                action = SERVICE_TURN_OFF
            else:
                raise ValueError("Invalid command")

        _LOGGER.info("Executing on/off command %s on %s", action, self.entity)

        await hass.services.async_call(
            domain=Platform.SWITCH,
            service=action,
            target={"entity_id": self.entity},
        )
        return False


class ClimateTransport(BistateTransport):
    """Transport for climate-backed bistate loads (`climate.set_hvac_mode`)."""

    def __init__(self, climate_entity: str, state_on: str, state_off: str) -> None:
        """Hold the climate entity plus the on/off HVAC modes.

        ``state_on`` and ``state_off`` are stored as mutable attributes so
        the host can update them through the climate-specific
        `climate_state_on` / `climate_state_off` setters at runtime.
        """
        self.entity = climate_entity
        self.state_on = state_on
        self.state_off = state_off

    def default_state_on(self) -> str:
        return self.state_on

    def default_state_off(self) -> str:
        return self.state_off

    def mode_options(self, hass: HomeAssistant) -> list[str]:
        return get_hvac_modes(hass, self.entity)

    # Unused param: CMD_IDLE — covered indirectly through `is_off_or_idle()`
    async def execute(
        self,
        hass: HomeAssistant,
        command: LoadCommand,
        override_state: str | None,
        state_on: str,
        state_off: str,
    ) -> bool | None:
        # Cheap silencing — keep CMD_IDLE imported so it stays linked to
        # the contract this branch checks.
        _ = CMD_IDLE

        if override_state is not None:
            hvac_mode = override_state
        else:
            if command.is_like(CMD_ON):
                hvac_mode = state_on
            elif command.is_off_or_idle():
                hvac_mode = state_off
            else:
                raise ValueError("Invalid command")

        data: dict[str, Any] = {
            ATTR_ENTITY_ID: self.entity,
            climate.ATTR_HVAC_MODE: hvac_mode,
        }

        _LOGGER.info("Executing climate set_hvac_mode %s on %s", hvac_mode, self.entity)

        await hass.services.async_call(climate.DOMAIN, climate.SERVICE_SET_HVAC_MODE, data)
        return False
