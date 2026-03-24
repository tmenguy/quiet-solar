# Story 3.13: Add Forecast.Solar Provider Support

Status: ready-for-dev

## Story

As TheAdmin,
I want Forecast.Solar to be available as a solar forecast provider alongside Solcast and Open-Meteo,
So that I have a third provider option and can compare its accuracy against others using the multi-provider infrastructure from Story 3.7.

## Acceptance Criteria

1. **AC1 - Provider discovery and config flow**
   **Given** the Forecast.Solar HA integration is installed and configured
   **When** TheAdmin configures a solar device in Quiet Solar
   **Then** "Forecast.Solar" appears as an available provider in the config flow
   **And** it can be selected alongside other providers (Solcast, Open-Meteo)

2. **AC2 - Forecast data extraction**
   **Given** the Forecast.Solar provider is active
   **When** it delivers forecast data
   **Then** the forecast is extracted from the coordinator's `Estimate.watts` dict
   **And** the provider's native temporal resolution is detected from the data
   **And** the data is returned as `list[tuple[datetime, float]]` in UTC, watts - consistent with all other providers

3. **AC3 - Full integration with multi-provider features**
   **Given** Forecast.Solar is configured as a provider
   **When** the system operates
   **Then** all Story 3.7 features apply: staleness detection, health monitoring, re-probing, scoring, dampening, auto-selection

## Tasks / Subtasks

- [ ] Task 1: Add Forecast.Solar domain constant and provider class (AC: 1, 2)
  - [ ] 1.1 Add `FORECAST_SOLAR_DOMAIN = "forecast_solar"` in `const.py` next to `SOLCAST_SOLAR_DOMAIN` (line 183) and `OPEN_METEO_SOLAR_DOMAIN` (line 184)
  - [ ] 1.2 Create `QSSolarProviderForecastSolar(QSSolarProvider)` in `ha_model/solar.py` after `QSSolarProviderOpenWeather` (line 886)
  - [ ] 1.3 Override `fill_orchestrators()`: use `hass.config_entries.async_entries(FORECAST_SOLAR_DOMAIN)` then `entry.runtime_data` directly (the coordinator IS runtime_data, not `.coordinator` on it - unlike Solcast). Pattern:
    ```python
    entries = self.hass.config_entries.async_entries(self.domain)
    for entry in entries:
        try:
            orch = entry.runtime_data  # Direct! ForecastSolarDataUpdateCoordinator
            self.orchestrators.append(orch)
            self._orchestrator_health[id(orch)] = True
        except (AttributeError, TypeError):
            pass
    ```
  - [ ] 1.4 Override `get_power_series_from_orchestrator()`: read `orchestrator.data.watts` (`dict[datetime, int]`), convert to sorted `list[tuple[datetime, float]]`. Nearly identical to `QSSolarProviderOpenWeather.get_power_series_from_orchestrator()` (lines 861-885)
  - [ ] 1.5 Write tests: provider creation, orchestrator discovery, data extraction, empty/None data handling

- [ ] Task 2: Register in factory and config flow (AC: 1)
  - [ ] 2.1 In `_create_provider_for_domain()` (solar.py line 67): add `if domain == FORECAST_SOLAR_DOMAIN: return QSSolarProviderForecastSolar(...)`
  - [ ] 2.2 In `config_flow.py` `async_step_solar()` (line 871): add provider detection. IMPORTANT: Forecast.Solar does NOT populate `hass.data[domain]` - it stores the coordinator in `entry.runtime_data`. Detection MUST use `hass.config_entries.async_entries(FORECAST_SOLAR_DOMAIN)` (not `hass.data.get()`)
  - [ ] 2.3 Add `SelectOptionDict(value=FORECAST_SOLAR_DOMAIN, label="Forecast.Solar")` when entries found
  - [ ] 2.4 In `domain_labels` dict (line 847): add `FORECAST_SOLAR_DOMAIN: "Forecast.Solar"`
  - [ ] 2.5 Add import of `FORECAST_SOLAR_DOMAIN` in config_flow.py (line 134 area)
  - [ ] 2.6 In `_migrate_solar_providers_config()` (solar.py line 53): add `"Forecast.Solar"` label mapping for the new domain (for edge case where old config format is used)
  - [ ] 2.7 Write tests: provider appears in options when integration installed, does not appear when not installed

- [ ] Task 3: Verify full feature integration (AC: 3)
  - [ ] 3.1 Write integration test: configure Forecast.Solar alongside Solcast, verify both providers score independently, dampening applies per-provider, auto-selection picks the better one
  - [ ] 3.2 Verify staleness detection and health monitoring work (inherited from base class, just test it)

## Dev Notes

### Forecast.Solar Integration Data Model (verified from HA core source)

The HA integration domain is `forecast_solar`. Source: `homeassistant/components/forecast_solar/`.

**Setup chain** (from `__init__.py`):
```python
coordinator = ForecastSolarDataUpdateCoordinator(hass, entry)
await coordinator.async_config_entry_first_refresh()
entry.runtime_data = coordinator  # <-- Direct assignment, no wrapper
```

**Coordinator** (from `coordinator.py`):
```python
type ForecastSolarConfigEntry = ConfigEntry[ForecastSolarDataUpdateCoordinator]

class ForecastSolarDataUpdateCoordinator(DataUpdateCoordinator[Estimate]):
    async def _async_update_data(self) -> Estimate:
        return await self.forecast.estimate()
```

**Data access chain:**
```
entry.runtime_data -> ForecastSolarDataUpdateCoordinator
coordinator.data   -> Estimate (from forecast_solar library)
estimate.watts     -> dict[datetime, int]  # power in watts at each timestamp
estimate.wh_period -> dict[datetime, int]  # energy per period
estimate.wh_days   -> dict[datetime, int]  # energy per day
```

### Critical: Orchestrator Discovery Difference

| Integration | Where coordinator lives | Detection method |
|---|---|---|
| Solcast | `entry.runtime_data.coordinator` | `config_entries.async_entries()` |
| Open-Meteo | `hass.data[domain][entry_id]` | `hass.data.get()` |
| **Forecast.Solar** | **`entry.runtime_data` directly** | **`config_entries.async_entries()`** |

Forecast.Solar uses the same `config_entries.async_entries()` pattern as Solcast for discovery, but the coordinator is `entry.runtime_data` directly (not `.coordinator` on it).

### Implementation Pattern: Near-Identical to Open-Meteo

`get_power_series_from_orchestrator()` is nearly identical to `QSSolarProviderOpenWeather` (solar.py:861-885):
- Same data structure: `orchestrator.data.watts` -> `dict[datetime, int]`
- Same conversion: dict items -> sorted list -> bisect_left -> slice -> `[(dt.astimezone(UTC), float(watts))]`
- Only difference: orchestrator discovery method (config_entries vs hass.data)

### What NOT to Do

- **Do NOT use `hass.data.get(FORECAST_SOLAR_DOMAIN)`** for config flow detection - Forecast.Solar does not populate `hass.data[domain]`. Use `hass.config_entries.async_entries()` instead
- **Do NOT access `entry.runtime_data.coordinator`** - the coordinator IS `entry.runtime_data` directly
- **Do NOT duplicate the `get_power_series_from_orchestrator` body** if you can share it with Open-Meteo. But don't over-abstract either - two small identical methods is acceptable
- **Do NOT hardcode temporal resolution** - detect from data timestamps
- **Do NOT add `forecast_solar` as a dependency** - it's an optional HA integration, discovered at runtime

### Architecture Constraints

- **Two-layer boundary**: Provider class lives in `ha_model/solar.py` (HA bridge layer)
- **All config keys in `const.py`**: Domain constant must be there
- **Logging**: Use lazy `%s` format, no f-strings, no periods at end
- **Async rules**: No blocking calls in async code

### Testing Strategy

- Mock the Forecast.Solar config entries and coordinator with `Estimate`-like object having `.watts` dict
- Verify the provider returns `list[tuple[datetime, float]]` format
- Use FakeHass with mock config entries (same pattern as Solcast tests in `test_solar_forecast_resilience.py`)
- Test empty data, None data, single entry, multiple entries
- **100% coverage mandatory**

### Project Structure Notes

- Domain constant -> `custom_components/quiet_solar/const.py` (line ~184)
- Provider class -> `custom_components/quiet_solar/ha_model/solar.py` (after line 886)
- Factory registration -> `custom_components/quiet_solar/ha_model/solar.py` `_create_provider_for_domain()` (line 67)
- Config flow update -> `custom_components/quiet_solar/config_flow.py` `async_step_solar()` (line 838)
- Tests -> `tests/test_solar_forecast_solar_provider.py` (new file)

### References

- [Source: homeassistant/components/forecast_solar/__init__.py - entry.runtime_data = coordinator (direct assignment)]
- [Source: homeassistant/components/forecast_solar/coordinator.py - ForecastSolarDataUpdateCoordinator(DataUpdateCoordinator[Estimate])]
- [Source: homeassistant/components/forecast_solar/const.py - DOMAIN = "forecast_solar"]
- [Source: custom_components/quiet_solar/ha_model/solar.py:857-885 - QSSolarProviderOpenWeather (near-identical data pattern)]
- [Source: custom_components/quiet_solar/ha_model/solar.py:805-854 - QSSolarProviderSolcast (config_entries orchestrator discovery)]
- [Source: custom_components/quiet_solar/ha_model/solar.py:67-74 - _create_provider_for_domain() factory]
- [Source: custom_components/quiet_solar/config_flow.py:838-883 - async_step_solar() provider detection]

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
