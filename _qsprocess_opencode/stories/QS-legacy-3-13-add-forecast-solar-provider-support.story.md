# Story 3.13: Add Forecast.Solar Provider Support

Status: review

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

- [x] Task 1: Add Forecast.Solar domain constant and provider class (AC: 1, 2)
  - [x] 1.1 Add `FORECAST_SOLAR_DOMAIN = "forecast_solar"` in `const.py`
  - [x] 1.2 Create `QSSolarProviderForecastSolar(QSSolarProvider)` in `ha_model/solar.py`
  - [x] 1.3 Override `fill_orchestrators()` using `config_entries.async_entries()` + `entry.runtime_data` directly
  - [x] 1.4 Override `get_power_series_from_orchestrator()` reading `orchestrator.data.watts`
  - [x] 1.5 Write tests: 19 tests covering provider creation, orchestrator discovery, data extraction, edge cases

- [x] Task 2: Register in factory and config flow (AC: 1)
  - [x] 2.1 Added `FORECAST_SOLAR_DOMAIN` branch in `_create_provider_for_domain()` factory
  - [x] 2.2 Added config flow detection using `hass.config_entries.async_entries(FORECAST_SOLAR_DOMAIN)`
  - [x] 2.3 Added `SelectOptionDict` option for Forecast.Solar
  - [x] 2.4 Added `FORECAST_SOLAR_DOMAIN: "Forecast.Solar"` to `domain_labels` dict
  - [x] 2.5 Added import of `FORECAST_SOLAR_DOMAIN` in config_flow.py
  - [x] 2.6 Updated `_migrate_solar_providers_config()` with Forecast.Solar label mapping
  - [x] 2.7 Write tests: 7 tests covering factory, migration, and regression

- [x] Task 3: Verify full feature integration (AC: 3)
  - [x] 3.1 Integration tests: multi-provider coexistence, score attributes, dampening attributes
  - [x] 3.2 Staleness detection and health monitoring verified via inherited base class

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

Claude Opus 4.6

### Debug Log References

### Completion Notes List

- Added `FORECAST_SOLAR_DOMAIN = "forecast_solar"` constant in const.py
- Created `QSSolarProviderForecastSolar` class with `fill_orchestrators()` (config_entries + entry.runtime_data) and `get_power_series_from_orchestrator()` (Estimate.watts dict extraction)
- Registered in `_create_provider_for_domain()` factory and `_migrate_solar_providers_config()`
- Updated config_flow.py: import, domain_labels, provider detection via `async_entries()`
- 30 new tests in `test_solar_forecast_solar_provider.py` covering provider creation, orchestrator discovery, data extraction, factory registration, migration, and multi-provider integration
- All quality gates pass: 4096 tests (0 failures), ruff lint/format clean, mypy clean
- Coverage: 99% overall (no regression — uncovered lines are pre-existing)

### File List

- `custom_components/quiet_solar/const.py` (modified — added FORECAST_SOLAR_DOMAIN)
- `custom_components/quiet_solar/ha_model/solar.py` (modified — added QSSolarProviderForecastSolar, updated factory and migration)
- `custom_components/quiet_solar/config_flow.py` (modified — added Forecast.Solar detection and options)
- `tests/test_solar_forecast_solar_provider.py` (new — 30 tests)
