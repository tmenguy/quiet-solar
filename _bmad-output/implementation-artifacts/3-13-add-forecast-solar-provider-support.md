# Story 3.13: Add Forecast.Solar Provider Support

Status: ready-for-dev

## Story

As TheAdmin,
I want Forecast.Solar to be available as a solar forecast provider alongside Solcast and Open-Meteo,
So that I have a third provider option and can compare its accuracy against others using the multi-provider infrastructure from Story 3.7.

## Acceptance Criteria

1. **AC1 — Provider discovery and config flow**
   **Given** the Forecast.Solar HA integration is installed and configured
   **When** TheAdmin configures a solar device in Quiet Solar
   **Then** "Forecast.Solar" appears as an available provider in the config flow
   **And** it can be selected alongside other providers (Solcast, Open-Meteo)

2. **AC2 — Forecast data extraction**
   **Given** the Forecast.Solar provider is active
   **When** it delivers forecast data
   **Then** the forecast is extracted from the coordinator's `Estimate.watts` dict
   **And** the provider's native temporal resolution is detected from the data
   **And** the data is returned as `list[tuple[datetime, float]]` in UTC, watts — consistent with all other providers

3. **AC3 — Full integration with multi-provider features**
   **Given** Forecast.Solar is configured as a provider
   **When** the system operates
   **Then** all Story 3.7 features apply: staleness detection, health monitoring, re-probing, scoring, dampening, auto-selection

## Tasks / Subtasks

- [ ] Task 1: Add Forecast.Solar domain constant and provider class (AC: 1, 2)
  - [ ] 1.1 Add `FORECAST_SOLAR_DOMAIN = "forecast_solar"` in `const.py` (alongside existing `SOLCAST_SOLAR_DOMAIN` and `OPEN_METEO_SOLAR_DOMAIN`)
  - [ ] 1.2 Create `QSSolarProviderForecastSolar(QSSolarProvider)` in `ha_model/solar.py`. The data structure is identical to Open-Meteo: `Estimate.watts` is `dict[datetime, int]`. Implementation is very similar to `QSSolarProviderOpenWeather`
  - [ ] 1.3 Implement `fill_orchestrators()`: use `hass.config_entries.async_entries(FORECAST_SOLAR_DOMAIN)` to get entries, then `entry.runtime_data` to get the `ForecastSolarDataUpdateCoordinator`. The coordinator's `.data` is an `Estimate` object
  - [ ] 1.4 Implement `get_power_series_from_orchestrator()`: read `orchestrator.data.watts` (dict[datetime, int]), convert to sorted list of `(datetime, float)` tuples. Values are already in watts (int), cast to float for consistency
  - [ ] 1.5 Write tests: provider creation, orchestrator discovery, data extraction, empty data handling

- [ ] Task 2: Config flow integration (AC: 1)
  - [ ] 2.1 In `config_flow.py` `async_step_solar()`: add discovery check for `FORECAST_SOLAR_DOMAIN` — check `hass.config_entries.async_entries(FORECAST_SOLAR_DOMAIN)` (same pattern as Solcast uses `async_entries`, not `hass.data`)
  - [ ] 2.2 Add "Forecast.Solar" as a `SelectOptionDict` option when entries exist
  - [ ] 2.3 In `QSSolar.__init__`: add elif branch for `FORECAST_SOLAR_DOMAIN` → instantiate `QSSolarProviderForecastSolar`. (Story 3.7 will refactor this to multi-provider; if 3.7 is done first, just register the new provider class in the provider registry)
  - [ ] 2.4 Write tests: provider appears in options when integration installed, does not appear when not installed

- [ ] Task 3: Verify full feature integration (AC: 3)
  - [ ] 3.1 Verify Forecast.Solar participates in scoring, dampening, auto-selection (these are generic features from Story 3.7 that work for any `QSSolarProvider` subclass — no extra code needed, just test it)
  - [ ] 3.2 Write integration test: configure Forecast.Solar alongside Solcast, verify both score, dampening works on both, auto-selection picks the better one

## Dev Notes

### Forecast.Solar Data Model (from `forecast_solar` v5.0.0)

The HA integration domain is `forecast_solar`. The coordinator is `ForecastSolarDataUpdateCoordinator(DataUpdateCoordinator[Estimate])`.

**Access pattern:**
```
entry.runtime_data → ForecastSolarDataUpdateCoordinator
coordinator.data → Estimate
estimate.watts → dict[datetime, int]  # power in watts at each timestamp
estimate.wh_period → dict[datetime, int]  # energy per period
estimate.wh_days → dict[datetime, int]  # energy per day
```

**Key facts:**
- `Estimate.watts` is `dict[datetime, int]` — same structure as Open-Meteo's `orchestrator.data.watts`
- Values are already in watts (not kW like Solcast) — no conversion needed, just cast to float
- Temporal resolution depends on account type: hourly for free, higher for paid. Detected from timestamps at runtime
- The coordinator stores data in `entry.runtime_data` (same pattern as Solcast), NOT in `hass.data[domain]` (the Open-Meteo pattern)

### Implementation is nearly identical to QSSolarProviderOpenWeather

The data access pattern `orchestrator.data.watts` → `dict[datetime, int]` is the same for both Forecast.Solar and Open-Meteo. The main differences:
1. **Orchestrator discovery**: Forecast.Solar uses `hass.config_entries.async_entries()` + `entry.runtime_data` (like Solcast), not `hass.data[domain]` (like Open-Meteo)
2. **Values are int, not float**: Cast to float for consistency with the `list[tuple[datetime, float]]` contract

### Architecture Constraints

- **Two-layer boundary**: The provider class lives in `ha_model/solar.py` (HA bridge layer) — it imports from HA to access config entries
- **All config keys in `const.py`**: The domain constant must be there
- **Logging**: Use lazy `%s` format, no f-strings

### What NOT to Do

- **Do NOT duplicate QSSolarProviderOpenWeather** — the data access is nearly identical. Consider whether a shared base or helper can serve both. But don't over-abstract if the differences are small enough to keep separate
- **Do NOT hardcode the temporal resolution** — detect from data, as Story 3.7 requires
- **Do NOT add forecast_solar as a dependency** — it's an optional HA integration, discovered at runtime

### Testing Strategy

- Mock the Forecast.Solar config entries and coordinator with `Estimate.watts` dict
- Verify the provider returns the same `list[tuple[datetime, float]]` format as other providers
- Use FakeHass with mock config entries
- **100% coverage mandatory**

### Project Structure Notes

- Domain constant → `custom_components/quiet_solar/const.py`
- Provider class → `custom_components/quiet_solar/ha_model/solar.py`
- Config flow update → `custom_components/quiet_solar/config_flow.py`
- Tests → `tests/test_solar_forecast_solar_provider.py`

### References

- [Source: venv314/.../homeassistant/components/forecast_solar/coordinator.py — ForecastSolarDataUpdateCoordinator]
- [Source: venv314/.../homeassistant/components/forecast_solar/__init__.py — entry.runtime_data = coordinator]
- [Source: forecast_solar.models — Estimate dataclass, watts: dict[datetime, int]]
- [Source: custom_components/quiet_solar/ha_model/solar.py — QSSolarProviderOpenWeather (near-identical data pattern)]
- [Source: custom_components/quiet_solar/ha_model/solar.py — QSSolarProviderSolcast (orchestrator discovery via config_entries)]
- [Source: https://www.home-assistant.io/integrations/forecast_solar/ — Integration docs]

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
