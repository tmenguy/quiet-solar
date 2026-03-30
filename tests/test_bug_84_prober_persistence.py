"""Tests for Bug #84: Fix solar forecast scores, prober persistence, and entity lookup.

Covers:
- QSforecastValueSensor serialize/restore round-trip (AC: 1, 4, 5)
- Entity lookup uses solar_plant.ha_entities, not home.ha_entities (AC: 6)
- QSSolarHistoryVals receives string entity_id (AC: 7)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytz

from custom_components.quiet_solar.ha_model.home import QSforecastValueSensor


# ============================================================================
# Task 3.1: Prober serialization round-trip
# ============================================================================


def test_serialize_stored_values_empty():
    """Serializing an empty prober returns an empty list."""
    prober = QSforecastValueSensor("test", 3600, lambda t: (t, 100.0))
    result = prober.serialize_stored_values()
    assert result == []


def test_serialize_stored_values_with_data():
    """Serializing a prober with stored values returns [[iso_str, float], ...]."""
    prober = QSforecastValueSensor("test", 3600, lambda t: (t, 100.0))
    t1 = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    t2 = datetime(2026, 3, 30, 13, 0, 0, tzinfo=pytz.UTC)
    prober._stored_values = [(t1, 500.0), (t2, 750.0)]

    result = prober.serialize_stored_values()

    assert len(result) == 2
    assert result[0][0] == t1.isoformat()
    assert result[0][1] == 500.0
    assert result[1][0] == t2.isoformat()
    assert result[1][1] == 750.0


def test_serialize_restore_round_trip():
    """Serialize then restore produces identical stored values."""
    prober = QSforecastValueSensor("test", 3600, lambda t: (t, 100.0))
    t1 = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    t2 = datetime(2026, 3, 30, 13, 0, 0, tzinfo=pytz.UTC)
    t3 = datetime(2026, 3, 30, 14, 0, 0, tzinfo=pytz.UTC)
    prober._stored_values = [(t1, 500.0), (t2, 750.0), (t3, 0.0)]

    serialized = prober.serialize_stored_values()

    new_prober = QSforecastValueSensor("test", 3600, lambda t: (t, 100.0))
    new_prober.restore_stored_values(serialized)

    assert len(new_prober._stored_values) == 3
    assert new_prober._stored_values[0] == (t1, 500.0)
    assert new_prober._stored_values[1] == (t2, 750.0)
    assert new_prober._stored_values[2] == (t3, 0.0)


# ============================================================================
# Task 3.2: Prober restore from serialized data
# ============================================================================


def test_restore_stored_values_from_empty():
    """Restoring from empty list results in empty stored values."""
    prober = QSforecastValueSensor("test", 3600, lambda t: (t, 100.0))
    prober.restore_stored_values([])
    assert prober._stored_values == []


def test_restore_stored_values_from_none():
    """Restoring from None is a no-op (no crash)."""
    prober = QSforecastValueSensor("test", 3600, lambda t: (t, 100.0))
    prober._stored_values = [(datetime(2026, 1, 1, tzinfo=pytz.UTC), 1.0)]
    prober.restore_stored_values(None)
    # Should not crash, stored values unchanged
    assert len(prober._stored_values) == 1


def test_restore_stored_values_preserves_timezone():
    """Restored datetime values preserve UTC timezone."""
    prober = QSforecastValueSensor("test", 3600, lambda t: (t, 100.0))
    t1 = datetime(2026, 3, 30, 15, 56, 0, tzinfo=pytz.UTC)
    serialized = [[t1.isoformat(), 42.0]]

    prober.restore_stored_values(serialized)

    assert prober._stored_values[0][0].tzinfo is not None
    assert prober._stored_values[0][0] == t1
    assert prober._stored_values[0][1] == 42.0


def test_restore_stored_values_skips_malformed_entries():
    """Malformed entries in serialized data are skipped without crashing."""
    prober = QSforecastValueSensor("test", 3600, lambda t: (t, 100.0))
    t1 = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    serialized = [
        [t1.isoformat(), 500.0],
        ["not-a-date", 100.0],  # malformed
        [t1.isoformat()],  # too short
    ]

    prober.restore_stored_values(serialized)

    # Only the valid entry should be restored
    assert len(prober._stored_values) == 1
    assert prober._stored_values[0] == (t1, 500.0)


# ============================================================================
# Task 3.3: Entity lookup uses solar_plant.ha_entities
# ============================================================================


async def test_solar_forecast_set_uses_solar_plant_ha_entities():
    """solar_forecast_set_and_reset looks up entities on solar_plant, not home."""
    from custom_components.quiet_solar.const import QSForecastSolarSensors
    from custom_components.quiet_solar.ha_model.home import (
        QSHomeSolarAndConsumptionHistoryAndForecast,
    )

    # Create a fake entity object with entity_id property
    fake_entity = MagicMock()
    fake_entity.entity_id = "sensor.quiet_solar_qs_solar_forecast_15mn"

    # solar_plant.ha_entities has the entity, home.ha_entities does NOT
    solar_plant = MagicMock()
    solar_plant.ha_entities = {
        list(QSForecastSolarSensors.keys())[0]: fake_entity,
    }
    solar_plant.solar_forecast_providers = {}
    solar_plant.solar_inverter_input_active_power = None

    home = MagicMock()
    home.ha_entities = {}  # empty — old bug would search here
    home.solar_plant = solar_plant

    forecast = QSHomeSolarAndConsumptionHistoryAndForecast.__new__(
        QSHomeSolarAndConsumptionHistoryAndForecast
    )
    forecast.home = home
    forecast._in_reset = False
    forecast.solar_production_history = MagicMock()
    forecast.solar_forecast_history = None
    forecast.solar_forecast_history_per_provider = MagicMock()
    forecast.storage_path = "/tmp"

    with patch.object(
        QSHomeSolarAndConsumptionHistoryAndForecast,
        "__init__",
        lambda self, **kwargs: None,
    ):
        # Mock QSSolarHistoryVals to capture entity_id passed
        captured_entity_ids = []

        class FakeHistoryVals:
            def __init__(self, entity_id, forecast):
                captured_entity_ids.append(entity_id)

            async def init(self, time, for_reset=False):
                pass

        with patch(
            "custom_components.quiet_solar.ha_model.home.QSSolarHistoryVals",
            FakeHistoryVals,
        ):
            time = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
            await forecast.solar_forecast_set_and_reset(time)

    # The lookup should have found the entity via solar_plant.ha_entities
    assert len(captured_entity_ids) > 0
    # entity_id should be a STRING, not the entity object itself
    assert all(isinstance(eid, str) for eid in captured_entity_ids)
    assert captured_entity_ids[0] == "sensor.quiet_solar_qs_solar_forecast_15mn"


# ============================================================================
# Task 3.4: entity_id passed as string to QSSolarHistoryVals
# ============================================================================


async def test_solar_forecast_per_provider_passes_string_entity_id():
    """Per-provider entity lookup also passes string entity_id."""
    from custom_components.quiet_solar.const import QSForecastSolarSensors
    from custom_components.quiet_solar.ha_model.home import (
        QSHomeSolarAndConsumptionHistoryAndForecast,
    )

    first_sensor_name = list(QSForecastSolarSensors.keys())[0]

    # Create a fake entity for the per-provider lookup
    fake_entity = MagicMock()
    fake_entity.entity_id = "sensor.quiet_solar_solcast_qs_solar_forecast_15mn"

    solar_plant = MagicMock()
    solar_plant.ha_entities = {
        first_sensor_name: MagicMock(entity_id="sensor.aggregate"),
        f"solcast_{first_sensor_name}": fake_entity,
    }
    solar_plant.solar_forecast_providers = {"solcast": MagicMock()}
    solar_plant.solar_inverter_input_active_power = None

    home = MagicMock()
    home.ha_entities = {}
    home.solar_plant = solar_plant

    forecast = QSHomeSolarAndConsumptionHistoryAndForecast.__new__(
        QSHomeSolarAndConsumptionHistoryAndForecast
    )
    forecast.home = home
    forecast._in_reset = False
    forecast.solar_production_history = MagicMock()
    forecast.solar_forecast_history = MagicMock()
    forecast.solar_forecast_history_per_provider = None
    forecast.storage_path = "/tmp"

    captured_entity_ids = []

    class FakeHistoryVals:
        def __init__(self, entity_id, forecast):
            captured_entity_ids.append(entity_id)

        async def init(self, time, for_reset=False):
            pass

    with patch(
        "custom_components.quiet_solar.ha_model.home.QSSolarHistoryVals",
        FakeHistoryVals,
    ):
        time = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
        await forecast.solar_forecast_set_and_reset(time)

    # Per-provider entity_ids must also be strings
    per_provider_ids = [eid for eid in captured_entity_ids if "solcast" in eid]
    assert len(per_provider_ids) > 0
    assert all(isinstance(eid, str) for eid in per_provider_ids)


async def test_solar_forecast_set_guards_solar_plant_none():
    """When solar_plant is None, solar_forecast_set_and_reset does not crash."""
    from custom_components.quiet_solar.ha_model.home import (
        QSHomeSolarAndConsumptionHistoryAndForecast,
    )

    home = MagicMock()
    home.solar_plant = None

    forecast = QSHomeSolarAndConsumptionHistoryAndForecast.__new__(
        QSHomeSolarAndConsumptionHistoryAndForecast
    )
    forecast.home = home
    forecast._in_reset = False
    forecast.solar_production_history = MagicMock()
    forecast.solar_forecast_history = None
    forecast.solar_forecast_history_per_provider = None
    forecast.storage_path = "/tmp"

    time = datetime(2026, 3, 30, 12, 0, 0, tzinfo=pytz.UTC)
    # Should not raise
    await forecast.solar_forecast_set_and_reset(time)
