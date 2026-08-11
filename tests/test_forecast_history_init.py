"""QS-346: forecast circular-buffer init must always terminate.

`QSSolarHistoryVals.init` walks backward through the persisted circular buffer
looking for the most recent "good" slot. When the buffer is read back with a
nonzero data row but an all-zero/stale day row (which happens when xdist workers
race on the shared on-disk forecast files), the scan used to loop forever. The
fix bounds the scan to one lap around the ring and falls back to a full refresh.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import pytz

from custom_components.quiet_solar.ha_model.home import (
    BUFFER_SIZE_DAYS,
    BUFFER_SIZE_IN_INTERVALS,
    QSHomeSolarAndConsumptionHistoryAndForecast,
    QSSolarHistoryVals,
)


@pytest.mark.timeout(10)
async def test_init_terminates_when_all_buffer_slots_stale(tmp_path):
    """All slots stale (day row all zero) must not infinite-loop; do a full refresh."""
    forecast = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path=str(tmp_path))
    vals = QSSolarHistoryVals(forecast, "sensor.qs_probe")

    # data row nonzero (so it is NOT the all-zero -> reset shortcut), day row all
    # zero -> every slot is "bad" -> the old `while True` scan never breaks.
    degenerate = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)
    degenerate[0][:] = 5

    now = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
    captured: dict[str, datetime] = {}

    async def fake_load(hass_arg, entity_id, start_time, end_time, no_attributes=True):
        captured["start"] = start_time
        captured["end"] = end_time
        return []

    with (
        patch.object(vals, "read_values_async", new=AsyncMock(return_value=degenerate)),
        patch("custom_components.quiet_solar.ha_model.home.load_from_history", side_effect=fake_load),
    ):
        history_start, history_end = await vals.init(now, for_reset=False)

    # No states were returned, so nothing was ingested.
    assert history_start is None
    assert history_end is None
    # Every slot was stale -> the whole buffer must be refreshed.
    span_days = (captured["end"] - captured["start"]).days
    assert span_days >= BUFFER_SIZE_DAYS - 1
