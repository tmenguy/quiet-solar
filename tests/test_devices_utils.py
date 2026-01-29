from datetime import datetime, timedelta
from unittest import TestCase

import pytz


from custom_components.quiet_solar.ha_model.device import (
    HADeviceMixin,
    compute_energy_Wh_rieman_sum,
    convert_current_to_amps,
    convert_distance_to_km,
    convert_power_to_w,
    get_average_power_energy_based,
    get_median_sensor,
)
from homeassistant.const import ATTR_UNIT_OF_MEASUREMENT, UnitOfElectricCurrent, UnitOfLength, UnitOfPower
from custom_components.quiet_solar.home_model.home_utils import get_average_time_series
from custom_components.quiet_solar.home_model.load import AbstractLoad, align_time_series_and_values


class QSCTest(HADeviceMixin, AbstractLoad):

    def __init__(self, **kwargs):
        self.fake_id_1 = "fake_id_1"
        self.fake_id_2 = "fake_id_2"



        self.fake_1 = [
            (datetime(year=2024, month=6, day=1, minute=0, second=0, tzinfo=pytz.UTC), 10),
            (datetime(year=2024, month=6, day=1, minute=0, second=10, tzinfo=pytz.UTC), 20),
            (datetime(year=2024, month=6, day=1, minute=0, second=20, tzinfo=pytz.UTC), 30),
            (datetime(year=2024, month=6, day=1, minute=0, second=30, tzinfo=pytz.UTC), None),
            (datetime(year=2024, month=6, day=1, minute=0, second=40, tzinfo=pytz.UTC), 40),
            (datetime(year=2024, month=6, day=1, minute=0, second=50, tzinfo=pytz.UTC), 50),
            (datetime(year=2024, month=6, day=1, minute=1, second=0, tzinfo=pytz.UTC), None),
            (datetime(year=2024, month=6, day=1, minute=1, second=10, tzinfo=pytz.UTC), 60),
            (datetime(year=2024, month=6, day=1, minute=1, second=20, tzinfo=pytz.UTC), 70),
            (datetime(year=2024, month=6, day=1, minute=1, second=30, tzinfo=pytz.UTC), None),
            (datetime(year=2024, month=6, day=1, minute=1, second=35, tzinfo=pytz.UTC), 80),
            (datetime(year=2024, month=6, day=1, minute=1, second=45, tzinfo=pytz.UTC), None)
        ]



        self.fake_2 = [
            (datetime(year=2024, month=6, day=1, minute=0, second=5, tzinfo=pytz.UTC), 1000),
            (datetime(year=2024, month=6, day=1, minute=0, second=25, tzinfo=pytz.UTC), 2000),
            (datetime(year=2024, month=6, day=1, minute=0, second=30, tzinfo=pytz.UTC), 3000),
            (datetime(year=2024, month=6, day=1, minute=0, second=35, tzinfo=pytz.UTC), None),
            (datetime(year=2024, month=6, day=1, minute=0, second=55, tzinfo=pytz.UTC), 4000),
            (datetime(year=2024, month=6, day=1, minute=1, second=0, tzinfo=pytz.UTC), None),
            (datetime(year=2024, month=6, day=1, minute=1, second=5, tzinfo=pytz.UTC), None),
            (datetime(year=2024, month=6, day=1, minute=1, second=15, tzinfo=pytz.UTC), 6000),
            (datetime(year=2024, month=6, day=1, minute=1, second=35, tzinfo=pytz.UTC), 7000),
            (datetime(year=2024, month=6, day=1, minute=1, second=40, tzinfo=pytz.UTC), None),
        ]

        super().__init__(hass=None, config_entry=None, name="toto", **kwargs)

        self.attach_ha_state_to_probe(self.fake_id_1,
                                      is_numerical=True,
                                      non_ha_entity_get_state=self.fake_id_1_state_getter)

        self.attach_ha_state_to_probe(self.fake_id_2,
                                      is_numerical=True,
                                      non_ha_entity_get_state=self.fake_id_2_state_getter)


    def get_value_from_time(self, vals, time: datetime):
        for t, v in vals:
            if t == time:
                return v
        return None

    def fake_id_1_state_getter(self, entity_id: str, time: datetime | None) -> (
            tuple[ datetime | None, float | str | None, dict | None] | None):
        res = None
        for t, v in self.fake_1:
            if t == time:
                res = (time, v, {})
        return res

    def fake_id_2_state_getter(self, entity_id: str, time: datetime | None) -> (
            tuple[ datetime | None, float | str | None, dict | None] | None):
        res = None
        for t, v in self.fake_2:
            if t == time:
                res = (time, v, {})

        return res

class TestDeviceUtils(TestCase):

    def test_utils(self):
        device = QSCTest()

        for t, _ in device.fake_1:
            device.add_to_history(device.fake_id_1, t)

        for t, _ in device.fake_2:
            device.add_to_history(device.fake_id_2, t)

        last = device.fake_1[-1][0]

        if last < device.fake_2[-1][0]:
            last = device.fake_2[-1][0]

        last += timedelta(seconds=1)

        val1_no_invalid = device.get_state_history_data(device.fake_id_1, num_seconds_before=200, to_ts=last, keep_invalid_states=False)
        fake_1_no_invalid = [(t, v) for t, v in device.fake_1 if v is not None]

        assert len(val1_no_invalid) == len(fake_1_no_invalid)
        for (t1, v1, attr1), (t2, v2) in zip(val1_no_invalid, fake_1_no_invalid):
            assert t1 == t2
            assert v1 == v2

        val2_no_invalid = device.get_state_history_data(device.fake_id_2, num_seconds_before=200, to_ts=last, keep_invalid_states=False)
        fake_2_no_invalid = [(t, v) for t, v in device.fake_2 if v is not None]
        for (t1, v1, attr1), (t2, v2) in zip(val2_no_invalid, fake_2_no_invalid):
            assert t1 == t2
            assert v1 == v2


        val1_all = device.get_state_history_data(device.fake_id_1, num_seconds_before=200, to_ts=last, keep_invalid_states=True)
        fake_1= device.fake_1
        assert len(val1_all) == len(fake_1)
        for (t1, v1, attr1), (t2, v2) in zip(val1_all, fake_1):
            assert t1 == t2
            assert v1 == v2

        val2_all = device.get_state_history_data(device.fake_id_2, num_seconds_before=200, to_ts=last, keep_invalid_states=True)
        fake_2= device.fake_2
        assert len(val2_all) == len(fake_2)
        for (t1, v1, attr1), (t2, v2) in zip(val2_all, fake_2):
            assert t1 == t2
            assert v1 == v2

        all_t_no_invalid = sorted(list(set([t for t, _ in fake_1_no_invalid] + [t for t, _ in fake_2_no_invalid])))
        all_t_all = sorted(list(set([t for t, _ in fake_1] + [t for t, _ in fake_2])))




        sum_no_invalid = align_time_series_and_values(val1_no_invalid, val2_no_invalid, operation=lambda x, y: x + y)
        sum_all = align_time_series_and_values(val1_all, val2_all, operation=lambda x, y: x + y)

        assert len(sum_no_invalid) == len(all_t_no_invalid)
        assert len(sum_all) == len(all_t_all)

        for (t1, v1, attr1), (t2) in zip(sum_no_invalid, all_t_no_invalid):
            assert t1 == t2

        for (t1, v1, attr1), (t2) in zip(sum_all, all_t_all):
            assert t1 == t2

        sum_no_invalid_res =    [
                   (datetime(2024, 6, 1, 0, 0,     tzinfo= pytz.UTC), 1010.0, {}), (
                    datetime(2024, 6, 1, 0, 0, 5,  tzinfo= pytz.UTC), 1015.0, {}), (
                    datetime(2024, 6, 1, 0, 0, 10, tzinfo= pytz.UTC), 1270.0, {}), (
                    datetime(2024, 6, 1, 0, 0, 20, tzinfo= pytz.UTC), 1780.0, {}), (
                    datetime(2024, 6, 1, 0, 0, 25, tzinfo= pytz.UTC), 2032.5, {}), (
                    datetime(2024, 6, 1, 0, 0, 30, tzinfo= pytz.UTC), 3035.0, {}), (
                    datetime(2024, 6, 1, 0, 0, 40, tzinfo= pytz.UTC), 3440.0, {}), (
                    datetime(2024, 6, 1, 0, 0, 50, tzinfo= pytz.UTC), 3850.0, {}), (
                    datetime(2024, 6, 1, 0, 0, 55, tzinfo= pytz.UTC), 4052.5, {}), (
                    datetime(2024, 6, 1, 0, 1, 10, tzinfo= pytz.UTC), 5560.0, {}), (
                    datetime(2024, 6, 1, 0, 1, 15, tzinfo= pytz.UTC), 6065.0, {}), (
                    datetime(2024, 6, 1, 0, 1, 20, tzinfo= pytz.UTC), 6320.0, {}), (
                    datetime(2024, 6, 1, 0, 1, 35, tzinfo= pytz.UTC), 7080.0, {})]

        for (t1, v1, attr1), (t2, v2, attr2) in zip(sum_no_invalid, sum_no_invalid_res):
            assert t1 == t2
            assert v1 == v2

        sum_all_res = [
                   (datetime(2024, 6, 1, 0, 0,0,   tzinfo=pytz.UTC), 1010.0, {}), (
                    datetime(2024, 6, 1, 0, 0, 5,  tzinfo=pytz.UTC), 1015.0, {}), (
                    datetime(2024, 6, 1, 0, 0, 10, tzinfo=pytz.UTC), 1270.0, {}), (
                    datetime(2024, 6, 1, 0, 0, 20, tzinfo=pytz.UTC), 1780.0, {}), (
                    datetime(2024, 6, 1, 0, 0, 25, tzinfo=pytz.UTC), 2030.0, {}), (
                    datetime(2024, 6, 1, 0, 0, 30, tzinfo=pytz.UTC), None, {}), (
                    datetime(2024, 6, 1, 0, 0, 35, tzinfo=pytz.UTC), None, {}), (
                    datetime(2024, 6, 1, 0, 0, 40, tzinfo=pytz.UTC), None, {}), (
                    datetime(2024, 6, 1, 0, 0, 50, tzinfo=pytz.UTC), None, {}), (
                    datetime(2024, 6, 1, 0, 0, 55, tzinfo=pytz.UTC), 4050.0, {}), (
                    datetime(2024, 6, 1, 0, 1,  0, tzinfo=pytz.UTC), None, {}), (
                    datetime(2024, 6, 1, 0, 1, 5,  tzinfo=pytz.UTC), None, {}), (
                    datetime(2024, 6, 1, 0, 1, 10, tzinfo=pytz.UTC), None, {}), (
                    datetime(2024, 6, 1, 0, 1, 15, tzinfo=pytz.UTC), 6065.0, {}), (
                    datetime(2024, 6, 1, 0, 1, 20, tzinfo=pytz.UTC), 6320.0, {}), (
                    datetime(2024, 6, 1, 0, 1, 30, tzinfo=pytz.UTC), None, {}), (
                    datetime(2024, 6, 1, 0, 1, 35, tzinfo=pytz.UTC), 7080.0, {}), (
                    datetime(2024, 6, 1, 0, 1, 40, tzinfo=pytz.UTC), None, {}), (
                    datetime(2024, 6, 1, 0, 1, 45, tzinfo=pytz.UTC), None, {})]


        for (t1, v1, attr1), (t2, v2, attr2) in zip(sum_all, sum_all_res):
            assert t1 == t2
            assert v1 == v2

        a_no  = get_average_time_series(sum_no_invalid)
        a_all = get_average_time_series(sum_all)
        m_no = get_median_sensor(sum_no_invalid)
        m_all = get_median_sensor(sum_all)

        assert int(a_no) ==  3574 # before was ... but changed with time based calculus np.mean([v for _, v, _ in sum_no_invalid_res])
        assert int(a_all) == 3473  # np.mean([v for _, v, _ in sum_all_res if v is not None])
        assert int(m_no) == 3440 # np.median([v for _, v, _ in sum_no_invalid_res])
        assert int(m_all) == 2030 # np.median([v for _, v, _ in sum_all_res if v is not None])


def test_compute_energy_rieman_sum_basic():
    """Test energy computation with rieman sum."""
    t0 = datetime(2024, 6, 1, 0, 0, tzinfo=pytz.UTC)
    t1 = datetime(2024, 6, 1, 1, 0, tzinfo=pytz.UTC)
    energy, duration = compute_energy_Wh_rieman_sum([(t0, 1000), (t1, 1000)])
    assert duration == 1.0
    assert energy == 1000.0


def test_compute_energy_rieman_sum_conservative_and_clip():
    """Test conservative energy and clip-to-zero behavior."""
    t0 = datetime(2024, 6, 1, 0, 0, tzinfo=pytz.UTC)
    t1 = datetime(2024, 6, 1, 0, 30, tzinfo=pytz.UTC)
    t2 = datetime(2024, 6, 1, 1, 0, tzinfo=pytz.UTC)
    energy, duration = compute_energy_Wh_rieman_sum(
        [(t0, -10), (t1, 100), (t2, 200)],
        conservative=True,
        clip_to_zero_under_power=0,
    )
    assert duration == 1.0
    assert energy == 50.0


def test_convert_units_helpers():
    """Test unit conversion helpers and attribute updates."""
    value, attrs = convert_power_to_w(1.0, {ATTR_UNIT_OF_MEASUREMENT: UnitOfPower.KILO_WATT})
    assert value == 1000.0
    assert attrs[ATTR_UNIT_OF_MEASUREMENT] == UnitOfPower.WATT

    value, attrs = convert_current_to_amps(1.0, {ATTR_UNIT_OF_MEASUREMENT: UnitOfElectricCurrent.AMPERE})
    assert value == 1.0
    assert attrs[ATTR_UNIT_OF_MEASUREMENT] == UnitOfElectricCurrent.AMPERE

    value, attrs = convert_distance_to_km(1.0, {ATTR_UNIT_OF_MEASUREMENT: UnitOfLength.METERS})
    assert value == 0.001
    assert attrs[ATTR_UNIT_OF_MEASUREMENT] == UnitOfLength.KILOMETERS

    value, attrs = convert_power_to_w(500.0)
    assert value == 500.0
    assert attrs is None


def test_average_power_energy_based():
    """Test average power calculation from energy."""
    t0 = datetime(2024, 6, 1, 0, 0, tzinfo=pytz.UTC)
    t1 = datetime(2024, 6, 1, 1, 0, tzinfo=pytz.UTC)

    assert get_average_power_energy_based([]) == 0
    assert get_average_power_energy_based([(t0, None)]) == 0.0
    assert get_average_power_energy_based([(t0, 100.0)]) == 100.0
    assert get_average_power_energy_based([(t0, 100.0), (t1, 300.0)]) == 200.0


def test_get_median_sensor_filters_and_last_timing():
    """Test median sensor with filtering and last timing."""
    t0 = datetime(2024, 6, 1, 0, 0, tzinfo=pytz.UTC)
    t1 = datetime(2024, 6, 1, 0, 0, 10, tzinfo=pytz.UTC)
    t2 = datetime(2024, 6, 1, 0, 0, 20, tzinfo=pytz.UTC)
    data = [(t0, 1.0), (t1, None), (t2, 3.0)]

    assert get_median_sensor([], None) == 0
    assert get_median_sensor([(t0, None)], None) == 0.0

    val = get_median_sensor(data, last_timing=t2 + timedelta(seconds=10), min_val=1.0, max_val=3.0)
    assert val == 2.0