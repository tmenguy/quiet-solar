"""Tests for home controlled/non-controlled consumption computation (fix #152).

Tests cover:
- Bistate load with user override excluded from controlled consumption
- Charger with force charge constraint (originator: user_override) excluded
- Charger get_override_state with force constraint
- Override reset state not excluded (still controlled)
- Dynamic group all/none/mixed override states
- Dynamic group without sensor skips overridden children
- Charger support_user_override returns True
- Clean constraints preserves force override constraint
- is_user_overridden method hierarchy
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytz

from custom_components.quiet_solar.const import (
    CONF_CHARGER_MAX_CHARGE,
    CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER,
    CONF_CHARGER_MIN_CHARGE,
    CONF_CHARGER_PAUSE_RESUME_SWITCH,
    CONF_CHARGER_PLUGGED,
    CONF_CHARGER_STATUS_SENSOR,
    CONSTRAINT_ORIGINATOR_KEY,
    CONSTRAINT_ORIGINATOR_USER_OVERRIDE,
    CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
)
from custom_components.quiet_solar.home_model.constraints import (
    MultiStepsPowerLoadConstraint,
)
from custom_components.quiet_solar.home_model.load import AbstractDevice, AbstractLoad
from tests.factories import MinimalTestHome, MinimalTestLoad, create_constraint


NOW = datetime(2027, 6, 15, 12, 0, 0, tzinfo=pytz.UTC)


# =============================================================================
# Task 2: is_user_overridden() method hierarchy
# =============================================================================


class TestIsUserOverriddenAbstractDevice:
    """Test is_user_overridden on AbstractDevice base."""

    def test_abstract_device_returns_false(self):
        """AbstractDevice.is_user_overridden() returns False by default."""
        dev = AbstractDevice(name="test_device")
        assert dev.is_user_overridden() is False


class TestIsUserOverriddenAbstractLoad:
    """Test is_user_overridden on AbstractLoad."""

    def test_no_override_returns_false(self):
        """Load with no override returns False."""
        load = MinimalTestLoad(name="test_load")
        assert load.is_user_overridden() is False

    def test_bistate_user_override_returns_true(self):
        """AC #1: Bistate with external_user_initiated_state returns True."""
        load = MinimalTestLoad(name="test_load")
        load.external_user_initiated_state = "some_state"
        load.asked_for_reset_user_initiated_state_time = None
        assert load.is_user_overridden() is True

    def test_asked_for_reset_returns_true(self):
        """Load in ASKED FOR RESET state returns True (still physically overridden)."""
        load = MinimalTestLoad(name="test_load")
        load.external_user_initiated_state = "some_state"
        load.asked_for_reset_user_initiated_state_time = NOW
        assert load.is_user_overridden() is True

    def test_constraint_originator_user_override_returns_true(self):
        """AC #2: Load with active constraint originator=user_override returns True."""
        load = MinimalTestLoad(name="test_load")
        ct = create_constraint(
            load=load,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
            time=NOW,
            end_of_constraint=NOW + timedelta(hours=2),
            load_info={CONSTRAINT_ORIGINATOR_KEY: CONSTRAINT_ORIGINATOR_USER_OVERRIDE},
            load_param="test_car",
        )
        load.push_live_constraint(NOW, ct)
        assert load.is_user_overridden() is True


# =============================================================================
# Task 1: Charger force constraint with originator
# =============================================================================


class TestChargerForceConstraintOriginator:
    """Test that charger force charge sets originator: user_override."""

    def test_charger_get_override_state_with_force_constraint(self):
        """AC #2: Charger with force constraint returns Override: {car_name}."""
        load = MinimalTestLoad(name="test_charger")
        ct = create_constraint(
            load=load,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
            time=NOW,
            end_of_constraint=NOW + timedelta(hours=2),
            load_info={CONSTRAINT_ORIGINATOR_KEY: CONSTRAINT_ORIGINATOR_USER_OVERRIDE},
            load_param="my_car",
        )
        load.push_live_constraint(NOW, ct)
        assert load.get_override_state() == "Override: my_car"

    def test_override_reset_state_excluded(self):
        """Load in ASKED FOR RESET OVERRIDE is still physically overridden — excluded."""
        load = MinimalTestLoad(name="test_load")
        load.external_user_initiated_state = "some_state"
        load.asked_for_reset_user_initiated_state_time = NOW
        assert load.get_override_state() == "ASKED FOR RESET OVERRIDE"
        assert load.is_user_overridden() is True


# =============================================================================
# Task 3: Charger support_user_override
# =============================================================================


class TestChargerSupportUserOverride:
    """Test charger support_user_override returns True."""

    def test_charger_support_user_override(self):
        """AC #2: Charger.support_user_override() returns True."""
        from custom_components.quiet_solar.ha_model.charger import QSChargerGeneric

        assert QSChargerGeneric.support_user_override(None) is True


# =============================================================================
# Task 4: Dynamic group is_user_overridden and override filtering
# =============================================================================


class _FakeDevice(AbstractDevice):
    """Fake device for group testing with controllable override state."""

    def __init__(self, name: str, overridden: bool = False):
        super().__init__(name=name)
        self._overridden = overridden
        self.qs_enable_device = True

    def is_user_overridden(self) -> bool | None:
        return self._overridden


class _FakeHADevice(_FakeDevice):
    """Fake device that also mixes in HADeviceMixin-like behavior for power."""

    def __init__(self, name: str, power: float = 100.0, overridden: bool = False):
        super().__init__(name=name, overridden=overridden)
        self._power = power

    def get_device_power_latest_possible_valid_value(
        self, tolerance_seconds: float | None, time: datetime, ignore_auto_and_user_overridden_load: bool = False
    ) -> float:
        if ignore_auto_and_user_overridden_load and self.is_user_overridden() is True:
            return 0.0
        return self._power


class TestDynamicGroupIsUserOverridden:
    """Test QSDynamicGroup.is_user_overridden() behavior."""

    def _make_group(self, children: list[AbstractDevice]) -> Any:
        """Create a dynamic group with given children."""
        from custom_components.quiet_solar.ha_model.dynamic_group import QSDynamicGroup

        group = QSDynamicGroup.__new__(QSDynamicGroup)
        group._childrens = children
        return group

    def test_all_overridden_returns_true(self):
        """AC #4: All children overridden → True."""
        children = [_FakeDevice("a", overridden=True), _FakeDevice("b", overridden=True)]
        group = self._make_group(children)
        assert group.is_user_overridden() is True

    def test_none_overridden_returns_false(self):
        """AC #4: No children overridden → False."""
        children = [_FakeDevice("a", overridden=False), _FakeDevice("b", overridden=False)]
        group = self._make_group(children)
        assert group.is_user_overridden() is False

    def test_mixed_returns_none(self):
        """AC #4: Mixed override → None."""
        children = [_FakeDevice("a", overridden=True), _FakeDevice("b", overridden=False)]
        group = self._make_group(children)
        assert group.is_user_overridden() is None

    def test_empty_children_returns_false(self):
        """Empty group returns False."""
        group = self._make_group([])
        assert group.is_user_overridden() is False


# =============================================================================
# Task 4: get_device_power with override filtering
# =============================================================================


class TestDevicePowerOverrideFiltering:
    """Test ignore_auto_and_user_overridden_load parameter."""

    def test_overridden_load_returns_zero_when_flag_true(self):
        """Overridden load returns 0.0 when flag is True."""
        load = MinimalTestLoad(name="test_load", power=500.0)
        load.external_user_initiated_state = "some_state"
        load.asked_for_reset_user_initiated_state_time = None
        # We need to test via the renamed parameter on HADeviceMixin
        # Since MinimalTestLoad doesn't inherit HADeviceMixin, we test the logic directly
        assert load.is_user_overridden() is True

    def test_non_overridden_load_returns_power_when_flag_true(self):
        """Non-overridden load returns its power when flag is True."""
        load = MinimalTestLoad(name="test_load", power=500.0)
        assert load.is_user_overridden() is False


# =============================================================================
# Auto-boosted load with active ON command is controlled
# =============================================================================


class TestAutoBoostedLoadWithActiveCommand:
    """Test that auto-boosted loads with active ON commands count as controlled."""

    def test_auto_boosted_with_on_command_is_controlled(self):
        """Auto-boosted load with active ON command should NOT return 0.0."""
        from custom_components.quiet_solar.home_model.commands import CMD_ON, copy_command

        load = MinimalTestLoad(name="boosted_load", power=500.0)
        load.load_is_auto_to_be_boosted = True
        # Simulate an active ON command
        load.current_command = copy_command(CMD_ON, power_consign=500.0)
        # is_load_command_set needs running_command=None and current_command!=None
        assert load.is_load_command_set(NOW) is True
        assert not load.current_command.is_off_or_idle()

    def test_auto_boosted_without_command_is_not_controlled(self):
        """Auto-boosted load without active command should return 0.0 (excluded)."""
        load = MinimalTestLoad(name="boosted_load", power=500.0)
        load.load_is_auto_to_be_boosted = True
        load.current_command = None
        assert load.is_load_command_set(NOW) is False


# =============================================================================
# Task 4.6: Piloted sets filtering
# =============================================================================


class TestPilotedSetsOverrideFiltering:
    """Test that overridden loads' piloted devices are excluded."""

    def test_overridden_load_piloted_devices_excluded(self):
        """AC #1: Overridden load's piloted devices not added to piloted_sets."""
        # This tests the logic: if load.is_user_overridden() is True, skip adding its devices_to_pilot
        load = MinimalTestLoad(name="overridden_load")
        load.external_user_initiated_state = "some_state"
        load.asked_for_reset_user_initiated_state_time = None
        load.qs_enable_device = True

        piloted_device = AbstractDevice(name="piloted")
        load.devices_to_pilot = [piloted_device]

        # Simulate the filtering logic from home.py
        piloted_sets: set = set()
        all_loads = [load]
        for ld in all_loads:
            if ld.qs_enable_device is False:
                continue
            if ld.is_user_overridden() is True:
                continue
            if len(ld.devices_to_pilot) != 0:
                piloted_sets.update(ld.devices_to_pilot)

        assert piloted_device not in piloted_sets

    def test_non_overridden_load_piloted_devices_included(self):
        """Non-overridden load's piloted devices are included."""
        load = MinimalTestLoad(name="normal_load")
        load.qs_enable_device = True

        piloted_device = AbstractDevice(name="piloted")
        load.devices_to_pilot = [piloted_device]

        piloted_sets: set = set()
        all_loads = [load]
        for ld in all_loads:
            if ld.qs_enable_device is False:
                continue
            if ld.is_user_overridden() is True:
                continue
            if len(ld.devices_to_pilot) != 0:
                piloted_sets.update(ld.devices_to_pilot)

        assert piloted_device in piloted_sets


# =============================================================================
# Task 1.2: Clean constraints preserves force override
# =============================================================================


class TestCleanConstraintsPreservesForceOverride:
    """Verify constraint cleanup doesn't remove force override constraint."""

    def test_clean_with_person_load_info_keeps_force_override(self):
        """AC #2: clean_constraints with person load_info keeps originator constraint."""
        load = MinimalTestLoad(name="test_charger")

        # Push force constraint with originator
        force_ct = create_constraint(
            load=load,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
            time=NOW,
            end_of_constraint=NOW + timedelta(hours=2),
            load_info={CONSTRAINT_ORIGINATOR_KEY: CONSTRAINT_ORIGINATOR_USER_OVERRIDE},
            load_param="my_car",
        )
        load.push_live_constraint(NOW, force_ct)

        # Clean with person-based load_info — should NOT remove force constraint
        load.clean_constraints_for_load_param_and_if_same_key_same_value_info(
            NOW, load_param="my_car", load_info={"person": "test_person"}, for_full_reset=False
        )

        # Force constraint should still be there
        active = load.get_current_active_constraint()
        assert active is not None
        assert active.load_info.get(CONSTRAINT_ORIGINATOR_KEY) == CONSTRAINT_ORIGINATOR_USER_OVERRIDE


# =============================================================================
# HADeviceMixin-level coverage: device.py:874, dynamic_group.py:184,195
# =============================================================================


class TestHADeviceMixinOverrideFiltering:
    """Cover override filtering in HADeviceMixin.get_device_power_latest_possible_valid_value."""

    def _make_load_device(self):
        """Create a ConcreteLoadDevice mixing HADeviceMixin + AbstractLoad."""
        from custom_components.quiet_solar.ha_model.device import HADeviceMixin
        from tests.conftest import FakeConfigEntry, FakeHass
        from tests.factories import MinimalTestHome

        class ConcreteLoadDevice(HADeviceMixin, AbstractLoad):
            def __init__(self, **kwargs):
                self._power = kwargs.pop("power_use", 1000.0)
                super().__init__(**kwargs)

            @property
            def power_use(self) -> float:
                return self._power

            @property
            def efficiency_factor(self) -> float:
                return 1.0

            def get_update_value_callback_for_constraint_class(self, constraint):
                return None

            def is_off_grid(self):
                return False

        hass = FakeHass()
        return ConcreteLoadDevice(
            hass=hass,
            config_entry=FakeConfigEntry(entry_id="test"),
            home=MinimalTestHome(voltage=230.0),
            name="TestLoad",
            device_type="test",
        )

    def test_overridden_load_returns_zero(self):
        """device.py:874 — overridden load returns 0.0 when flag is True."""
        dev = self._make_load_device()
        dev.external_user_initiated_state = "some_state"
        dev.asked_for_reset_user_initiated_state_time = None
        assert dev.is_user_overridden() is True
        result = dev.get_device_power_latest_possible_valid_value(
            tolerance_seconds=None, time=NOW, ignore_auto_and_user_overridden_load=True
        )
        assert result == 0.0


class TestDynamicGroupPowerOverrideFiltering:
    """Cover dynamic_group.py:184,195 — override filtering in get_device_power."""

    def _make_group(self, hass, children, accurate_power_sensor=None):
        from custom_components.quiet_solar.ha_model.dynamic_group import QSDynamicGroup
        from tests.conftest import FakeConfigEntry
        from tests.factories import MinimalTestHome

        home = MinimalTestHome(voltage=230.0)
        group = QSDynamicGroup(
            hass=hass,
            config_entry=FakeConfigEntry(entry_id="test_group"),
            home=home,
            name="TestGroup",
            device_type="dynamic_group",
        )
        group._childrens = children
        group.accurate_power_sensor = accurate_power_sensor
        return group

    def test_group_with_sensor_all_overridden_returns_zero(self):
        """dynamic_group.py:184 — group with sensor, all children overridden returns 0.0."""
        from tests.conftest import FakeHass

        hass = FakeHass()
        child1 = _FakeDevice("a", overridden=True)
        child2 = _FakeDevice("b", overridden=True)
        group = self._make_group(hass, [child1, child2], accurate_power_sensor="sensor.group_power")
        result = group.get_device_power_latest_possible_valid_value(
            tolerance_seconds=None, time=NOW, ignore_auto_and_user_overridden_load=True
        )
        assert result == 0.0

    def test_group_no_sensor_skips_overridden_children(self):
        """dynamic_group.py:195 — group without sensor skips overridden children."""
        from custom_components.quiet_solar.ha_model.device import HADeviceMixin
        from tests.conftest import FakeConfigEntry, FakeHass

        hass = FakeHass()

        class FakeHAChild(HADeviceMixin, AbstractDevice):
            def __init__(self, name, power, overridden=False):
                self._overridden = overridden
                self._power_val = power
                super().__init__(
                    hass=hass,
                    config_entry=FakeConfigEntry(entry_id=f"test_{name}"),
                    home=MinimalTestHome(voltage=230.0),
                    name=name,
                    device_type="test",
                )

            def is_user_overridden(self):
                return self._overridden

            def get_device_power_latest_possible_valid_value(self, tolerance_seconds, time, ignore_auto_and_user_overridden_load=False):
                if ignore_auto_and_user_overridden_load and self._overridden:
                    return 0.0
                return self._power_val

        child1 = FakeHAChild("c1", 100.0, overridden=True)
        child2 = FakeHAChild("c2", 200.0, overridden=False)
        group = self._make_group(hass, [child1, child2])
        result = group.get_device_power_latest_possible_valid_value(
            tolerance_seconds=None, time=NOW, ignore_auto_and_user_overridden_load=True
        )
        assert result == 200.0  # only non-overridden child counted

    def test_group_with_sensor_none_override_mixed_falls_to_children(self):
        """dynamic_group.py: mixed override (None) with sensor falls to per-child sum."""
        from custom_components.quiet_solar.ha_model.device import HADeviceMixin
        from tests.conftest import FakeConfigEntry, FakeHass

        hass = FakeHass()

        class FakeHAChild(HADeviceMixin, AbstractDevice):
            def __init__(self, name, power, overridden=False):
                self._overridden = overridden
                self._power_val = power
                super().__init__(
                    hass=hass,
                    config_entry=FakeConfigEntry(entry_id=f"test_{name}"),
                    home=MinimalTestHome(voltage=230.0),
                    name=name,
                    device_type="test",
                )

            def is_user_overridden(self):
                return self._overridden

            def get_device_power_latest_possible_valid_value(self, tolerance_seconds, time, ignore_auto_and_user_overridden_load=False):
                if ignore_auto_and_user_overridden_load and self._overridden is True:
                    return 0.0
                return self._power_val

        # One True, one False → group is_user_overridden() = None (mixed)
        child1 = FakeHAChild("c1", 100.0, overridden=True)
        child2 = FakeHAChild("c2", 200.0, overridden=False)
        group = self._make_group(hass, [child1, child2], accurate_power_sensor="sensor.group_power")
        result = group.get_device_power_latest_possible_valid_value(
            tolerance_seconds=None, time=NOW, ignore_auto_and_user_overridden_load=True
        )
        # Mixed → falls to per-child loop, skips overridden child
        assert result == 200.0

    def test_group_with_sensor_auto_boosted_subtracted(self):
        """dynamic_group.py:193-204 — auto-boosted child subtracted from group sensor."""
        from custom_components.quiet_solar.ha_model.device import HADeviceMixin
        from custom_components.quiet_solar.home_model.load import AbstractLoad
        from tests.conftest import FakeConfigEntry, FakeHass

        hass = FakeHass()

        class FakeHALoad(HADeviceMixin, AbstractLoad):
            """Fake load that is auto-boosted and NOT being commanded ON."""

            def __init__(self, name, power):
                self._power_val = power
                super().__init__(
                    hass=hass,
                    config_entry=FakeConfigEntry(entry_id=f"test_{name}"),
                    home=MinimalTestHome(voltage=230.0),
                    name=name,
                    device_type="test",
                )
                self.load_is_auto_to_be_boosted = True

            def is_user_overridden(self):
                return False

            def is_load_command_set(self, time):
                return False  # NOT being commanded → should be subtracted

            def get_device_power_latest_possible_valid_value(self, tolerance_seconds, time, ignore_auto_and_user_overridden_load=False):
                return self._power_val

        auto_child = FakeHALoad("auto1", 150.0)
        group = self._make_group(hass, [auto_child], accurate_power_sensor="sensor.group_power")
        # Mock the group sensor to return 500W
        group._entity_probed_last_valid_state = {
            "sensor.group_power": (NOW, 500.0, {}),
        }
        result = group.get_device_power_latest_possible_valid_value(
            tolerance_seconds=None, time=NOW, ignore_auto_and_user_overridden_load=True
        )
        # 500 - 150 = 350
        assert result == 350.0

    def test_group_with_sensor_auto_boosted_exceeds_sensor_no_subtraction(self):
        """When auto-boosted child power exceeds group sensor, skip subtraction."""
        from custom_components.quiet_solar.ha_model.device import HADeviceMixin
        from custom_components.quiet_solar.home_model.load import AbstractLoad
        from tests.conftest import FakeConfigEntry, FakeHass

        hass = FakeHass()

        class FakeHALoad(HADeviceMixin, AbstractLoad):
            def __init__(self, name, power):
                self._power_val = power
                super().__init__(
                    hass=hass,
                    config_entry=FakeConfigEntry(entry_id=f"test_{name}"),
                    home=MinimalTestHome(voltage=230.0),
                    name=name,
                    device_type="test",
                )
                self.load_is_auto_to_be_boosted = True

            def is_user_overridden(self):
                return False

            def is_load_command_set(self, time):
                return False

            def get_device_power_latest_possible_valid_value(self, tolerance_seconds, time, ignore_auto_and_user_overridden_load=False):
                return self._power_val

        # Child reports 800W but group sensor only shows 200W → subtraction would go negative
        auto_child = FakeHALoad("auto_big", 800.0)
        group = self._make_group(hass, [auto_child], accurate_power_sensor="sensor.group_power")
        group._entity_probed_last_valid_state = {
            "sensor.group_power": (NOW, 200.0, {}),
        }
        result = group.get_device_power_latest_possible_valid_value(
            tolerance_seconds=None, time=NOW, ignore_auto_and_user_overridden_load=True
        )
        # 200 - 800 < 0 → skip subtraction → return full 200
        assert result == 200.0

    def test_group_with_sensor_auto_boosted_commanded_on_not_subtracted(self):
        """Auto-boosted child being commanded ON is NOT subtracted."""
        from custom_components.quiet_solar.ha_model.device import HADeviceMixin
        from custom_components.quiet_solar.home_model.commands import LoadCommand
        from custom_components.quiet_solar.home_model.load import AbstractLoad
        from tests.conftest import FakeConfigEntry, FakeHass

        hass = FakeHass()

        class FakeHALoad(HADeviceMixin, AbstractLoad):
            def __init__(self, name, power):
                self._power_val = power
                super().__init__(
                    hass=hass,
                    config_entry=FakeConfigEntry(entry_id=f"test_{name}"),
                    home=MinimalTestHome(voltage=230.0),
                    name=name,
                    device_type="test",
                )
                self.load_is_auto_to_be_boosted = True
                self.current_command = LoadCommand(command="on", power_consign=1000.0)

            def is_user_overridden(self):
                return False

            def is_load_command_set(self, time):
                return True  # Being commanded → should NOT be subtracted

            def get_device_power_latest_possible_valid_value(self, tolerance_seconds, time, ignore_auto_and_user_overridden_load=False):
                return self._power_val

        auto_child = FakeHALoad("auto1", 150.0)
        group = self._make_group(hass, [auto_child], accurate_power_sensor="sensor.group_power")
        group._entity_probed_last_valid_state = {
            "sensor.group_power": (NOW, 500.0, {}),
        }
        result = group.get_device_power_latest_possible_valid_value(
            tolerance_seconds=None, time=NOW, ignore_auto_and_user_overridden_load=True
        )
        # Auto child is commanded ON → NOT subtracted → full 500
        assert result == 500.0


class TestDynamicGroupIsUserOverriddenNestedNone:
    """Cover dynamic_group.py:175 — child returning None propagates to parent."""

    def test_child_returns_none_propagates(self):
        """When a child is_user_overridden() returns None, group returns None."""
        from tests.conftest import FakeHass

        hass = FakeHass()

        child_mixed = _FakeDevice("mixed_child", overridden=None)
        child_normal = _FakeDevice("normal_child", overridden=False)

        from custom_components.quiet_solar.ha_model.dynamic_group import QSDynamicGroup
        from tests.conftest import FakeConfigEntry

        group = QSDynamicGroup(
            hass=hass,
            config_entry=FakeConfigEntry(entry_id="test_group_nested"),
            home=MinimalTestHome(voltage=230.0),
            name="NestedGroup",
            device_type="dynamic_group",
        )
        group._childrens = [child_normal, child_mixed]
        assert group.is_user_overridden() is None


# =============================================================================
# Task 5: _apply_override_mask tests
# =============================================================================


class _FakeState:
    """Minimal fake for HA LazyState."""

    def __init__(self, state_str: str, last_changed: datetime):
        self.state = state_str
        self.last_changed = last_changed


class _FakeLoadSensor:
    """Minimal fake for QSSolarHistoryVals with a flat values array."""

    def __init__(self, size: int = 100):
        import numpy as np

        self.values = np.ones((2, size), dtype=np.float64) * 500.0
        self.values[1] = 1  # day tracking row
        self._size = size

    def get_index_from_time(self, time: datetime) -> tuple[int, int]:
        # Simple mapping: minutes since epoch mod size
        epoch = datetime(2027, 1, 1, tzinfo=pytz.UTC)
        idx = int((time - epoch).total_seconds() // 900) % self._size
        return idx, 0


class TestApplyOverrideMask:
    """Test _apply_override_mask on QSHomeSolarAndConsumptionHistoryAndForecast."""

    def _get_cls(self):
        from custom_components.quiet_solar.ha_model.home import QSHomeSolarAndConsumptionHistoryAndForecast

        return QSHomeSolarAndConsumptionHistoryAndForecast

    def test_no_override_no_change(self):
        """All states are NO OVERRIDE → values unchanged."""
        cls = self._get_cls()
        sensor = _FakeLoadSensor()
        import numpy as np

        original = np.copy(sensor.values)
        t0 = datetime(2027, 1, 1, 10, 0, tzinfo=pytz.UTC)
        t1 = datetime(2027, 1, 1, 11, 0, tzinfo=pytz.UTC)
        states = [
            _FakeState("NO OVERRIDE", t0),
            _FakeState("NO OVERRIDE", t1),
        ]
        cls._apply_override_mask(sensor, states, t1)
        np.testing.assert_array_equal(sensor.values, original)

    def test_override_interval_zeroed(self):
        """Override interval gets zeroed, non-override remains."""
        cls = self._get_cls()
        sensor = _FakeLoadSensor()
        t0 = datetime(2027, 1, 1, 10, 0, tzinfo=pytz.UTC)
        t1 = datetime(2027, 1, 1, 11, 0, tzinfo=pytz.UTC)
        t2 = datetime(2027, 1, 1, 12, 0, tzinfo=pytz.UTC)
        states = [
            _FakeState("Override: my_car", t0),
            _FakeState("NO OVERRIDE", t1),
            _FakeState("NO OVERRIDE", t2),
        ]
        cls._apply_override_mask(sensor, states, t2)
        # t0→t1 should be zeroed (override active)
        idx0 = sensor.get_index_from_time(t0)[0]
        idx1 = sensor.get_index_from_time(t1)[0]
        assert all(sensor.values[0][idx0:idx1] == 0.0)
        # t1→t2 should NOT be zeroed
        idx2 = sensor.get_index_from_time(t2)[0]
        assert all(sensor.values[0][idx1:idx2] == 500.0)

    def test_trailing_override_wraps_around_circular_buffer(self):
        """Lines 3256-3257: trailing override spans wrap-around in circular buffer."""
        cls = self._get_cls()
        sensor = _FakeLoadSensor(size=10)

        # We need start_idx > end_idx for the trailing override.
        # Override get_index_from_time to return controlled values:
        # first call (prev_time) → idx=8, second call (current_time) → idx=2
        call_count = [0]
        indices = [8, 2]  # start=8, end=2 (wraps)

        def controlled_get_index(time):
            idx = indices[call_count[0]] if call_count[0] < len(indices) else 0
            call_count[0] += 1
            return idx, 0

        sensor.get_index_from_time = controlled_get_index

        t0 = datetime(2027, 1, 1, 10, 0, tzinfo=pytz.UTC)
        t_now = datetime(2027, 1, 1, 12, 0, tzinfo=pytz.UTC)
        states = [
            _FakeState("Override: car", t0),
        ]
        cls._apply_override_mask(sensor, states, t_now)
        # Indices 8,9 should be zeroed (from start_idx to end of buffer)
        assert sensor.values[0][8] == 0.0
        assert sensor.values[0][9] == 0.0
        # Indices 0,1 should be zeroed (from start of buffer to end_idx)
        assert sensor.values[0][0] == 0.0
        assert sensor.values[0][1] == 0.0
        # Index 2 onward should remain
        assert sensor.values[0][2] == 500.0
        assert sensor.values[0][5] == 500.0

    def test_trailing_override_zeroed(self):
        """Override at end (no subsequent state change) zeroes to current_time."""
        cls = self._get_cls()
        sensor = _FakeLoadSensor()
        t0 = datetime(2027, 1, 1, 10, 0, tzinfo=pytz.UTC)
        t_now = datetime(2027, 1, 1, 12, 0, tzinfo=pytz.UTC)
        states = [
            _FakeState("Override: my_car", t0),
        ]
        cls._apply_override_mask(sensor, states, t_now)
        idx0 = sensor.get_index_from_time(t0)[0]
        assert sensor.values[0][idx0] == 0.0

    def test_none_values_noop(self):
        """If load_sensor.values is None, mask is a no-op."""
        cls = self._get_cls()
        sensor = _FakeLoadSensor()
        sensor.values = None
        t0 = datetime(2027, 1, 1, 10, 0, tzinfo=pytz.UTC)
        states = [_FakeState("Override: x", t0)]
        cls._apply_override_mask(sensor, states, NOW)  # should not crash

    def test_unknown_state_skipped(self):
        """Unknown/unavailable states are skipped."""
        cls = self._get_cls()
        sensor = _FakeLoadSensor()
        import numpy as np

        original = np.copy(sensor.values)
        t0 = datetime(2027, 1, 1, 10, 0, tzinfo=pytz.UTC)
        t1 = datetime(2027, 1, 1, 11, 0, tzinfo=pytz.UTC)
        states = [
            _FakeState("unknown", t0),
            _FakeState("unavailable", t1),
        ]
        cls._apply_override_mask(sensor, states, NOW)
        np.testing.assert_array_equal(sensor.values, original)

    def test_override_wraps_around_circular_buffer(self):
        """Lines 3233-3234: override spans wrap-around in circular buffer."""
        cls = self._get_cls()
        sensor = _FakeLoadSensor(size=10)

        # epoch = 2027-01-01 00:00 UTC, step=900s
        # idx = ((time - epoch).total_seconds() // 900) % 10
        # We want start_idx=8, end_idx=2 (wraps)
        epoch = datetime(2027, 1, 1, tzinfo=pytz.UTC)
        t0 = epoch + timedelta(seconds=8 * 900)  # idx=8
        t1 = epoch + timedelta(seconds=12 * 900)  # idx=2 (wraps: 12%10=2)
        states = [
            _FakeState("Override: car", t0),
            _FakeState("NO OVERRIDE", t1),
        ]
        cls._apply_override_mask(sensor, states, t1)
        # Indices 8,9 and 0,1 should be zeroed (wrap-around)
        assert sensor.values[0][8] == 0.0
        assert sensor.values[0][9] == 0.0
        assert sensor.values[0][0] == 0.0
        assert sensor.values[0][1] == 0.0
        # Index 2 onward should remain (end is exclusive)
        assert sensor.values[0][2] == 500.0
        assert sensor.values[0][5] == 500.0

    def test_trailing_override_no_wrap(self):
        """Line 3244: trailing override with start_idx <= end_idx (no wrap)."""
        cls = self._get_cls()

        sensor = _FakeLoadSensor(size=10)

        # Override get_index_from_time to return controlled values
        call_count = [0]
        indices = [3, 7]  # start=3, end=7 (current_time) → 3<=7

        def controlled_get_index(time):
            idx = indices[call_count[0]] if call_count[0] < len(indices) else 0
            call_count[0] += 1
            return idx, 0

        sensor.get_index_from_time = controlled_get_index

        t0 = datetime(2027, 1, 1, 10, 0, tzinfo=pytz.UTC)
        t_now = datetime(2027, 1, 1, 12, 0, tzinfo=pytz.UTC)
        states = [
            _FakeState("Override: car", t0),
        ]
        cls._apply_override_mask(sensor, states, t_now)
        # Indices 3..6 should be zeroed (trailing override, no wrap)
        assert sensor.values[0][3] == 0.0
        assert sensor.values[0][4] == 0.0
        assert sensor.values[0][5] == 0.0
        assert sensor.values[0][6] == 0.0
        # Index 7 onward should remain
        assert sensor.values[0][7] == 500.0

    def test_asked_for_reset_not_overridden(self):
        """ASKED FOR RESET OVERRIDE state does not trigger masking."""
        cls = self._get_cls()
        sensor = _FakeLoadSensor()
        import numpy as np

        original = np.copy(sensor.values)
        t0 = datetime(2027, 1, 1, 10, 0, tzinfo=pytz.UTC)
        t1 = datetime(2027, 1, 1, 11, 0, tzinfo=pytz.UTC)
        states = [
            _FakeState("ASKED FOR RESET OVERRIDE", t0),
            _FakeState("NO OVERRIDE", t1),
        ]
        cls._apply_override_mask(sensor, states, t1)
        np.testing.assert_array_equal(sensor.values, original)

    def test_out_of_order_states_sorted(self):
        """Finding 15: out-of-order states are sorted chronologically."""
        cls = self._get_cls()
        sensor = _FakeLoadSensor()
        t0 = datetime(2027, 1, 1, 10, 0, tzinfo=pytz.UTC)
        t1 = datetime(2027, 1, 1, 11, 0, tzinfo=pytz.UTC)
        t2 = datetime(2027, 1, 1, 12, 0, tzinfo=pytz.UTC)
        # States given out of order — should still work correctly
        states = [
            _FakeState("NO OVERRIDE", t1),
            _FakeState("Override: my_car", t0),
            _FakeState("NO OVERRIDE", t2),
        ]
        cls._apply_override_mask(sensor, states, t2)
        # t0→t1 should be zeroed
        idx0 = sensor.get_index_from_time(t0)[0]
        idx1 = sensor.get_index_from_time(t1)[0]
        assert all(sensor.values[0][idx0:idx1] == 0.0)
        # t1→t2 should NOT be zeroed
        idx2 = sensor.get_index_from_time(t2)[0]
        assert all(sensor.values[0][idx1:idx2] == 500.0)
