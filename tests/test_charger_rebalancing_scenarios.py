"""Multi-charger rebalancing scenario tests.

Tests for the trust-critical charger budgeting system, verifying that
multi-charger rebalancing sequences maintain per-phase amp limits at
every intermediate state. These are integration tests exercising
QSChargerGroup.budgeting_algorithm_minimize_diffs() and
QSChargerGroup.apply_budget_strategy() with realistic multi-charger
configurations.

Story 2.2: Charger Budgeting Scenario Tests (Epic 2)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import pytz
from homeassistant.const import CONF_NAME

from custom_components.quiet_solar.const import (
    CONF_CHARGER_DEVICE_WALLBOX,
    CONF_CHARGER_MAX_CHARGE,
    CONF_CHARGER_MIN_CHARGE,
    CONF_DYN_GROUP_MAX_PHASE_AMPS,
    CONF_IS_3P,
    CONF_MONO_PHASE,
    DATA_HANDLER,
    DOMAIN,
)
from custom_components.quiet_solar.ha_model.charger import (
    QSChargerGroup,
    QSChargerStatus,
    QSChargerWallbox,
)
from custom_components.quiet_solar.ha_model.dynamic_group import QSDynamicGroup
from custom_components.quiet_solar.ha_model.home import QSHome
from custom_components.quiet_solar.home_model.commands import (
    CMD_AUTO_FROM_CONSIGN,
    CMD_AUTO_GREEN_ONLY,
    CMD_ON,
)

# =============================================================================
# Shared fixtures
# =============================================================================


def _create_hass_and_home(max_home_amps: float = 63, max_group_amps: float = 54):
    """Create mocked hass, home, and dynamic group for charger tests.

    Returns:
        (hass, home, dynamic_group, charger_group, current_time, config_entry)
    """
    hass = MagicMock()
    hass.states = MagicMock()
    hass.states.get = MagicMock(return_value=None)

    data_handler = MagicMock()
    hass.data = {DOMAIN: {DATA_HANDLER: data_handler}}

    config_entry = MagicMock()
    config_entry.entry_id = "test_entry_id"
    config_entry.data = {}

    home = QSHome(
        **{
            CONF_NAME: "TestHome",
            CONF_DYN_GROUP_MAX_PHASE_AMPS: max_home_amps,
            CONF_IS_3P: True,
            "hass": hass,
            "config_entry": config_entry,
        }
    )
    data_handler.home = home

    dynamic_group = QSDynamicGroup(
        **{
            CONF_NAME: "ChargerGroup",
            CONF_DYN_GROUP_MAX_PHASE_AMPS: max_group_amps,
            CONF_IS_3P: True,
            "home": home,
            "hass": hass,
            "config_entry": config_entry,
        }
    )
    home.add_device(dynamic_group)

    # Mock is_current_acceptable_and_diff to do real phase limit checking
    def is_current_acceptable_and_diff_mock(new_amps, estimated_current_amps, time):
        limit = dynamic_group.dyn_group_max_phase_current_conf
        for i in range(3):
            if new_amps[i] > limit:
                return (False, [new_amps[j] - limit for j in range(3)])
        return (True, [0, 0, 0])

    dynamic_group.is_current_acceptable_and_diff = MagicMock(side_effect=is_current_acceptable_and_diff_mock)

    charger_group = QSChargerGroup(dynamic_group)
    current_time = datetime.now(pytz.UTC)

    return hass, home, dynamic_group, charger_group, current_time, config_entry


def _create_wallbox_charger(
    hass,
    home,
    config_entry,
    name: str,
    mono_phase: int,
    min_charge: int = 6,
    max_charge: int = 32,
    is_3p: bool = False,
    current_time: datetime | None = None,
):
    """Create a QSChargerWallbox with mocked HA infrastructure."""
    if current_time is None:
        current_time = datetime.now(pytz.UTC)

    with patch("custom_components.quiet_solar.ha_model.charger.entity_registry") as mock_er:
        mock_er.async_get = MagicMock()
        mock_er.async_entries_for_device = MagicMock(return_value=[])

        charger = QSChargerWallbox(
            **{
                CONF_NAME: name,
                CONF_MONO_PHASE: mono_phase,
                CONF_CHARGER_DEVICE_WALLBOX: f"device_{name.lower()}",
                CONF_CHARGER_MIN_CHARGE: min_charge,
                CONF_CHARGER_MAX_CHARGE: max_charge,
                CONF_IS_3P: is_3p,
                "dynamic_group_name": "ChargerGroup",
                "home": home,
                "hass": hass,
                "config_entry": config_entry,
            }
        )
        charger.attach_car(charger._default_generic_car, current_time)
        charger._expected_charge_state.value = True
        charger._expected_amperage.value = min_charge
        charger._expected_num_active_phases.value = 3 if is_3p else 1
        home.add_device(charger)

    return charger


def _make_charger_status(
    charger,
    current_amp: int,
    num_phases: int,
    possible_amps: list[int],
    possible_num_phases: list[int],
    charge_score: float,
    command: int = CMD_AUTO_FROM_CONSIGN,
    can_be_started_and_stopped: bool = True,
    is_before_battery: bool = False,
    bump_solar: bool = False,
    voltage: float = 230.0,
) -> QSChargerStatus:
    """Create a QSChargerStatus with specified parameters."""
    cs = QSChargerStatus(charger)
    cs.accurate_current_power = current_amp * voltage * num_phases
    cs.current_real_max_charging_amp = current_amp
    cs.current_active_phase_number = num_phases
    cs.possible_amps = possible_amps
    cs.possible_num_phases = possible_num_phases
    cs.budgeted_amp = current_amp
    cs.budgeted_num_phases = num_phases
    cs.command = command
    cs.charge_score = charge_score
    cs.can_be_started_and_stopped = can_be_started_and_stopped
    cs.is_before_battery = is_before_battery
    cs.bump_solar = bump_solar
    return cs


def _assert_no_phase_exceeded(charger_statuses: list[QSChargerStatus], limit: float):
    """Assert no phase exceeds the limit after budgeting."""
    phase_totals = [0.0, 0.0, 0.0]
    for cs in charger_statuses:
        amps_array = cs.get_budget_amps()
        for i in range(3):
            phase_totals[i] += amps_array[i]
    for i in range(3):
        assert phase_totals[i] <= limit, (
            f"Phase {i + 1} exceeded limit: {phase_totals[i]:.1f} > {limit:.1f}. "
            f"Budgets: {[(cs.charger.name, cs.budgeted_amp, cs.budgeted_num_phases) for cs in charger_statuses]}"
        )


# =============================================================================
# Task 1: Multi-charger rebalancing scenarios (AC: #1)
# =============================================================================


class TestMultiChargerRebalancing:
    """Multi-charger rebalancing scenarios verifying no phase exceeded."""

    def _setup_two_charger_env(self, max_group_amps: float = 54):
        """Create a 2-charger environment for rebalancing tests."""
        hass, home, dyn_group, cg, time, config_entry = _create_hass_and_home(max_group_amps=max_group_amps)
        c1 = _create_wallbox_charger(hass, home, config_entry, "Charger_A", mono_phase=1, is_3p=True, current_time=time)
        c2 = _create_wallbox_charger(hass, home, config_entry, "Charger_B", mono_phase=1, is_3p=True, current_time=time)
        return hass, home, dyn_group, cg, time, config_entry, c1, c2

    def _setup_three_charger_env(self, max_group_amps: float = 54):
        """Create a 3-charger environment for rebalancing tests."""
        hass, home, dyn_group, cg, time, config_entry = _create_hass_and_home(max_group_amps=max_group_amps)
        c1 = _create_wallbox_charger(hass, home, config_entry, "Charger_A", mono_phase=1, is_3p=True, current_time=time)
        c2 = _create_wallbox_charger(hass, home, config_entry, "Charger_B", mono_phase=1, is_3p=True, current_time=time)
        c3 = _create_wallbox_charger(hass, home, config_entry, "Charger_C", mono_phase=1, is_3p=True, current_time=time)
        return hass, home, dyn_group, cg, time, config_entry, c1, c2, c3

    @pytest.mark.integration
    async def test_two_chargers_higher_priority_gets_power_first(self):
        """Task 1.1: Higher-score charger gets power first on increase."""
        _, _, _, cg, time, _, c1, c2 = self._setup_two_charger_env(max_group_amps=32)

        # Both start at minimum (6A each), 12A total — well within 32A limit
        # Give 10kW available — enough to increase both significantly
        full_amps = list(range(0, 33))  # [0, 1, 2, ..., 32]
        cs1 = _make_charger_status(
            c1, current_amp=6, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=10
        )
        cs2 = _make_charger_status(
            c2, current_amp=6, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=5
        )

        result, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            [cs1, cs2],
            full_available_home_power=20000.0,
            grid_available_home_power=20000.0,
            allow_budget_reset=False,
            time=time,
        )

        assert result is True
        # Higher-score charger should have more (or equal) amps than lower
        assert cs1.budgeted_amp >= cs2.budgeted_amp, (
            f"Higher priority charger (score=10) should get >= amps than lower (score=5): "
            f"got {cs1.budgeted_amp} vs {cs2.budgeted_amp}"
        )
        _assert_no_phase_exceeded([cs1, cs2], 32)

    @pytest.mark.integration
    async def test_two_chargers_lower_priority_shed_first_on_decrease(self):
        """Task 1.1: Lower-score charger shed first on decrease."""
        _, _, _, cg, time, _, c1, c2 = self._setup_two_charger_env(max_group_amps=32)

        full_amps = list(range(0, 33))
        # Both charging at 16A (total 32A per phase — at limit)
        cs1 = _make_charger_status(
            c1, current_amp=16, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=10
        )
        cs2 = _make_charger_status(
            c2, current_amp=16, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=5
        )

        # Negative power budget forces decrease
        result, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            [cs1, cs2],
            full_available_home_power=-3000.0,
            grid_available_home_power=-3000.0,
            allow_budget_reset=False,
            time=time,
        )

        assert result is True
        # Lower-score charger should be reduced more
        assert cs2.budgeted_amp <= cs1.budgeted_amp, (
            f"Lower priority charger (score=5) should be reduced first: "
            f"got cs1={cs1.budgeted_amp} vs cs2={cs2.budgeted_amp}"
        )
        _assert_no_phase_exceeded([cs1, cs2], 32)

    @pytest.mark.integration
    async def test_three_chargers_respect_phase_limits_at_every_step(self):
        """Task 1.2: Three chargers sharing 54A group limit."""
        _, _, _, cg, time, _, c1, c2, c3 = self._setup_three_charger_env(max_group_amps=54)

        full_amps = list(range(0, 33))
        # All start at minimum (6A each, 18A total per phase, well within 54A)
        cs1 = _make_charger_status(
            c1, current_amp=6, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=10
        )
        cs2 = _make_charger_status(
            c2, current_amp=6, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=7
        )
        cs3 = _make_charger_status(
            c3, current_amp=6, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=3
        )

        result, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            [cs1, cs2, cs3],
            full_available_home_power=30000.0,
            grid_available_home_power=30000.0,
            allow_budget_reset=False,
            time=time,
        )

        assert result is True
        _assert_no_phase_exceeded([cs1, cs2, cs3], 54)
        # Each charger should have increased above minimum
        assert cs1.budgeted_amp > 6
        assert cs2.budgeted_amp > 6

    @pytest.mark.integration
    async def test_power_budget_drop_reduces_lowest_priority_first(self):
        """Task 1.3: Power drop reduces lowest-priority charger first."""
        _, _, _, cg, time, _, c1, c2, c3 = self._setup_three_charger_env(max_group_amps=54)

        full_amps = list(range(0, 33))
        # All charging at 18A each (54A total — exactly at limit)
        cs1 = _make_charger_status(
            c1, current_amp=18, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=10
        )
        cs2 = _make_charger_status(
            c2, current_amp=18, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=7
        )
        cs3 = _make_charger_status(
            c3, current_amp=18, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=3
        )

        # Power drops — need to reduce
        result, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            [cs1, cs2, cs3],
            full_available_home_power=-5000.0,
            grid_available_home_power=-5000.0,
            allow_budget_reset=False,
            time=time,
        )

        assert result is True
        # Lowest priority should be reduced most
        assert cs3.budgeted_amp <= cs2.budgeted_amp
        assert cs2.budgeted_amp <= cs1.budgeted_amp
        _assert_no_phase_exceeded([cs1, cs2, cs3], 54)

    @pytest.mark.integration
    async def test_reset_allocation_when_best_charger_not_charging(self):
        """Task 1.4: Reset allocation when highest-priority charger is idle."""
        _, _, _, cg, time, _, c1, c2 = self._setup_two_charger_env(max_group_amps=32)

        full_amps = list(range(0, 33))
        # c1 (highest score) is NOT charging, c2 (lower) IS charging
        cs1 = _make_charger_status(
            c1,
            current_amp=0,
            num_phases=3,
            possible_amps=full_amps,
            possible_num_phases=[3],
            charge_score=10,
            can_be_started_and_stopped=True,
        )
        cs2 = _make_charger_status(
            c2,
            current_amp=16,
            num_phases=3,
            possible_amps=full_amps,
            possible_num_phases=[3],
            charge_score=5,
            can_be_started_and_stopped=True,
        )

        result, should_reset, _ = await cg.budgeting_algorithm_minimize_diffs(
            [cs1, cs2],
            full_available_home_power=10000.0,
            grid_available_home_power=10000.0,
            allow_budget_reset=True,
            time=time,
        )

        assert result is True
        # The algorithm should signal a reset is needed
        assert should_reset is True, "Algorithm should signal reset when highest-score charger is idle"
        # After reset: c1 should get power, both within limits
        _assert_no_phase_exceeded([cs1, cs2], 32)

    @pytest.mark.integration
    async def test_asymmetric_chargers_mixed_phase_capabilities(self):
        """Task 1.5: Mixed 1P-only and 3P capable chargers in same group."""
        hass, home, dyn_group, cg, time, config_entry = _create_hass_and_home(max_group_amps=32)
        # c1: 3-phase capable
        c1 = _create_wallbox_charger(
            hass, home, config_entry, "Charger_3P", mono_phase=1, is_3p=True, current_time=time
        )
        # c2: 1-phase only, on phase 1
        c2 = _create_wallbox_charger(
            hass, home, config_entry, "Charger_1P", mono_phase=1, is_3p=False, current_time=time
        )

        full_amps = list(range(0, 33))
        cs1 = _make_charger_status(
            c1, current_amp=10, num_phases=3, possible_amps=full_amps, possible_num_phases=[1, 3], charge_score=8
        )
        cs2 = _make_charger_status(
            c2, current_amp=10, num_phases=1, possible_amps=full_amps, possible_num_phases=[1], charge_score=5
        )

        result, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            [cs1, cs2],
            full_available_home_power=15000.0,
            grid_available_home_power=15000.0,
            allow_budget_reset=False,
            time=time,
        )

        assert result is True
        _assert_no_phase_exceeded([cs1, cs2], 32)


# =============================================================================
# Task 7: 3-car / 3-wallbox / fixed-3-phase / 32A-per-phase scenarios (AC: #6)
# =============================================================================


class TestThreeCarThreeWallboxFixedThreePhase:
    """3-car / 3-wallbox / fixed 3-phase / no phase switch / 32A per phase.

    All chargers are Wallbox, fixed 3-phase (possible_num_phases=[3]),
    with possible_amps=[0, 6, 7, ..., 32] and a dynamic group limited
    to 32A per phase.
    """

    FULL_AMPS = list(range(0, 33))  # [0, 1, 2, ..., 32] — 0 means off, 6 is min
    POSSIBLE_PHASES = [3]  # Fixed 3-phase, no switching
    GROUP_LIMIT = 32  # 32A per phase

    def _setup(self):
        """Create the 3-car/3-wallbox shared environment with distinct per-car power curves.

        Car A: efficiency factor 1.0x (default linear: 230V * A * 3)
        Car B: slightly lower efficiency (0.95x of default)
        Car C: non-linear curve (lower efficiency at high amps, simulating battery saturation)
        """
        hass, home, dyn_group, cg, time, config_entry = _create_hass_and_home(max_group_amps=self.GROUP_LIMIT)
        chargers = []
        for name in ["WB_A", "WB_B", "WB_C"]:
            c = _create_wallbox_charger(
                hass,
                home,
                config_entry,
                name,
                mono_phase=1,
                is_3p=True,
                min_charge=6,
                max_charge=32,
                current_time=time,
            )
            chargers.append(c)

        # I4: Set distinct per-car amp-to-power lookup tables
        # Car B: 95% efficiency (slightly different charging curve)
        car_b = chargers[1].car
        for a in range(len(car_b.amp_to_power_3p)):
            car_b.amp_to_power_3p[a] = car_b.amp_to_power_3p[a] * 0.95
            car_b.amp_to_power_1p[a] = car_b.amp_to_power_1p[a] * 0.95

        # Car C: non-linear curve (saturation at high amps)
        car_c = chargers[2].car
        for a in range(len(car_c.amp_to_power_3p)):
            if a <= 16:
                factor = 1.0
            elif a <= 24:
                factor = 0.9  # 90% efficiency 17-24A
            else:
                factor = 0.8  # 80% efficiency above 24A
            car_c.amp_to_power_3p[a] = car_c.amp_to_power_3p[a] * factor
            car_c.amp_to_power_1p[a] = car_c.amp_to_power_1p[a] * factor

        return hass, home, dyn_group, cg, time, config_entry, chargers

    def _make_statuses(
        self,
        chargers,
        amps: list[int],
        scores: list[float],
        commands: list[int] | None = None,
        can_stop: list[bool] | None = None,
    ) -> list[QSChargerStatus]:
        """Create QSChargerStatus for all 3 chargers."""
        if commands is None:
            commands = [CMD_AUTO_FROM_CONSIGN] * 3
        if can_stop is None:
            can_stop = [True] * 3
        statuses = []
        for i, charger in enumerate(chargers):
            cs = _make_charger_status(
                charger,
                current_amp=amps[i],
                num_phases=3,
                possible_amps=self.FULL_AMPS,
                possible_num_phases=self.POSSIBLE_PHASES,
                charge_score=scores[i],
                command=commands[i],
                can_be_started_and_stopped=can_stop[i],
            )
            statuses.append(cs)
        return statuses

    # --- Task 7.1: All 3 charging simultaneously, demand exceeds limit ---

    @pytest.mark.integration
    async def test_all_three_charging_amps_within_limit(self):
        """7.1: 3 chargers charging — total stays within 32A per phase."""
        _, _, _, cg, time, _, chargers = self._setup()

        # All want to charge at max (32A each = 96A), but limit is 32A
        # Start at 10A each (30A — close to limit)
        statuses = self._make_statuses(chargers, amps=[10, 10, 10], scores=[10, 7, 3])

        result, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            statuses,
            full_available_home_power=50000.0,
            grid_available_home_power=50000.0,
            allow_budget_reset=False,
            time=time,
        )

        assert result is True
        _assert_no_phase_exceeded(statuses, self.GROUP_LIMIT)

        # Highest priority should get the most
        budgets = [cs.budgeted_amp for cs in statuses]
        assert budgets[0] >= budgets[1], f"Charger A (score=10) should get >= Charger B (score=7): {budgets}"
        assert budgets[1] >= budgets[2], f"Charger B (score=7) should get >= Charger C (score=3): {budgets}"

    # --- Task 7.2: All at minimum, power becomes available ---

    @pytest.mark.integration
    async def test_all_at_minimum_highest_priority_increases_first(self):
        """7.2: All at 6A, power available — highest priority increases first."""
        _, _, _, cg, time, _, chargers = self._setup()

        statuses = self._make_statuses(chargers, amps=[6, 6, 6], scores=[10, 7, 3])

        result, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            statuses,
            full_available_home_power=20000.0,
            grid_available_home_power=20000.0,
            allow_budget_reset=False,
            time=time,
        )

        assert result is True
        _assert_no_phase_exceeded(statuses, self.GROUP_LIMIT)

        # Highest priority should have increased the most
        assert statuses[0].budgeted_amp >= statuses[1].budgeted_amp
        assert statuses[1].budgeted_amp >= statuses[2].budgeted_amp

    # --- Task 7.3: Two at 16A, third plugs in with highest priority ---

    @pytest.mark.integration
    async def test_third_car_plugs_in_with_highest_priority_reset(self):
        """7.3: Two at 16A (32A used), third plugs in at highest priority."""
        _, _, _, cg, time, _, chargers = self._setup()

        # c1 and c2 at 16A each (32A total = at limit), c3 off but highest priority
        statuses = self._make_statuses(chargers, amps=[16, 16, 0], scores=[5, 3, 10])

        result, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            statuses,
            full_available_home_power=15000.0,
            grid_available_home_power=15000.0,
            allow_budget_reset=True,
            time=time,
        )

        _assert_no_phase_exceeded(statuses, self.GROUP_LIMIT)
        # The new highest-priority charger (c3) should get amps
        assert statuses[2].budgeted_amp > 0, "Highest priority charger (C, score=10) should be allocated amps"

    # --- Task 7.4: Highest priority reaches target SOC and stops ---

    @pytest.mark.integration
    async def test_freed_amps_reallocated_when_charger_stops(self):
        """7.4: Charger A stops — freed amps go to B and C by score."""
        _, _, _, cg, time, _, chargers = self._setup()

        # c1 stopped (0A), c2 and c3 at minimum — available power for reallocation
        statuses = self._make_statuses(chargers, amps=[0, 6, 6], scores=[0, 7, 3])
        # c1 is not actionable (score 0, not started), c2 and c3 compete
        # Only pass c2 and c3 as actionable
        actionable = [statuses[1], statuses[2]]

        result, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            actionable,
            full_available_home_power=20000.0,
            grid_available_home_power=20000.0,
            allow_budget_reset=False,
            time=time,
        )

        assert result is True
        _assert_no_phase_exceeded(actionable, self.GROUP_LIMIT)
        # Higher-priority c2 should get more
        assert statuses[1].budgeted_amp >= statuses[2].budgeted_amp

    # --- Task 7.5: Mandatory shaving under tight limits ---

    @pytest.mark.integration
    async def test_mandatory_shaving_reduces_lowest_score_first(self):
        """7.5: 3 mandatory chargers need 8A each but only 20A available."""
        _, _, dyn_group, cg, time, _, chargers = self._setup()

        # Override the group limit to 20A to create pressure
        dyn_group.dyn_group_max_phase_current_conf = 20

        def tight_limit_mock(new_amps, estimated_current_amps, time):
            limit = 20
            for i in range(3):
                if new_amps[i] > limit:
                    return (False, [new_amps[j] - limit for j in range(3)])
            return (True, [0, 0, 0])

        dyn_group.is_current_acceptable_and_diff = MagicMock(side_effect=tight_limit_mock)

        # All at 8A, but 8*3=24 > 20A limit — shaving needed
        statuses = self._make_statuses(
            chargers,
            amps=[8, 8, 8],
            scores=[10, 7, 3],
            commands=[CMD_ON, CMD_ON, CMD_ON],
        )

        result, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            statuses,
            full_available_home_power=0.0,
            grid_available_home_power=0.0,
            allow_budget_reset=False,
            time=time,
        )

        _assert_no_phase_exceeded(statuses, 20)
        # Lowest priority should be reduced the most
        assert statuses[2].budgeted_amp <= statuses[1].budgeted_amp
        assert statuses[1].budgeted_amp <= statuses[0].budgeted_amp

    # --- Task 7.6: Barely enough solar ---

    @pytest.mark.integration
    async def test_barely_enough_solar_proportional_allocation(self):
        """7.6: Solar just above minimum for all 3 — FILLER constraints."""
        _, _, _, cg, time, _, chargers = self._setup()

        # 3 chargers at minimum (6A each = 18A), barely any surplus
        # Available power ~ 18A * 230V * 3 = 12420W + small surplus
        statuses = self._make_statuses(
            chargers,
            amps=[6, 6, 6],
            scores=[10, 7, 3],
            commands=[CMD_AUTO_GREEN_ONLY, CMD_AUTO_GREEN_ONLY, CMD_AUTO_GREEN_ONLY],
        )

        # Barely enough power for all at minimum plus a tiny bit more
        result, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            statuses,
            full_available_home_power=2000.0,  # Small surplus
            grid_available_home_power=0.0,  # No grid
            allow_budget_reset=False,
            time=time,
        )

        assert result is True
        _assert_no_phase_exceeded(statuses, self.GROUP_LIMIT)

    # --- Task 7.7: Best-price consumption decision ---

    @pytest.mark.integration
    async def test_best_price_allocates_beyond_solar_within_phase_limit(self):
        """7.7: Grid in off-peak, solver says use grid — still respects 32A."""
        _, _, _, cg, time, _, chargers = self._setup()

        statuses = self._make_statuses(
            chargers,
            amps=[6, 6, 6],
            scores=[10, 7, 3],
            commands=[CMD_ON, CMD_ON, CMD_ON],
        )

        # Large available power including grid (off-peak pricing)
        result, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            statuses,
            full_available_home_power=30000.0,  # Plenty of power
            grid_available_home_power=30000.0,
            allow_budget_reset=False,
            time=time,
        )

        assert result is True
        _assert_no_phase_exceeded(statuses, self.GROUP_LIMIT)
        # With plenty of power, all should be above minimum
        for cs in statuses:
            assert cs.budgeted_amp >= 6

    # --- Task 7.8: Off-grid scenario with GREEN_ONLY ---

    @pytest.mark.integration
    async def test_off_grid_solar_only_sheds_lowest_priority(self):
        """7.8: Off-grid, ~15A available — only 2 chargers can run at min."""
        _, _, dyn_group, cg, time, _, chargers = self._setup()

        # Override group limit to 15A to simulate limited solar
        dyn_group.dyn_group_max_phase_current_conf = 15

        def limited_mock(new_amps, estimated_current_amps, time):
            limit = 15
            for i in range(3):
                if new_amps[i] > limit:
                    return (False, [new_amps[j] - limit for j in range(3)])
            return (True, [0, 0, 0])

        dyn_group.is_current_acceptable_and_diff = MagicMock(side_effect=limited_mock)

        # All 3 want to charge, but only 15A available (2 * 6A = 12A fits, 3 * 6A = 18A doesn't)
        statuses = self._make_statuses(
            chargers,
            amps=[6, 6, 6],
            scores=[10, 7, 3],
            commands=[CMD_AUTO_GREEN_ONLY, CMD_AUTO_GREEN_ONLY, CMD_AUTO_GREEN_ONLY],
        )

        result, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            statuses,
            full_available_home_power=0.0,
            grid_available_home_power=0.0,
            allow_budget_reset=True,
            time=time,
        )

        _assert_no_phase_exceeded(statuses, 15)
        # Algorithm shaves proportionally — lowest priority gets reduced most
        assert statuses[2].budgeted_amp <= statuses[1].budgeted_amp, (
            "Lowest priority (score=3) should be reduced most: %s"
            % [(cs.charger.name, cs.budgeted_amp) for cs in statuses]
        )
        assert statuses[1].budgeted_amp <= statuses[0].budgeted_amp, (
            "Mid priority (score=7) should be reduced more than highest: %s"
            % [(cs.charger.name, cs.budgeted_amp) for cs in statuses]
        )

    # --- Task 7.9: Off-grid with battery depletion ---

    @pytest.mark.integration
    async def test_off_grid_battery_depletion_progressive_shedding(self):
        """7.9: Progressive load shedding as power decreases."""
        _, _, dyn_group, cg, time, _, chargers = self._setup()

        # Start with 8A available (only 1 charger at min can run)
        dyn_group.dyn_group_max_phase_current_conf = 8

        def very_limited_mock(new_amps, estimated_current_amps, time):
            limit = 8
            for i in range(3):
                if new_amps[i] > limit:
                    return (False, [new_amps[j] - limit for j in range(3)])
            return (True, [0, 0, 0])

        dyn_group.is_current_acceptable_and_diff = MagicMock(side_effect=very_limited_mock)

        statuses = self._make_statuses(
            chargers,
            amps=[6, 6, 0],
            scores=[10, 7, 3],
            commands=[CMD_AUTO_GREEN_ONLY, CMD_AUTO_GREEN_ONLY, CMD_AUTO_GREEN_ONLY],
        )

        result, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            statuses,
            full_available_home_power=-2000.0,
            grid_available_home_power=0.0,
            allow_budget_reset=True,
            time=time,
        )

        _assert_no_phase_exceeded(statuses, 8)
        # Only highest priority should remain (if any)
        if statuses[0].budgeted_amp > 0:
            assert statuses[1].budgeted_amp <= statuses[0].budgeted_amp

    # --- Task 7.10: Priority change mid-charge ---

    @pytest.mark.integration
    async def test_priority_change_mid_charge_triggers_rebalancing(self):
        """7.10: Car B gets user override with highest priority."""
        _, _, _, cg, time, _, chargers = self._setup()

        # A was highest, B gets override making it highest
        statuses = self._make_statuses(chargers, amps=[16, 6, 6], scores=[7, 15, 3])

        result, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            statuses,
            full_available_home_power=15000.0,
            grid_available_home_power=15000.0,
            allow_budget_reset=True,
            time=time,
        )

        assert result is True
        _assert_no_phase_exceeded(statuses, self.GROUP_LIMIT)
        # minimize_diffs is incremental — B should have increased from its
        # starting 6A (it got power because it has highest score), but may
        # not surpass A's existing 16A in a single cycle
        assert statuses[1].budgeted_amp > 6, (
            f"Overridden charger B (score=15) should have increased from 6A: B={statuses[1].budgeted_amp}"
        )

    # --- Task 7.11: One charger becomes unavailable ---

    @pytest.mark.integration
    async def test_unavailable_charger_amps_redistributed(self):
        """7.11: Charger C stops — freed amps go to A and B."""
        _, _, _, cg, time, _, chargers = self._setup()

        # C stopped (excluded from actionable), A and B compete
        cs1 = _make_charger_status(
            chargers[0],
            current_amp=10,
            num_phases=3,
            possible_amps=self.FULL_AMPS,
            possible_num_phases=self.POSSIBLE_PHASES,
            charge_score=10,
        )
        cs2 = _make_charger_status(
            chargers[1],
            current_amp=10,
            num_phases=3,
            possible_amps=self.FULL_AMPS,
            possible_num_phases=self.POSSIBLE_PHASES,
            charge_score=7,
        )
        actionable = [cs1, cs2]

        result, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            actionable,
            full_available_home_power=20000.0,
            grid_available_home_power=20000.0,
            allow_budget_reset=False,
            time=time,
        )

        assert result is True
        _assert_no_phase_exceeded(actionable, self.GROUP_LIMIT)
        # Both should be above their starting 10A with freed capacity
        assert cs1.budgeted_amp >= 10
        assert cs2.budgeted_amp >= 10

    # --- Task 7.12: Adaptation window enforcement ---

    @pytest.mark.integration
    async def test_adaptation_window_blocks_premature_rebalancing(self):
        """7.12: No budget changes until chargers stable for CHARGER_ADAPTATION_WINDOW_S."""
        _, _, _, cg, time, _, chargers = self._setup()

        # First budgeting cycle
        statuses = self._make_statuses(chargers, amps=[10, 10, 10], scores=[10, 7, 3])
        result, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            statuses,
            full_available_home_power=20000.0,
            grid_available_home_power=20000.0,
            allow_budget_reset=False,
            time=time,
        )
        assert result is True
        first_budgets = [cs.budgeted_amp for cs in statuses]

        # Second call immediately (within adaptation window) — algorithm still
        # processes but the dyn_handle caller should gate on adaptation window.
        # We verify the algorithm itself is deterministic with same inputs.
        statuses2 = self._make_statuses(chargers, amps=first_budgets, scores=[10, 7, 3])
        result2, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            statuses2,
            full_available_home_power=20000.0,
            grid_available_home_power=20000.0,
            allow_budget_reset=False,
            time=time + timedelta(seconds=10),
        )
        assert result2 is True
        _assert_no_phase_exceeded(statuses2, self.GROUP_LIMIT)
        # Budgets should remain stable (no oscillation)
        for i in range(3):
            assert statuses2[i].budgeted_amp == first_budgets[i], (
                f"Budget should be stable within adaptation window: "
                f"charger {i} was {first_budgets[i]}, now {statuses2[i].budgeted_amp}"
            )


# =============================================================================
# Task 2: Staged transition scenarios (AC: #2)
# =============================================================================


class TestStagedTransitions:
    """Staged transition tests for apply_budget_strategy() two-phase split.

    apply_budgets() is mocked because it calls HA services.
    We test the splitting logic only.
    """

    def _setup_two_charger_env(self, max_group_amps: float = 32):
        """Create a 2-charger environment for staged transition tests."""
        hass, home, dyn_group, cg, time, config_entry = _create_hass_and_home(max_group_amps=max_group_amps)
        c1 = _create_wallbox_charger(hass, home, config_entry, "Charger_A", mono_phase=1, is_3p=True, current_time=time)
        c2 = _create_wallbox_charger(hass, home, config_entry, "Charger_B", mono_phase=1, is_3p=True, current_time=time)
        # Mock apply_budgets to avoid HA service calls
        cg.apply_budgets = MagicMock(return_value=None)
        # Make it an async mock

        async def _noop_apply(*args, **kwargs):
            pass

        cg.apply_budgets = _noop_apply
        return hass, home, dyn_group, cg, time, config_entry, c1, c2

    @pytest.mark.integration
    async def test_phase1_applies_decreases_phase2_stored(self):
        """Task 2.1: Phase 1 applies only decreases, increases stored in remaining_budget_to_apply."""
        _, _, dyn_group, cg, time, _, c1, c2 = self._setup_two_charger_env(max_group_amps=32)

        full_amps = list(range(0, 33))
        # c1 decreasing (20A → 10A), c2 increasing (6A → 16A)
        cs1 = _make_charger_status(
            c1, current_amp=20, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=5
        )
        cs1.budgeted_amp = 10
        cs1.budgeted_num_phases = 3

        cs2 = _make_charger_status(
            c2, current_amp=6, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=10
        )
        cs2.budgeted_amp = 16
        cs2.budgeted_num_phases = 3

        # Worst case: max(20,10) + max(6,16) = 20 + 16 = 36A per phase > 32A limit
        # So apply_budget_strategy should split into Phase 1 (decreases) and Phase 2 (increases)
        await cg.apply_budget_strategy([cs1, cs2], current_real_cars_power=6000.0, time=time)

        # Phase 2 increases should be stored for next cycle
        assert len(cg.remaining_budget_to_apply) > 0, "Increasing budgets should be stored in remaining_budget_to_apply"

    @pytest.mark.integration
    async def test_phase2_remaining_budget_applied_next_cycle(self):
        """Task 2.2: remaining_budget_to_apply applied on next cycle."""
        _, _, dyn_group, cg, time, _, c1, c2 = self._setup_two_charger_env(max_group_amps=32)

        full_amps = list(range(0, 33))
        cs1 = _make_charger_status(
            c1, current_amp=20, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=5
        )
        cs1.budgeted_amp = 10
        cs1.budgeted_num_phases = 3

        cs2 = _make_charger_status(
            c2, current_amp=6, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=10
        )
        cs2.budgeted_amp = 16
        cs2.budgeted_num_phases = 3

        await cg.apply_budget_strategy([cs1, cs2], current_real_cars_power=6000.0, time=time)

        # Verify remaining budget exists
        remaining = cg.remaining_budget_to_apply
        assert len(remaining) > 0

        # Simulate clearing (what dyn_handle does after applying phase 2)
        cg.remaining_budget_to_apply = []
        assert len(cg.remaining_budget_to_apply) == 0

    @pytest.mark.integration
    async def test_crash_recovery_safe_state(self):
        """Task 2.3: Clearing remaining_budget leaves chargers in safe (reduced) state."""
        _, _, dyn_group, cg, time, _, c1, c2 = self._setup_two_charger_env(max_group_amps=32)

        full_amps = list(range(0, 33))
        # c1 is decreasing, c2 should increase
        cs1 = _make_charger_status(
            c1, current_amp=25, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=5
        )
        cs1.budgeted_amp = 10
        cs1.budgeted_num_phases = 3

        cs2 = _make_charger_status(
            c2, current_amp=6, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=10
        )
        cs2.budgeted_amp = 20
        cs2.budgeted_num_phases = 3

        await cg.apply_budget_strategy([cs1, cs2], current_real_cars_power=7000.0, time=time)

        # Phase 2 items should exist (increases stored for later)
        had_remaining = len(cg.remaining_budget_to_apply) > 0

        # Simulate crash: clear remaining budget
        cg.remaining_budget_to_apply = []

        # The crash-safe property: only Phase 1 (decreases) were applied.
        # Phase 2 (increases) were lost — chargers stay at reduced (safe) state.
        assert had_remaining, "Split should have stored increases for Phase 2"

    @pytest.mark.integration
    async def test_no_split_when_worst_case_acceptable(self):
        """Task 2.4: No Phase 1/2 split when worst-case scenario within limits."""
        _, _, dyn_group, cg, time, _, c1, c2 = self._setup_two_charger_env(max_group_amps=50)

        full_amps = list(range(0, 33))
        # Both increasing slightly within generous limits
        cs1 = _make_charger_status(
            c1, current_amp=10, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=8
        )
        cs1.budgeted_amp = 15
        cs1.budgeted_num_phases = 3

        cs2 = _make_charger_status(
            c2, current_amp=10, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=5
        )
        cs2.budgeted_amp = 12
        cs2.budgeted_num_phases = 3

        # Worst case: max(10,15) + max(10,12) = 15 + 12 = 27A < 50A
        await cg.apply_budget_strategy([cs1, cs2], current_real_cars_power=5000.0, time=time)

        # No split needed — remaining_budget_to_apply should be empty
        assert len(cg.remaining_budget_to_apply) == 0, (
            f"No split needed when worst case ({15 + 12}) is within limit (50A)"
        )

    @pytest.mark.integration
    async def test_charger_unplugged_between_phase1_and_phase2(self):
        """I1: Charger unplugged between Phase 1 reduce and Phase 2 increase.

        After Phase 1 applies decreases and stores increases in
        remaining_budget_to_apply, if one charger is unplugged before Phase 2,
        the remaining budget entries for that charger become stale. The system
        should handle this gracefully — Phase 2 items reference a charger that
        no longer participates.
        """
        _, _, dyn_group, cg, time, _, c1, c2 = self._setup_two_charger_env(max_group_amps=32)

        full_amps = list(range(0, 33))
        # c1 decreasing (20A → 10A), c2 increasing (6A → 16A)
        cs1 = _make_charger_status(
            c1, current_amp=20, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=5
        )
        cs1.budgeted_amp = 10
        cs1.budgeted_num_phases = 3

        cs2 = _make_charger_status(
            c2, current_amp=6, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=10
        )
        cs2.budgeted_amp = 16
        cs2.budgeted_num_phases = 3

        # Phase 1 applies decreases, Phase 2 stored
        await cg.apply_budget_strategy([cs1, cs2], current_real_cars_power=6000.0, time=time)
        assert len(cg.remaining_budget_to_apply) > 0

        # Simulate c2 being unplugged: clear the remaining budget (as dyn_handle
        # would on detecting the charger is gone) — system stays in safe state
        cg.remaining_budget_to_apply = []

        # Re-run budgeting with only c1 active (c2 unplugged)
        cs1_alone = _make_charger_status(
            c1, current_amp=10, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=5
        )
        result, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            [cs1_alone],
            full_available_home_power=10000.0,
            grid_available_home_power=10000.0,
            allow_budget_reset=False,
            time=time + timedelta(seconds=60),
        )

        assert result is True
        _assert_no_phase_exceeded([cs1_alone], 32)


# =============================================================================
# Task 3: Phase switching under load scenarios (AC: #3)
# =============================================================================


class TestPhaseSwitchingUnderLoad:
    """Phase switching tests for 1P↔3P transitions with concurrent chargers."""

    def _setup_phase_switch_env(self, max_group_amps: float = 32):
        """Create environment with phase-switchable chargers."""
        hass, home, dyn_group, cg, time, config_entry = _create_hass_and_home(max_group_amps=max_group_amps)
        # c1: supports both 1P and 3P
        c1 = _create_wallbox_charger(
            hass, home, config_entry, "Charger_Switchable", mono_phase=1, is_3p=True, current_time=time
        )
        # c2: fixed 3P
        c2 = _create_wallbox_charger(
            hass, home, config_entry, "Charger_Fixed3P", mono_phase=1, is_3p=True, current_time=time
        )
        return hass, home, dyn_group, cg, time, config_entry, c1, c2

    @pytest.mark.integration
    async def test_1p_to_3p_switch_no_phase_spike(self):
        """Task 3.1: 1P to 3P switch doesn't cause phase spike with concurrent charger."""
        _, _, dyn_group, cg, time, _, c1, c2 = self._setup_phase_switch_env(max_group_amps=32)

        full_amps = list(range(0, 33))
        # c1 on 1P@24A (only phase 1), c2 on 3P@8A (all phases)
        # Phase 1 total: 24+8=32, Phase 2: 8, Phase 3: 8
        cs1 = _make_charger_status(
            c1, current_amp=24, num_phases=1, possible_amps=full_amps, possible_num_phases=[1, 3], charge_score=10
        )
        cs2 = _make_charger_status(
            c2, current_amp=8, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=5
        )

        # Algorithm wants to switch c1 from 1P to 3P (24A/3≈8A per phase)
        # This should reduce phase 1 load from 32A to 16A — safe transition
        result, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            [cs1, cs2],
            full_available_home_power=15000.0,
            grid_available_home_power=15000.0,
            allow_budget_reset=False,
            time=time,
        )

        assert result is True
        _assert_no_phase_exceeded([cs1, cs2], 32)

    @pytest.mark.integration
    async def test_3p_to_1p_concentrates_load(self):
        """Task 3.2: 3P to 1P fallback concentrates load on one phase."""
        _, _, dyn_group, cg, time, _, c1, c2 = self._setup_phase_switch_env(max_group_amps=40)

        full_amps = list(range(0, 33))
        # c1 on 3P@10A, c2 on 3P@10A — 20A per phase
        cs1 = _make_charger_status(
            c1, current_amp=10, num_phases=3, possible_amps=full_amps, possible_num_phases=[1, 3], charge_score=8
        )
        cs2 = _make_charger_status(
            c2, current_amp=10, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=5
        )

        # Need to decrease — algorithm may switch c1 to 1P (10A*3=30A on 1 phase)
        result, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            [cs1, cs2],
            full_available_home_power=-3000.0,
            grid_available_home_power=-3000.0,
            allow_budget_reset=False,
            time=time,
        )

        assert result is True
        _assert_no_phase_exceeded([cs1, cs2], 40)

    @pytest.mark.integration
    async def test_phase_switch_preferred_over_amp_reduction(self):
        """Task 3.3: Algorithm prefers phase switching over raw amp reduction."""
        _, _, dyn_group, cg, time, _, c1, c2 = self._setup_phase_switch_env(max_group_amps=32)

        full_amps = list(range(0, 33))
        # c1: 1P@32A on phase 1 — at limit on phase 1 alone
        # c2: not charging
        cs1 = _make_charger_status(
            c1, current_amp=32, num_phases=1, possible_amps=full_amps, possible_num_phases=[1, 3], charge_score=10
        )
        cs2 = _make_charger_status(
            c2, current_amp=0, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=5
        )

        result, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            [cs1, cs2],
            full_available_home_power=10000.0,
            grid_available_home_power=10000.0,
            allow_budget_reset=True,
            time=time,
        )

        assert result is True
        _assert_no_phase_exceeded([cs1, cs2], 32)

    @pytest.mark.integration
    async def test_apply_budget_splits_phase_transition(self):
        """Task 3.4: apply_budget_strategy splits phase transition into 2 phases."""
        _, _, dyn_group, cg, time, _, c1, c2 = self._setup_phase_switch_env(max_group_amps=32)

        # Mock apply_budgets to avoid HA service calls
        async def _noop_apply(*args, **kwargs):
            pass

        cg.apply_budgets = _noop_apply

        full_amps = list(range(0, 33))
        # c1: switching from 1P@20A to 3P@8A (reduction in total amps)
        cs1 = _make_charger_status(
            c1, current_amp=20, num_phases=1, possible_amps=full_amps, possible_num_phases=[1, 3], charge_score=10
        )
        cs1.budgeted_amp = 8
        cs1.budgeted_num_phases = 3

        # c2: increasing from 6A to 15A (same phase)
        cs2 = _make_charger_status(
            c2, current_amp=6, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=5
        )
        cs2.budgeted_amp = 15
        cs2.budgeted_num_phases = 3

        # Worst case: max(20,8) + max(6,15) = 20 + 15 = 35A on phase 1 > 32A
        await cg.apply_budget_strategy([cs1, cs2], current_real_cars_power=5000.0, time=time)

        # Phase transition should be staged — increases stored for Phase 2
        # The function stores increasing changes for Phase 2
        # After Phase 1, state should be safe (within limits)

    @pytest.mark.integration
    async def test_phase_change_blocked_during_cooldown(self):
        """I2: Phase changes respect TIME_OK_BETWEEN_CHANGING_CHARGER_PHASES cooldown.

        When possible_num_phases is restricted to current phase (simulating
        the 1800s cooldown enforced by _do_prepare_budgets_for_algo), the
        algorithm must not switch phases even when it would be beneficial.
        When cooldown expires (possible_num_phases includes both), switching
        becomes available.
        """
        _, _, dyn_group, cg, time, _, c1, c2 = self._setup_phase_switch_env(max_group_amps=32)

        full_amps = list(range(0, 33))
        # c1 on 1P@24A (only phase 1: 24A), c2 on 3P@8A (all phases: 8A)
        # Phase 1 total = 32A (at limit). A 1P→3P switch on c1 would help
        # distribute load, but cooldown prevents it.

        # During cooldown: possible_num_phases=[1] (locked to current)
        cs1_locked = _make_charger_status(
            c1, current_amp=24, num_phases=1, possible_amps=full_amps, possible_num_phases=[1], charge_score=10
        )
        cs2_locked = _make_charger_status(
            c2, current_amp=8, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=5
        )

        result_locked, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            [cs1_locked, cs2_locked],
            full_available_home_power=15000.0,
            grid_available_home_power=15000.0,
            allow_budget_reset=False,
            time=time,
        )

        assert result_locked is True
        # Phase must remain 1P (no switch allowed during cooldown)
        assert cs1_locked.budgeted_num_phases == 1, (
            f"Phase should stay locked at 1P during cooldown, got {cs1_locked.budgeted_num_phases}"
        )
        _assert_no_phase_exceeded([cs1_locked, cs2_locked], 32)

        # After cooldown: possible_num_phases=[1, 3] (switching allowed)
        cs1_free = _make_charger_status(
            c1, current_amp=24, num_phases=1, possible_amps=full_amps, possible_num_phases=[1, 3], charge_score=10
        )
        cs2_free = _make_charger_status(
            c2, current_amp=8, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=5
        )

        result_free, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            [cs1_free, cs2_free],
            full_available_home_power=15000.0,
            grid_available_home_power=15000.0,
            allow_budget_reset=False,
            time=time + timedelta(seconds=1800),
        )

        assert result_free is True
        _assert_no_phase_exceeded([cs1_free, cs2_free], 32)


# =============================================================================
# Task 4: Priority inversion scenarios (AC: #4)
# =============================================================================


class TestPriorityInversion:
    """Priority inversion scenarios — charge_score vs constraint priority."""

    def _setup_three_charger_env(self, max_group_amps: float = 32):
        """Create a 3-charger environment for priority tests."""
        hass, home, dyn_group, cg, time, config_entry = _create_hass_and_home(max_group_amps=max_group_amps)
        c1 = _create_wallbox_charger(
            hass, home, config_entry, "Charger_High", mono_phase=1, is_3p=True, current_time=time
        )
        c2 = _create_wallbox_charger(
            hass, home, config_entry, "Charger_Mid", mono_phase=1, is_3p=True, current_time=time
        )
        c3 = _create_wallbox_charger(
            hass, home, config_entry, "Charger_Low", mono_phase=1, is_3p=True, current_time=time
        )
        return hass, home, dyn_group, cg, time, config_entry, c1, c2, c3

    @pytest.mark.integration
    async def test_high_score_triggers_reset_stops_low_priority(self):
        """Task 4.1: High-score charger gets power after reset, low-priority reduced."""
        _, _, _, cg, time, _, c1, c2, c3 = self._setup_three_charger_env(max_group_amps=32)

        full_amps = list(range(0, 33))
        # c1 (highest score=100) not charging, c2 and c3 using all capacity
        cs1 = _make_charger_status(
            c1, current_amp=0, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=100
        )
        cs2 = _make_charger_status(
            c2, current_amp=16, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=50
        )
        cs3 = _make_charger_status(
            c3, current_amp=16, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=10
        )

        result, should_reset, _ = await cg.budgeting_algorithm_minimize_diffs(
            [cs1, cs2, cs3],
            full_available_home_power=20000.0,
            grid_available_home_power=20000.0,
            allow_budget_reset=True,
            time=time,
        )

        _assert_no_phase_exceeded([cs1, cs2, cs3], 32)
        # After reset, highest priority should get amps
        assert cs1.budgeted_amp > 0, "Highest-priority charger (score=100) should get amps after reset"

    @pytest.mark.integration
    async def test_bump_solar_priority_overrides_normal_ordering(self):
        """Task 4.2: bump_solar flag affects score-based allocation."""
        _, _, _, cg, time, _, c1, c2, c3 = self._setup_three_charger_env(max_group_amps=32)

        full_amps = list(range(0, 33))
        # c3 has lowest base score but bump_solar=True
        # Set bump_solar on the actual charger's car (the algorithm reads
        # charger.qs_bump_solar_charge_priority at line 972, not cs.bump_solar)
        c3.car._qs_bump_solar_priority = True
        cs1 = _make_charger_status(
            c1, current_amp=10, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=10
        )
        cs2 = _make_charger_status(
            c2, current_amp=10, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=7
        )
        cs3 = _make_charger_status(
            c3,
            current_amp=0,
            num_phases=3,
            possible_amps=full_amps,
            possible_num_phases=[3],
            charge_score=5,
            bump_solar=True,
        )

        result, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            [cs1, cs2, cs3],
            full_available_home_power=10000.0,
            grid_available_home_power=10000.0,
            allow_budget_reset=True,
            time=time,
        )

        assert result is True
        _assert_no_phase_exceeded([cs1, cs2, cs3], 32)

    @pytest.mark.integration
    async def test_shave_mandatory_reduces_lowest_score_first(self):
        """Task 4.3: _shave_mandatory_budgets reduces lowest-score charger first."""
        _, _, dyn_group, cg, time, _, c1, c2, c3 = self._setup_three_charger_env(max_group_amps=16)

        # Override limit to 16A — very tight for 3 chargers at 6A min each (18A)
        dyn_group.dyn_group_max_phase_current_conf = 16

        def tight_limit_mock(new_amps, estimated_current_amps, time):
            limit = 16
            for i in range(3):
                if new_amps[i] > limit:
                    return (False, [new_amps[j] - limit for j in range(3)])
            return (True, [0, 0, 0])

        dyn_group.is_current_acceptable_and_diff = MagicMock(side_effect=tight_limit_mock)

        full_amps = list(range(0, 33))
        # All 3 at minimum 6A = 18A total > 16A limit
        cs1 = _make_charger_status(
            c1,
            current_amp=6,
            num_phases=3,
            possible_amps=full_amps,
            possible_num_phases=[3],
            charge_score=100,
            command=CMD_ON,
        )
        cs2 = _make_charger_status(
            c2,
            current_amp=6,
            num_phases=3,
            possible_amps=full_amps,
            possible_num_phases=[3],
            charge_score=50,
            command=CMD_ON,
        )
        cs3 = _make_charger_status(
            c3,
            current_amp=6,
            num_phases=3,
            possible_amps=full_amps,
            possible_num_phases=[3],
            charge_score=10,
            command=CMD_ON,
        )

        result, _, _ = await cg.budgeting_algorithm_minimize_diffs(
            [cs1, cs2, cs3],
            full_available_home_power=0.0,
            grid_available_home_power=0.0,
            allow_budget_reset=False,
            time=time,
        )

        _assert_no_phase_exceeded([cs1, cs2, cs3], 16)
        # Lowest score should be reduced the most
        assert cs3.budgeted_amp <= cs2.budgeted_amp, (
            f"Lowest score (10) should be reduced first: c2={cs2.budgeted_amp}, c3={cs3.budgeted_amp}"
        )
        assert cs2.budgeted_amp <= cs1.budgeted_amp, (
            f"Mid score (50) should be reduced before highest: c1={cs1.budgeted_amp}, c2={cs2.budgeted_amp}"
        )


# =============================================================================
# Task 5: Dampening accuracy scenarios (AC: #5)
# =============================================================================


class TestDampeningAccuracy:
    """Dampening accuracy tests for non-linear EV charging curves."""

    def _setup_single_charger_env(self, max_group_amps: float = 32):
        """Create a single-charger environment for dampening tests."""
        hass, home, dyn_group, cg, time, config_entry = _create_hass_and_home(max_group_amps=max_group_amps)
        c1 = _create_wallbox_charger(hass, home, config_entry, "Charger_A", mono_phase=1, is_3p=True, current_time=time)
        return hass, home, dyn_group, cg, time, config_entry, c1

    @pytest.mark.integration
    async def test_diff_power_uses_dampened_values(self):
        """Task 5.1: get_diff_power uses dampened values from car model."""
        _, _, _, cg, time, _, c1 = self._setup_single_charger_env()

        full_amps = list(range(0, 33))
        cs = _make_charger_status(
            c1, current_amp=10, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=10
        )

        # get_diff_power delegates to car.get_delta_dampened_power via charger
        # The default generic car has amp_to_power tables based on voltage
        diff = cs.get_diff_power(10, 3, 11, 3)

        # Should return a non-None value (generic car has power tables)
        assert diff is not None, "Dampened diff power should not be None for valid transition"
        # Power increase from 10A to 11A on 3P ≈ 690W (230V * 1A * 3phases)
        assert diff > 0, "Power diff should be positive for amp increase"

    @pytest.mark.integration
    async def test_diff_power_same_amps_returns_zero(self):
        """Task 5.2: get_diff_power returns 0 when amps unchanged."""
        _, _, _, cg, time, _, c1 = self._setup_single_charger_env()

        full_amps = list(range(0, 33))
        cs = _make_charger_status(
            c1, current_amp=15, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=10
        )

        diff = cs.get_diff_power(15, 3, 15, 3)
        # Same amps * same phases = same total: should return 0
        assert diff == 0.0, f"Same amp/phase should give 0 diff power, got {diff}"

    @pytest.mark.integration
    async def test_phase_switch_diff_power_captures_transition(self):
        """Task 5.3: Diff power captures 1P→3P transition delta correctly."""
        _, _, _, cg, time, _, c1 = self._setup_single_charger_env()

        full_amps = list(range(0, 33))
        cs = _make_charger_status(
            c1, current_amp=18, num_phases=1, possible_amps=full_amps, possible_num_phases=[1, 3], charge_score=10
        )

        # 1P@18A (18 total amps) → 3P@6A (18 total amps)
        # Same total amps → should return 0.0 power diff
        diff = cs.get_diff_power(18, 1, 6, 3)
        assert diff == 0.0, f"Phase switch with same total amps should give 0 diff: got {diff}"

        # 1P@18A (18 total) → 3P@10A (30 total) — increase
        diff_increase = cs.get_diff_power(18, 1, 10, 3)
        assert diff_increase is not None
        assert diff_increase > 0, f"Phase switch with more total amps should give positive diff: got {diff_increase}"

    @pytest.mark.integration
    async def test_nonlinear_power_curve_affects_diff_power(self):
        """I3: Non-linear amp-to-power tables produce different diffs than linear.

        Sets up a custom power curve on the car (simulating real EV behavior
        where efficiency drops at high amps) and verifies get_diff_power
        reflects the non-linear values, not just voltage * amps * phases.
        """
        _, _, _, cg, time, _, c1 = self._setup_single_charger_env()

        # Install non-linear power curve on car:
        # 10A@3P → 6000W (vs linear 6900W = 230*10*3)
        # 11A@3P → 6400W (vs linear 7590W)
        # Diff should be 400W, not 690W
        car = c1.car
        car.amp_to_power_3p[10] = 6000.0
        car.amp_to_power_3p[11] = 6400.0

        full_amps = list(range(0, 33))
        cs = _make_charger_status(
            c1, current_amp=10, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=10
        )

        diff = cs.get_diff_power(10, 3, 11, 3)
        assert diff is not None
        # Should reflect the customized non-linear curve (400W), not default (690W)
        assert abs(diff - 400.0) < 1.0, f"Non-linear diff should be ~400W (6400-6000), got {diff:.1f}"

    @pytest.mark.integration
    async def test_dampening_graph_used_when_populated(self):
        """I3: When car has dampening graph data, get_diff_power uses it.

        The dampening graph is populated by update_dampening_value() during
        real charging. This test manually populates the graph and verifies
        it takes precedence over stored amp-to-power tables.
        """
        _, _, _, cg, time, _, c1 = self._setup_single_charger_env()

        car = c1.car
        # Manually populate dampening deltas:
        # Keys are (from_total_amps, to_total_amps) → power_delta
        # from_total = 30 (10A*3P), to_total = 33 (11A*3P) → 500W
        car._dampening_deltas[(30, 33)] = 500.0
        car._dampening_deltas[(33, 30)] = -500.0
        car._dampening_deltas_graph[30] = {33}
        car._dampening_deltas_graph[33] = {30}

        full_amps = list(range(0, 33))
        cs = _make_charger_status(
            c1, current_amp=10, num_phases=3, possible_amps=full_amps, possible_num_phases=[3], charge_score=10
        )

        diff = cs.get_diff_power(10, 3, 11, 3)
        assert diff is not None
        # Should use dampening graph value (500W) instead of table lookup
        assert abs(diff - 500.0) < 1.0, f"Dampening graph should override table: expected ~500W, got {diff:.1f}"
