"""Shared helpers for writing domain values to third-party `number` entities.

Extracted from `ha_model/battery.py` (QS-349 review-fix #10 AA2). The problem
class — "write a domain value to a `number` entity you do not control, and know
what actually landed" — is generic, not battery-specific: unit conversion,
domain clamp, direction-aware step snap, landed-value prediction, tolerant
read-back comparison, and a bounded once-per-divergence warning latch.

Generic over the value domain: the default converter treats the entity as power
(W); a percent (or other) consumer can inject a different `UnitConverter`. The
divergence latch is per-instance, so each consumer device owns its own.

Layering: this is `ha_model` (may import `home_model`) — it imports
`coerce_finite_float` from `home_model.battery`. It is pure over its injected
``hass`` and carries NO battery-specific semantics (the "is this the safety
floor / a max?" decision stays in the caller).

Candidate follow-up (out of scope here): migrate the other two hand-rolled
partial implementations — `ha_model/charger.py` `low_level_set_max_charging_current`
and `ha_model/car.py` (bisect snap-up to allowed steps + read-back compare) —
onto this utility.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from homeassistant.const import (
    ATTR_UNIT_OF_MEASUREMENT,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    UnitOfPower,
)
from homeassistant.util.unit_conversion import PowerConverter

from ..home_model.battery import coerce_finite_float

_LOGGER = logging.getLogger(__name__)


if TYPE_CHECKING:
    from typing import Protocol

    class UnitConverter(Protocol):
        """Convert between a domain base unit and a number entity's advertised unit.

        The default (`POWER_UNIT_CONVERTER`) treats the base unit as W and
        converts to/from any `UnitOfPower` member; a non-convertible unit
        (percent, a corrupt non-string attribute) is an identity pass-through.
        A consumer in a different domain injects its own matching object
        (structural typing — no subclassing required).
        """

        def convertible(self, unit) -> bool: ...

        def to_entity(self, value_base: float, unit) -> float: ...

        def to_base(self, value_entity: float, unit) -> float: ...


class _PowerUnitConverter:
    """W <-> entity-unit power conversion (the battery / default path)."""

    def convertible(self, unit) -> bool:
        # Y6: a present-but-None (or non-str) unit attribute is corrupt — treat as base
        return isinstance(unit, str) and unit in UnitOfPower and unit != UnitOfPower.WATT

    def to_entity(self, value_base: float, unit) -> float:
        return PowerConverter.convert(value=value_base, from_unit=UnitOfPower.WATT, to_unit=unit)

    def to_base(self, value_entity: float, unit) -> float:
        return PowerConverter.convert(value=value_entity, from_unit=unit, to_unit=UnitOfPower.WATT)


POWER_UNIT_CONVERTER = _PowerUnitConverter()


class NumberEntityTargeter:
    """Map a domain value to a `number` entity's `set_value`, and read it back.

    One instance per consumer device (it owns the divergence latch). All the
    battery-independent logic that used to live on `QSBattery` moved here
    verbatim (review-fix #10 AA2); the behaviour — U1/U2 write policy, V1 probe
    tolerance, and every X/Y/Z guard from rounds 6-9 — is unchanged.
    """

    # Hard cap on the divergence-warned latch. A permanently pinned entity under
    # solver-varying consigns mints one entry per distinct request; evict the
    # oldest at the cap instead of growing forever (X4).
    LATCH_MAX = 64

    def __init__(self, hass, *, latch_max: int | None = None, unit_converter: UnitConverter | None = None) -> None:
        self._hass = hass
        self._latch_max = self.LATCH_MAX if latch_max is None else latch_max
        self._conv = unit_converter or POWER_UNIT_CONVERTER
        # dedupe for the divergence warning (U5/W3); (entity, landed, above,
        # request) keys. Insertion-ordered dict used as a bounded set.
        self._divergence_warned: dict[tuple[str | None, int, bool, int], None] = {}

    def target(
        self,
        entity_id: str | None,
        value_base: float,
        *,
        snap_up: bool,
        domain_min: float,
        domain_max: float,
        warn: bool = True,
    ) -> tuple[float, int]:
        """Map a base-unit target to (value_to_write, expected_landed_base) for an entity.

        The **same** helper backs both the write and the probe expectation, so
        they can never disagree (eternal retry). Steps:

        1. domain clamp to ``[domain_min, domain_max]`` (shared by write and
           probe — T2);
        2. convert to the entity's unit and clamp to its ``min`` / ``max``;
        3. snap to the entity ``step`` — UP for a safety floor (``snap_up``),
           DOWN for a maximum;
        4. **snap-policy priority** (U1/U2): the configured max (``domain_max``)
           wins downward and the safety floor (``domain_min``) wins upward — a
           non-step-aligned write at either domain bound is accepted (HA core
           validates min/max, not step alignment). The entity's own
           ``min`` / ``max`` remain the hard bounds HA would reject outside of.

        Inconsistent (``min > max``) or non-numeric entity attributes are treated
        as absent (T7 / U6). A landed value diverging from the request by more
        than one step is warned once per ``(entity, landed, direction, request)``
        (U3/U5/T8/V2).
        """
        request_base = min(float(domain_max), max(float(domain_min), float(value_base)))
        write_value = request_base
        attributes: dict = {}
        if entity_id is not None:
            state = self._hass.states.get(entity_id)
            if state is not None and state.state not in (STATE_UNKNOWN, STATE_UNAVAILABLE):
                attributes = state.attributes or {}

        unit = attributes.get(ATTR_UNIT_OF_MEASUREMENT, UnitOfPower.WATT)
        to_entity_unit = self._conv.convertible(unit)

        def _to_unit(value_b: float) -> float:
            return self._conv.to_entity(value_b, unit) if to_entity_unit else value_b

        def _to_base(value_u: float) -> float:
            return self._conv.to_base(value_u, unit) if to_entity_unit else value_u

        write_value = _to_unit(write_value)
        dmin_u = _to_unit(float(domain_min))
        dmax_u = _to_unit(float(domain_max))

        ent_min = coerce_finite_float(attributes.get("min"), None)
        ent_max = coerce_finite_float(attributes.get("max"), None)
        # Y1/Y2: the shared corrupt-step rule (tiny/huge/non-finite => absent)
        # governs the snap AND the warn tolerance below — see entity_step.
        step_u, step_base = self.entity_step(entity_id, float(domain_max))
        # U6: mutually inconsistent bounds are as unusable as non-numeric ones
        if ent_min is not None and ent_max is not None and ent_min > ent_max:
            ent_min = ent_max = None

        if ent_min is not None:
            write_value = max(write_value, ent_min)
        if ent_max is not None:
            write_value = min(write_value, ent_max)

        if step_u > 0.0:
            ratio = write_value / step_u
            # Y2: a corrupt entity bound can inflate the value beyond the step
            # scale, overflowing the ratio — skip the snap (step absent) instead
            # of raising OverflowError in ceil/floor.
            if math.isfinite(ratio):
                if snap_up:
                    # a safety minimum: never below the request. FP-safe so an
                    # exact step multiple does not overshoot a whole step.
                    write_value = math.ceil(ratio - 1e-9) * step_u
                else:
                    # a maximum: never above the request.
                    write_value = math.floor(ratio + 1e-9) * step_u

        # snap-policy priority (U1/U2): the configured max wins down, the safety
        # floor wins up — accepting a non-step-aligned value at either bound.
        write_value = min(write_value, dmax_u)
        write_value = max(write_value, dmin_u)
        # the entity's own range is the hard bound HA validates (U7: a floor above
        # the entity max lands at the raw entity max, never zeroed by step math).
        if ent_max is not None:
            write_value = min(write_value, ent_max)
        if ent_min is not None:
            write_value = max(write_value, ent_min)

        # Z1: a corrupt-but-finite entity bound (e.g. a 1e306 kW min) survives
        # coerce_finite_float, wins the hard re-clamp above, and overflows the
        # unit conversion to inf — int(round(inf)) would raise OverflowError.
        # Treat the corrupt bound as absent for the landed EXPECTATION and fall
        # back to the domain-clamped request (the write keeps the entity's hard
        # bound — HA validates min/max).
        landed_to_base = _to_base(write_value)
        landed = int(round(landed_to_base)) if math.isfinite(landed_to_base) else int(round(request_base))

        # V2/W2: direction-aware warning tolerance. The step snap can only
        # legitimately move the landed value in ONE direction (UP on ceil, DOWN
        # on floor) and always strictly LESS than one step — so in the snap
        # direction a divergence reaching a full step (`>= step_base`) is an
        # external bound (e.g. the entity min), not quantization: warn. The
        # OPPOSITE direction means an external cap is binding — surface even a
        # small one there (~1). Distinct from the probe's device-echo tolerance
        # (V1). The step is the sanity-checked one (Y1).
        if warn:
            snap_step = step_base
            delta_above = landed - request_base
            delta_below = request_base - landed

            def _snap_dir_diverges(delta: float) -> bool:
                return delta >= snap_step if snap_step > 0.0 else delta > 1.0

            above_diverges = _snap_dir_diverges(delta_above) if snap_up else delta_above > 1.0
            below_diverges = delta_below > 1.0 if snap_up else _snap_dir_diverges(delta_below)
            if above_diverges:
                self._warn_divergence(entity_id, landed, request_base, above=True)
            elif below_diverges:
                self._warn_divergence(entity_id, landed, request_base, above=False)

        return write_value, landed

    def reading_matches(
        self, entity_id: str | None, read_value, expected_value, write_value: float | None, domain_max: float
    ) -> bool:
        """Step-aware comparison shared by the probe AND the write-skip checks (W1).

        ``domain_max`` is the calling LEG's configured domain max, used as the
        step sanity bound (X5) — passed by the caller because the same entity may
        back both legs, so it cannot be derived from the entity id (Z3).

        A backing integration may quantize a non-step-aligned domain-bound write
        (the accepted U1/U2 policy) to its advertised step and echo a
        step-neighbour — which is always strictly LESS than one step from the
        expected landed value, so accept strictly ``< step`` there. A
        step-ALIGNED write can only be echoed exactly: any non-equal reading is
        provably stale/foreign (a swallowed restore or floor write must NOT be
        confirmed — it would silence the retry loop). No advertised step (or no
        known written value) requires exact equality (V1). A ZERO reading never
        confirms a non-zero landed value even inside the step window — that would
        silently zero a safety floor or cap (U7). This device-echo tolerance is
        distinct from the divergence-warning tolerance.
        """
        if read_value is None or expected_value is None:
            return read_value == expected_value
        if read_value == expected_value:
            return True
        # U7/W1: a zero reading is never a tolerable quantization echo of a
        # NON-zero landed value — even when it is a step-neighbour (step > entity
        # max), accepting it would silently zero a safety floor (or cap) and
        # skip/confirm away the retry loop. Always rewrite instead.
        if read_value == 0:
            return False
        step_u, step_base = self.entity_step(entity_id, domain_max)
        if step_base <= 0.0 or write_value is None:
            return False
        # X6: fp-tolerant alignment check judged in the VALUE domain — an absolute
        # epsilon on the ratio misreads large value / tiny step pairs as
        # non-aligned, silently loosening the probe to the step window.
        ratio = write_value / step_u
        # Y2: a corrupt entity bound can inflate the write far beyond the step
        # scale, overflowing the ratio — treat the step as absent (exact match
        # already failed above) instead of raising OverflowError in round().
        if not math.isfinite(ratio):
            return False
        if math.isclose(write_value, round(ratio) * step_u, rel_tol=1e-9, abs_tol=1e-6):
            # step-aligned write: a quantizing device echoes it exactly
            return False
        return abs(read_value - expected_value) < step_base

    def entity_step(self, entity_id: str | None, domain_max: float) -> tuple[float, float]:
        """The entity's advertised step as (entity-unit, base-unit); (0.0, 0.0) if none/unreadable.

        One corrupt-step rule shared by the snap, echo-match and warn paths
        (Y1/Y2): a step is treated as ABSENT when its base conversion is
        non-finite (overflowed unit conversion), denormal-tiny (< 1e-6 —
        ``value / step`` would overflow and crash ceil/round), or wider than
        ``domain_max`` — the calling LEG's configured domain max (X5 — it would
        widen tolerance windows arbitrarily, or on the warn path suppress every
        divergence). The bound is passed by the caller, which knows the leg: the
        same entity may back both legs, so it cannot be derived from the entity
        id (Z3). Y3: a side effect kept on purpose — with the step then treated
        as absent, a device echo ABOVE the configured hardware max can never be
        confirmed (exact match against a landed value <= the max always fails).
        """
        if entity_id is None:
            return 0.0, 0.0
        state = self._hass.states.get(entity_id)
        if state is None or state.state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            return 0.0, 0.0
        attributes = state.attributes or {}
        step = coerce_finite_float(attributes.get("step"), None)
        if step is None or step <= 0.0:
            return 0.0, 0.0
        step_base = step
        unit = attributes.get(ATTR_UNIT_OF_MEASUREMENT, UnitOfPower.WATT)
        if self._conv.convertible(unit):
            step_base = self._conv.to_base(step, unit)
        if not math.isfinite(step_base) or step_base < 1e-6 or step_base > domain_max:
            return 0.0, 0.0
        return step, step_base

    def clear_latch(self, entity_id: str | None, confirmed: int | None) -> None:
        """Drop this entity's RESOLVED divergence latch entries (V5/W3).

        ``confirmed`` is the EXPECTED landed value the probe just confirmed (X3)
        — the latch keys the landed value, and an echo-confirm may read a
        step-neighbour of it. Latched divergences that landed elsewhere are
        resolved (drop them so a recurrence warns again), while an entry whose
        landed value IS the confirmed one is still current — keep it latched so a
        permanently diverging entity warns once, not once per command/probe
        cycle. Never touch other entities' entries.
        """
        if confirmed is None:
            # Z2: no landed value was confirmed for this entity — clearing would
            # wipe its whole latch (no int key matches None) and re-arm warnings
            # for still-current divergences.
            return
        self._divergence_warned = {
            key: None for key in self._divergence_warned if key[0] != entity_id or key[1] == confirmed
        }

    def _warn_divergence(self, entity_id: str | None, landed: int, request: float, above: bool) -> None:
        """Warn once per distinct (entity, landed, direction, request) divergence (U5/W3).

        The latch is bounded (X4): at the cap the oldest entry is evicted, so a
        pinned entity under ever-varying consigns cannot grow it forever.
        """
        key = (entity_id, landed, above, int(round(request)))
        if key in self._divergence_warned:
            return
        while len(self._divergence_warned) >= self._latch_max:
            self._divergence_warned.pop(next(iter(self._divergence_warned)))
        self._divergence_warned[key] = None
        # message kept verbatim from the battery original (review-fix #10 AA2 is a
        # pure refactor — no behaviour/log-text change); the "W" label is the
        # default power domain's unit.
        _LOGGER.warning(
            "number entity %s lands %s W, %s the requested %s W",
            entity_id,
            landed,
            "above" if above else "below",
            int(round(request)),
        )
