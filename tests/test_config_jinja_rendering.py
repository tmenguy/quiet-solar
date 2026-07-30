"""Tests for the Legrand/Netatmo facade switches in ``config/configuration.jinja``.

The design these tests pin is stated in ``docs/stories/QS-305.story_review_fix_#01.md``
Part 1, which supersedes the original story. The rule everything derives from:

    **Never trust a value you just wrote. Never report a value you can't trust.**

We write ``input_boolean.<slug>_command`` and we write the Netatmo cloud switch, so neither
can be evidence -- they only parrot back what we told them. The Netatmo switch in particular
sets ``_attr_is_on`` optimistically and writes state immediately
(``custom_components/netatmo/switch.py:77-90``, a **fork** that shadows the core integration),
so reading it back after our own write is guaranteed to return the target value. The mirror,
``input_boolean.<slug>_state``, is written only by Apple Home from watching the real device,
and is therefore the only honest witness.

These tests catch bugs where:

- A wait or notification condition consults the cloud leg, making the wait vacuously true and
  the notification branch unreachable (the shipped-then-reverted defect: an un-actuated
  ``turn_on`` was never reported).
- A wait is expressed as "the mirror is not ``on``" instead of positively confirming the
  target, so an ``unknown``/``unavailable`` mirror silently satisfies a turn-off.
- ``state:`` or ``availability:`` reports a confident value while the mirror cannot be
  trusted -- an entity confidently reporting ``off`` while a boiler ran is the whole incident.
- A predicate or ``enabled: false`` can prevent either actuation write from executing.

**Permanent gate note.** ``config/configuration.jinja`` is a non-Python file, so testmon
cannot select this test: ``--impacted`` is blind to a template-only change. Run
``python scripts/qs/quality_gate.py --quick tests/test_config_jinja_rendering.py``
in any PR that touches the template. CI's whole-suite run is the durable net.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jinja2
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"

# Exercises both generator-side filters: only the first entry passes
# `startswith('sensor.')` and `endswith('_power')`.
FIXTURE_NETATMO_ENTITIES = [
    "sensor.cumulus_pool_house_power",  # passes both filters
    "sensor.not_a_power_entity",  # fails endswith('_power')
    "switch.cumulus_pool_house",  # fails startswith('sensor.')
]

LINK_SENSOR = "binary_sensor.legrand_homekit_link_healthy"
HUB_SENSOR = "binary_sensor.homekit_hub_healthy"
HEALTH_SENSORS = (LINK_SENSOR, HUB_SENSOR)

DIRECTIONS = ("turn_on", "turn_off")

# The value each direction positively confirms on the mirror.
TARGET_OF = {"turn_on": "on", "turn_off": "off"}


@dataclass(frozen=True)
class Row:
    """One scenario for the emitted ``state:`` and ``availability:`` templates."""

    mirror: str
    link: str
    hub: str
    state: str
    available: str
    why: str


# Rows are NAMED, not numbered, so the state and availability expectations cannot drift.
#
# The trust predicate is `hk_ok and mirror in ['on','off']`. `R_link_down` and `R_hub_down`
# vary the two health sensors independently, which pins `hk_ok` as an AND -- an OR
# implementation would pass if only one of them were ever exercised.
ROWS: dict[str, Row] = {
    "R_mirror_on_trusted": Row(
        "on", "on", "on", "True", "True", "the normal running state"
    ),
    "R_mirror_off_trusted": Row(
        "off", "on", "on", "False", "True", "the only case that may report a confident off"
    ),
    "R_mirror_unknown": Row(
        "unknown", "on", "on", "unknown", "False", "mirror not yet restored -> go dark"
    ),
    "R_mirror_unavailable": Row(
        "unavailable", "on", "on", "unknown", "False", "mirror invalid -> go dark"
    ),
    "R_link_down": Row(
        "on", "off", "on", "unknown", "False", "pins hk_ok as AND on the link sensor"
    ),
    "R_hub_down": Row(
        "on", "on", "off", "unknown", "False", "pins hk_ok as AND on the hub sensor"
    ),
    "R_health_down_mirror_off": Row(
        "off",
        "off",
        "off",
        "unknown",
        "False",
        "THE incident class: never report a confident off when we cannot see",
    ),
}


# ============================================================================
# Layer 1: render the generator template
# ============================================================================


def render_config() -> str:
    """Render ``config/configuration.jinja`` with the generator-layer globals stubbed."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(CONFIG_DIR),
        undefined=jinja2.StrictUndefined,
    )
    env.globals["integration_entities"] = lambda domain: FIXTURE_NETATMO_ENTITIES
    return env.get_template("configuration.jinja").render()


# ============================================================================
# Layer 2: parse the emitted YAML
# ============================================================================


class _TagTolerantLoader(yaml.SafeLoader):
    """SafeLoader that degrades unknown ``!tag`` nodes to plain strings.

    The rendered config uses ``!include``, which ``yaml.SafeLoader`` rejects. The
    multi-constructor is registered on this subclass rather than on ``yaml.SafeLoader``
    so every other test's ``yaml.safe_load`` stays strict.
    """


_TagTolerantLoader.add_multi_constructor(
    "!", lambda loader, suffix, node: f"!{suffix}"
)


def parse_config(rendered: str) -> dict[str, Any]:
    """Parse a rendered config with the tag-tolerant loader."""
    return yaml.load(rendered, Loader=_TagTolerantLoader)


def facade_blocks(parsed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the generated Legrand facade switch blocks, keyed by device slug."""
    blocks: dict[str, dict[str, Any]] = {}
    for item in parsed["template"]:
        for blk in item.get("switch", []):
            unique_id = blk.get("unique_id", "")
            if unique_id.endswith("_facade"):
                blocks[unique_id.removesuffix("_facade")] = blk
    assert blocks, "no facade blocks found; every other assertion would be vacuous"
    return blocks


def cloud_entity_of(blk: dict[str, Any]) -> str:
    """Return the Netatmo cloud entity id a facade block actuates.

    Sourced from the actuation step's target, because the cloud entity is deliberately
    absent from ``availability:`` (it may appear only as an actuation target and as
    diagnostic text in a notification message).
    """
    entity_id = blk["turn_on"][1]["target"]["entity_id"]
    assert entity_id.startswith("switch."), entity_id
    return entity_id


def wait_of(blk: dict[str, Any], direction: str) -> str:
    """Return the ``wait_template`` of a facade block's direction script."""
    return blk[direction][2]["wait_template"]


def complain_of(blk: dict[str, Any], direction: str) -> str:
    """Return the diagnostic ``choose`` condition of a direction script."""
    return blk[direction][3]["choose"][0]["conditions"]


def notification_of(blk: dict[str, Any], direction: str) -> dict[str, Any]:
    """Return the notification ``data`` mapping of a direction script."""
    return blk[direction][3]["choose"][0]["sequence"][0]["data"]


def all_notification_data(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the ``data`` of every ``persistent_notification.create`` in the config."""
    found: list[dict[str, Any]] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("action") == "persistent_notification.create":
                found.append(node["data"])
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for value in node:
                _walk(value)

    _walk(parsed)
    return found


# ============================================================================
# Layer 3: evaluate the emitted HA-runtime templates
# ============================================================================


def _negate(rendered: str) -> str:
    """Flip a rendered boolean string. Total over the two values it is used with."""
    return {"True": "False", "False": "True"}[rendered]


def eval_ha_template(text: str, states: dict[str, str]) -> str:
    """Render an emitted HA-runtime template with stubbed HA globals.

    Whitespace is collapsed rather than merely stripped, because ``state:`` is a ``>``
    folded scalar whose more-indented body lines keep their newlines.
    """

    def _state_of(entity_id: str) -> str:
        if entity_id not in states:
            raise KeyError(f"template read unstubbed entity {entity_id!r}")
        return states[entity_id]

    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    env.globals["states"] = _state_of
    env.globals["is_state"] = lambda entity_id, value: _state_of(entity_id) == value
    return " ".join(env.from_string(text).render().split())


def trusted_states(slug: str, row: Row) -> dict[str, str]:
    """Build the ``states`` dict for one row.

    The cloud entity is **deliberately absent**. Every template under test must consult
    only the mirror and the two health sensors, so if a future edit reintroduces the cloud
    leg into ``state:``, ``availability:``, a wait or a notification condition, the raising
    accessor turns it into a loud ``KeyError`` rather than a silent pass.
    """
    return {
        f"input_boolean.{slug}_state": row.mirror,
        LINK_SENSOR: row.link,
        HUB_SENSOR: row.hub,
    }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def rendered() -> str:
    """The rendered ``config/configuration.jinja`` output."""
    return render_config()


@pytest.fixture(scope="module")
def parsed(rendered: str) -> dict[str, Any]:
    """The parsed rendered config."""
    return parse_config(rendered)


@pytest.fixture(scope="module")
def blocks(parsed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The generated Legrand facade switch blocks, keyed by device slug."""
    return facade_blocks(parsed)


# ============================================================================
# Render, parse, inventory
# ============================================================================


def test_config_jinja_renders_and_parses(parsed: dict[str, Any]) -> None:
    """The template renders under the stubbed environment and parses as YAML."""
    assert isinstance(parsed, dict)
    assert "template" in parsed
    assert "input_boolean" in parsed


def test_facade_block_inventory(
    parsed: dict[str, Any], blocks: dict[str, dict[str, Any]]
) -> None:
    """Every mapped Legrand device yields exactly one cloud-actuating facade block.

    This is what keeps every other assertion non-vacuous: it pins the set they quantify
    over. Do not prune it.
    """
    expected_slugs = {
        key.removesuffix("_command")
        for key in parsed["input_boolean"]
        if key.endswith("_command")
    }
    assert expected_slugs, "no command booleans found; the template changed shape"
    assert set(blocks) == expected_slugs

    for slug, blk in blocks.items():
        assert cloud_entity_of(blk).startswith("switch."), slug


# ============================================================================
# Reporting: state and availability (Part 1 section 1.5)
# ============================================================================


@pytest.mark.parametrize("row_name", list(ROWS))
def test_facade_reports_only_what_the_mirror_can_prove(
    blocks: dict[str, dict[str, Any]], row_name: str
) -> None:
    """``state:`` is the mirror when trusted, and nothing at all when it is not.

    Never report ``off`` when the mirror cannot be trusted. The Netatmo leg is not a
    fallback: a backup that lies is worse than no backup, because it sits at ``off`` and
    reports it with full confidence, turning every HomeKit outage into exactly the false
    ``off`` this work exists to eliminate. Going dark honestly is better.
    """
    row = ROWS[row_name]
    for slug, blk in blocks.items():
        states = trusted_states(slug, row)
        assert eval_ha_template(blk["state"], states) == row.state, f"{slug} / {row_name}"
        assert (
            eval_ha_template(blk["availability"], states) == row.available
        ), f"{slug} / {row_name} availability"


def test_facade_never_reports_confident_off_while_untrusted(
    blocks: dict[str, dict[str, Any]],
) -> None:
    """The incident class, asserted directly rather than only via the row table.

    An entity confidently reporting ``off`` while a boiler ran is the entire incident.
    """
    for row_name, row in ROWS.items():
        if row.available == "True":
            continue
        for slug, blk in blocks.items():
            reported = eval_ha_template(blk["state"], trusted_states(slug, row))
            assert reported == "unknown", f"{slug} / {row_name} leaked {reported!r}"


def test_cloud_leg_cannot_move_the_facade(blocks: dict[str, dict[str, Any]]) -> None:
    """The cloud leg has no effect whatsoever on what the facade reports.

    Seeds the cloud entity at ``on`` and at ``off`` against a mirror reading the opposite
    and asserts the reported state and availability are identical. This is the regression
    guard for "the Netatmo leg is not a fallback for state".
    """
    for slug, blk in blocks.items():
        cloud = cloud_entity_of(blk)
        for mirror in ("on", "off"):
            row = Row(mirror, "on", "on", "", "", "probe")
            base = trusted_states(slug, row)
            results = set()
            avail = set()
            for cloud_value in ("on", "off", "unavailable"):
                probe = dict(base) | {cloud: cloud_value}
                results.add(eval_ha_template(blk["state"], probe))
                avail.add(eval_ha_template(blk["availability"], probe))
            assert results == {"True" if mirror == "on" else "False"}, f"{slug}/{mirror}"
            assert avail == {"True"}, f"{slug}/{mirror}"


def test_state_and_availability_never_read_last_changed(
    blocks: dict[str, dict[str, Any]],
) -> None:
    """No ``last_changed`` anywhere: a stale leg keeps an old timestamp."""
    for slug, blk in blocks.items():
        assert "last_changed" not in blk["state"], slug
        assert "last_changed" not in blk["availability"], slug


def test_cloud_entity_confined_to_actuation_and_message(
    blocks: dict[str, dict[str, Any]],
) -> None:
    """The cloud entity id appears only where Part 1 section 1.6 permits it.

    Allowed: the target of the actuation write, and the notification message as diagnostic
    context. Forbidden: ``state:``, ``availability:``, either wait, either notification
    condition.
    """
    for slug, blk in blocks.items():
        cloud = cloud_entity_of(blk)
        assert cloud not in blk["state"], f"{slug}: cloud leaked into state"
        assert cloud not in blk["availability"], f"{slug}: cloud leaked into availability"
        for direction in DIRECTIONS:
            assert cloud not in wait_of(blk, direction), f"{slug}/{direction} wait"
            assert cloud not in complain_of(blk, direction), f"{slug}/{direction} condition"


# ============================================================================
# Actuation: unconditional, both channels (Part 1 sections 1.3, 1.6)
# ============================================================================


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_both_channels_driven_unconditionally(
    blocks: dict[str, dict[str, Any]], direction: str
) -> None:
    """Both channels are driven every time, and nothing can suppress either write.

    We cannot know in advance which channel will work, and *deciding* is exactly what
    caused the incident. Both steps carry ``continue_on_error`` so a failure of one still
    drives the other -- notably, an exception on the local boolean must not abort before
    the cloud, which is the verified-working actuator.
    """
    for slug, blk in blocks.items():
        steps = blk[direction]
        assert len(steps) == 4, f"{slug} / {direction}: expected actuation head + diagnostic"

        assert steps[0].get("action") == f"input_boolean.{direction}", f"{slug}/{direction}"
        assert steps[0]["target"]["entity_id"] == f"input_boolean.{slug}_command"

        assert steps[1].get("action") == f"switch.{direction}", f"{slug}/{direction}"
        assert steps[1]["target"]["entity_id"] == cloud_entity_of(blk)

        for index in (0, 1):
            assert steps[index].get("continue_on_error") is True, f"{slug}/{direction}/{index}"
            # HA honours `enabled: false` on a script action, which would silently
            # disable an actuation write while every other assertion still passed.
            assert "enabled" not in steps[index], f"{slug}/{direction}: step {index} disabled"

        assert steps[2]["timeout"] == "00:00:10", f"{slug}/{direction}"
        assert steps[2]["continue_on_timeout"] is True, f"{slug}/{direction}"

        for index, step in enumerate(steps):
            assert "variables" not in step, f"{slug}/{direction}: step {index} is conditional"
            assert ("choose" in step) == (index == 3), (
                f"{slug}/{direction}: step {index} must not branch"
            )


# ============================================================================
# Diagnostics: positive confirmation on the mirror (Part 1 sections 1.3, 1.4)
# ============================================================================


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_wait_positively_confirms_the_target_value(
    blocks: dict[str, dict[str, Any]], direction: str
) -> None:
    """Each wait confirms the target value itself, never the absence of its opposite.

    ``not is_state(mirror,'on')`` is satisfied by an ``unavailable`` or ``unknown`` mirror,
    which would let us silently conclude a turn-off succeeded. Confirm the target.
    """
    target = TARGET_OF[direction]
    for slug, blk in blocks.items():
        wait = wait_of(blk, direction)
        assert f"'{target}'" in wait, f"{slug}/{direction} does not mention {target!r}"
        assert "not " not in wait, f"{slug}/{direction} is expressed as a negation: {wait}"

        # An invalid mirror must NOT satisfy either direction's wait.
        for mirror in ("unknown", "unavailable"):
            states = trusted_states(slug, Row(mirror, "on", "on", "", "", "probe"))
            assert eval_ha_template(wait, states) == "False", f"{slug}/{direction}/{mirror}"


@pytest.mark.parametrize("row_name", list(ROWS))
def test_wait_and_complain_track_the_mirror_only(
    blocks: dict[str, dict[str, Any]], row_name: str
) -> None:
    """The wait is satisfied exactly when the mirror reads the target value.

    Deliberately **not** the old ``wait == not state`` identity. ``state:`` answers "is the
    load on?" and consults ``hk_ok``; the wait answers "did the mirror confirm the target?"
    and does not. The divergence is intentional -- do not "fix" it back.
    """
    row = ROWS[row_name]
    for slug, blk in blocks.items():
        states = trusted_states(slug, row)
        for direction in DIRECTIONS:
            expected = "True" if row.mirror == TARGET_OF[direction] else "False"
            wait = eval_ha_template(wait_of(blk, direction), states)
            assert wait == expected, f"{slug}/{row_name}/{direction}"
            # We complain exactly when the wait was not satisfied.
            complain = eval_ha_template(complain_of(blk, direction), states)
            assert complain == _negate(wait), f"{slug}/{row_name}/{direction} condition"


def test_notifications_have_per_direction_id(
    parsed: dict[str, Any], blocks: dict[str, dict[str, Any]]
) -> None:
    """Every facade notification carries a stable, per-direction ``notification_id``.

    Without an id HA generates a fresh one per call, so repeated failures pile up
    duplicates that cannot be replaced. Direction attribution is asserted through
    ``notification_of``, because a flat walk cannot tell direction: ten *swapped* ids
    would otherwise pass.
    """
    # "Mirror did not confirm" is absent from this list on purpose: under the mirror-only
    # design the message legitimately says the mirror did not confirm.
    stale_phrases = ("falling back", "HK turn")
    suffixes = {"turn_on": "_facade_on_unconfirmed", "turn_off": "_facade_off_unconfirmed"}
    valid_ids = {f"{slug}{suffix}" for slug in blocks for suffix in suffixes.values()}

    entries = all_notification_data(parsed)
    assert len(entries) == len(blocks) * len(DIRECTIONS)

    for entry in entries:
        assert entry["notification_id"] in valid_ids, entry
        for phrase in stale_phrases:
            assert phrase not in entry["title"], entry
            assert phrase not in entry["message"], entry

    for slug, blk in blocks.items():
        for direction, suffix in suffixes.items():
            data = notification_of(blk, direction)
            assert data["notification_id"] == f"{slug}{suffix}", f"{slug}/{direction}"
            # Both leg states are reported: the cheapest evidence that distinguishes a
            # truthful mirror from one frozen at its old value.
            assert f"input_boolean.{slug}_state" in data["message"], f"{slug}/{direction}"
            assert cloud_entity_of(blk) in data["message"], f"{slug}/{direction}"


def test_state_stub_raises_on_unknown_entity() -> None:
    """The layer-3 accessor raises on an unstubbed entity id.

    This is what makes ``trusted_states``' omission of the cloud entity load-bearing: a
    template that reads it fails loudly instead of silently passing.
    """
    with pytest.raises(KeyError, match="unstubbed entity"):
        eval_ha_template("{{ states('sensor.never_stubbed') }}", {})
