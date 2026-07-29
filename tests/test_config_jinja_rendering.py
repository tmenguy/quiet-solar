"""Tests for the Legrand/Netatmo facade switches in ``config/configuration.jinja``.

These tests catch bugs where:

- The facade's ``state:`` template reports ``off`` while some leg still reports ``on``
  (the QS-305 incident: a boiler ran ~7 hours after being commanded off).
- A predicate suppresses actuation on one of the two channels, so the channel that
  actually works is never driven.
- The wait / complain predicates disagree with what ``state:`` reports.
- ``persistent_notification`` calls lose their per-direction ``notification_id``.

**Permanent gate note.** ``config/configuration.jinja`` is a non-Python file, so
testmon cannot select this test: ``--impacted`` is blind to a template-only change.
Run ``python scripts/qs/quality_gate.py --quick tests/test_config_jinja_rendering.py``
in any PR that touches the template. CI's whole-suite run is the durable net.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import jinja2
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"

# Exercises both generator-side filters at lines 67-84: only the first entry
# passes `startswith('sensor.')` and `endswith('_power')`.
FIXTURE_NETATMO_ENTITIES = [
    "sensor.cumulus_pool_house_power",  # passes both filters
    "sensor.not_a_power_entity",  # fails endswith('_power')
    "switch.cumulus_pool_house",  # fails startswith('sensor.')
]

HEALTH_SENSORS = (
    "binary_sensor.legrand_homekit_link_healthy",
    "binary_sensor.homekit_hub_healthy",
)

DIRECTIONS = ("turn_on", "turn_off")

# The A1 fail-safe reporting table. Rows are NAMED, not numbered, so A1 and A3
# cannot drift apart. Each entry is (mirror, cloud, expected rendered state).
#
# `R_both_on` is load-bearing: without a both-legs-on row, an XOR implementation
# (`(mirror=='on') != (cloud=='on')`) passes every other row while reporting `off`
# with both legs on. The mirror uses `unknown`, never `unavailable`, because an
# `input_boolean` never goes unavailable.
STATE_ROWS: dict[str, tuple[str, str, str]] = {
    "R_both_on": ("on", "on", "True"),
    "R_incident": ("on", "off", "True"),
    "R_cloud_only": ("off", "on", "True"),
    "R_both_off": ("off", "off", "False"),
    "R_mirror_invalid_on": ("unknown", "on", "True"),
    "R_mirror_invalid_off": ("unknown", "off", "False"),
    "R_cloud_unavailable": ("off", "unavailable", "False"),
    "R_none_valid": ("unknown", "unknown", "unknown"),
}


# ============================================================================
# Layer 1: render the generator template
# ============================================================================


def render_config() -> str:
    """Render ``config/configuration.jinja`` with the generator-layer globals stubbed.

    ``integration_entities`` is the only global the template needs at generation time.
    """
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
    multi-constructor is registered on this subclass rather than on
    ``yaml.SafeLoader`` so every other test's ``yaml.safe_load`` stays strict.
    """


_TagTolerantLoader.add_multi_constructor(
    "!", lambda loader, suffix, node: f"!{suffix}"
)


def parse_config(rendered: str) -> dict[str, Any]:
    """Parse a rendered config with the tag-tolerant loader."""
    return yaml.load(rendered, Loader=_TagTolerantLoader)


def facade_blocks(parsed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the generated Legrand facade switch blocks, keyed by device slug.

    Walks ``template:`` items (some also carry ``trigger:``, so no item-shape
    assumption is made) and keeps ``- switch:`` entries whose ``unique_id`` ends
    with ``_facade``.
    """
    blocks: dict[str, dict[str, Any]] = {}
    for item in parsed["template"]:
        for blk in item.get("switch", []):
            unique_id = blk.get("unique_id", "")
            if unique_id.endswith("_facade"):
                blocks[unique_id.removesuffix("_facade")] = blk
    assert blocks, "no facade blocks found; the inventory would be vacuous"
    return blocks


def cloud_entity_of(blk: dict[str, Any]) -> str:
    """Return the Netatmo cloud entity id a facade block falls back to.

    Read out of ``availability:``, which is untouched by QS-305, so this helper
    works identically before and after the edit.
    """
    matches = [
        entity_id
        for entity_id in re.findall(r"states\('([^']+)'\)", blk["availability"])
        if entity_id.startswith("switch.")
    ]
    assert len(matches) == 1, f"expected exactly one cloud entity, got {matches}"
    return matches[0]


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

    Whitespace is collapsed rather than merely stripped, because ``state:`` is a
    ``>`` folded scalar whose more-indented body lines keep their newlines.
    """

    def _state_of(entity_id: str) -> str:
        if entity_id not in states:
            raise KeyError(f"template read unstubbed entity {entity_id!r}")
        return states[entity_id]

    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    env.globals["states"] = _state_of
    env.globals["is_state"] = lambda entity_id, value: _state_of(entity_id) == value
    return " ".join(env.from_string(text).render().split())


def leg_states(blk: dict[str, Any], slug: str, mirror: str, cloud: str) -> dict[str, str]:
    """Build a four-key ``states`` dict for one row of the A1 table.

    The two health sensors are seeded because the *pre-edit* ``state:`` reads them
    before the race tail; a two-key dict would raise ``KeyError`` on every row and
    mask the failure under test.
    """
    return {
        f"input_boolean.{slug}_state": mirror,
        cloud_entity_of(blk): cloud,
        HEALTH_SENSORS[0]: "on",
        HEALTH_SENSORS[1]: "on",
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
# A5: render and parse
# ============================================================================


def test_config_jinja_renders_and_parses(parsed: dict[str, Any]) -> None:
    """The template renders under the stubbed environment and parses as YAML."""
    assert isinstance(parsed, dict)
    assert "template" in parsed
    assert "input_boolean" in parsed


def test_facade_block_inventory(parsed: dict[str, Any], blocks: dict[str, dict[str, Any]]) -> None:
    """Every mapped Legrand device yields exactly one cloud-backed facade block.

    This is what keeps A1-A4 non-vacuous: it pins the set the other tests quantify
    over, and asserts cloud-backedness with a predicate that holds both before and
    after the QS-305 edit. Do not prune it.
    """
    expected_slugs = {
        key.removesuffix("_command")
        for key in parsed["input_boolean"]
        if key.endswith("_command")
    }
    assert expected_slugs, "no command booleans found; the template changed shape"
    assert set(blocks) == expected_slugs

    for slug, blk in blocks.items():
        assert "switch." in blk["availability"], f"{slug} is not cloud-backed"
        assert cloud_entity_of(blk).startswith("switch."), slug


# ============================================================================
# A1: fail-safe reporting
# ============================================================================


def test_facade_state_never_reads_last_changed(blocks: dict[str, dict[str, Any]]) -> None:
    """The facade ``state:`` elects no winner and trusts no health sensor.

    ``last_changed`` was the wrong arbiter (a stale leg keeps an old timestamp), and
    gating mirror validity on ``hk_ok`` let a healthy-looking link vouch for a frozen
    mirror. The health-sensor entity ids are asserted absent *positively*, so a
    re-inlined health test cannot survive as a mere name change.
    """
    for slug, blk in blocks.items():
        state = blk["state"]
        assert "last_changed" not in state, slug
        assert "hk_ok" not in state, slug
        for sensor in HEALTH_SENSORS:
            assert sensor not in state, f"{slug} still reads {sensor}"


@pytest.mark.parametrize("row", list(STATE_ROWS))
def test_facade_state_is_fail_safe(blocks: dict[str, dict[str, Any]], row: str) -> None:
    """The facade reports ``off`` only when no valid leg says ``on``.

    False-``off`` is the incident's failure mode and is narrowed to "all valid legs
    stuck at ``off``"; false-``on`` is the accepted cost and is what D3 reports on.
    """
    mirror, cloud, expected = STATE_ROWS[row]
    for slug, blk in blocks.items():
        states = leg_states(blk, slug, mirror, cloud)
        assert eval_ha_template(blk["state"], states) == expected, f"{slug} / {row}"


# ============================================================================
# A2: unconditional actuation
# ============================================================================


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_both_channels_driven_unconditionally(
    blocks: dict[str, dict[str, Any]], direction: str
) -> None:
    """Both the HomeKit command boolean and the Netatmo cloud entity are always driven.

    This is the QS-305 fix. The old script *predicted* whether the HomeKit leg had
    worked and drove the cloud leg only if it concluded otherwise; any leg can lie, so
    a wrong prediction meant the one channel that worked was never driven. There is no
    longer a decision to get wrong: no ``variables:``, no ``choose:`` in the actuation
    head, and therefore no predicate that can suppress a command.

    The local boolean is written *first* so it still happens if the cloud call fails.
    ``continue_on_error`` on the cloud step keeps a ``HomeAssistantError`` from aborting
    the script before the diagnostic wait can report the failure.
    """
    for slug, blk in blocks.items():
        steps = blk[direction]
        assert len(steps) == 4, f"{slug} / {direction}: expected actuation head + diagnostic"

        assert steps[0].get("action") == f"input_boolean.{direction}", f"{slug} / {direction}"
        assert steps[0]["target"]["entity_id"] == f"input_boolean.{slug}_command"

        assert steps[1].get("action") == f"switch.{direction}", f"{slug} / {direction}"
        assert steps[1]["target"]["entity_id"] == cloud_entity_of(blk)
        assert steps[1].get("continue_on_error") is True, f"{slug} / {direction}"

        # `continue_on_timeout` is load-bearing: without it a timeout errors the script
        # and the diagnostic `choose` never runs.
        assert steps[2]["timeout"] == "00:00:10", f"{slug} / {direction}"
        assert steps[2]["continue_on_timeout"] is True, f"{slug} / {direction}"

        for index, step in enumerate(steps):
            assert "variables" not in step, f"{slug} / {direction}: step {index} is conditional"
            assert ("choose" in step) == (index == 3), (
                f"{slug} / {direction}: step {index} must not branch"
            )


# ============================================================================
# A3: the diagnostic predicates agree with reporting
# ============================================================================


@pytest.mark.parametrize("row", list(STATE_ROWS))
def test_diagnostic_predicates_agree_with_state(
    blocks: dict[str, dict[str, Any]], row: str
) -> None:
    """The wait and complain predicates are derived from what ``state:`` reports.

    Letting ``state:``, ``wait_template:`` and the fallback condition disagree was the
    QS-305 incident's first defect. With ``S`` the reported state, this pins
    ``wait(turn_off) = not S``, ``wait(turn_on) = S`` and ``complain = not wait`` in both
    directions, so the three predicates cannot drift apart again.
    """
    mirror, cloud, expected = STATE_ROWS[row]
    for slug, blk in blocks.items():
        states = leg_states(blk, slug, mirror, cloud)
        waits = {d: eval_ha_template(wait_of(blk, d), states) for d in DIRECTIONS}
        complains = {d: eval_ha_template(complain_of(blk, d), states) for d in DIRECTIONS}

        if row == "R_none_valid":
            # `state:` reports `unknown`, which is not a boolean, so the identities are
            # pinned as literals instead. Both legs invalid reads as "not on".
            assert expected == "unknown", row
            assert waits["turn_off"] == "True", slug
            assert waits["turn_on"] == "False", slug
            assert complains["turn_off"] == "False", slug
            assert complains["turn_on"] == "True", slug
        else:
            assert waits["turn_off"] == _negate(expected), f"{slug} / {row}"
            assert waits["turn_on"] == expected, f"{slug} / {row}"

        for direction in DIRECTIONS:
            assert complains[direction] == _negate(waits[direction]), f"{slug} / {row} / {direction}"


def test_diagnostic_reports_the_incident_state(blocks: dict[str, dict[str, Any]]) -> None:
    """In the observed incident state, a ``turn_off`` complains instead of going silent.

    ``R_incident`` (mirror ``on``, cloud ``off``) is what the boiler actually reported
    while it ran for ~7 hours. Pre-fix the wait was satisfied by the cloud leg alone and
    nothing was raised; now the wait fails and the notification fires.
    """
    mirror, cloud, _ = STATE_ROWS["R_incident"]
    for slug, blk in blocks.items():
        states = leg_states(blk, slug, mirror, cloud)
        assert eval_ha_template(wait_of(blk, "turn_off"), states) == "False", slug
        assert eval_ha_template(complain_of(blk, "turn_off"), states) == "True", slug


# ============================================================================
# A4: notifications
# ============================================================================


def test_notifications_have_per_direction_id(
    parsed: dict[str, Any], blocks: dict[str, dict[str, Any]]
) -> None:
    """Every facade notification carries a stable, per-direction ``notification_id``.

    Without an id HA generates a fresh one per call, so repeated failures pile up
    duplicates that cannot be replaced. Per-direction ids also stop a ``turn_off``
    result from erasing a ``turn_on`` failure.

    Direction attribution is asserted through ``notification_of``, because a flat walk
    cannot tell direction: ten *swapped* ids would pass a whole-render check alone.
    """
    stale_phrases = ("falling back", "Mirror did not confirm", "HK turn")
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
            assert data["notification_id"] == f"{slug}{suffix}", f"{slug} / {direction}"
            # The message carries both leg states, the only cheap evidence that
            # distinguishes a truthful mirror from one frozen at its old value.
            assert f"input_boolean.{slug}_state" in data["message"], f"{slug} / {direction}"
            assert cloud_entity_of(blk) in data["message"], f"{slug} / {direction}"


def test_state_stub_raises_on_unknown_entity() -> None:
    """The layer-3 accessor raises on an unstubbed entity id.

    A typo'd entity id must be a loud red, never a silent falsy read.
    """
    with pytest.raises(KeyError, match="unstubbed entity"):
        eval_ha_template("{{ states('sensor.never_stubbed') }}", {})
