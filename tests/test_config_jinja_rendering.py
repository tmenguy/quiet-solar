"""Tests for the Legrand/Netatmo facade switches in ``config/configuration.jinja``.

The design pinned here is ``docs/stories/QS-305.story_review_fix_%2301.md`` Part 1 as amended
by ``docs/stories/QS-305.story_review_fix_%2302.md`` Part A. The rule everything derives from:

    **Never trust a value you just wrote. Never report a value you can't trust.**

We write ``input_boolean.<slug>_command`` and we write the Netatmo cloud switch, so neither can
be evidence -- they only parrot back what we told them. The mirror,
``input_boolean.<slug>_state``, is written only by Apple Home from watching the real device, so
it is the only honest witness. See the story for the forked-integration source citations.

These tests catch bugs where:

- The facade becomes ``unavailable``. HA filters unavailable entities out of entity-service
  calls (``helpers/service.py:763``) and only logs a warning, and ``switch.turn_on/off`` are
  entity services -- so an unavailable facade **silently swallows every actuation**, with no
  script run and therefore no diagnostic either. That is the incident's shape by another route,
  and it would fire on every HA restart while the health sensors warm up.
- The untrusted branch emits the *string* ``unknown`` instead of ``none``. ``cv.boolean`` is
  applied without ``none_on_unknown_unavailable``, so a string logs an ERROR on every
  evaluation; only a real ``None`` is accepted cleanly.
- A wait or notification condition consults the cloud leg, making the wait vacuously true and
  the notification unreachable.
- A wait is expressed as "the mirror is not ``on``" instead of positively confirming the
  target, so an ``unknown``/``unavailable`` mirror silently satisfies a turn-off.
- A cloud-side exception aborts the script before the diagnostic can fire. ``pyatmo`` raises
  plain ``Exception`` subclasses, which ``continue_on_error`` does **not** swallow, so the cloud
  write lives in its own ``parallel:`` branch.
- A predicate, or ``enabled: false``, can prevent either actuation write from executing.

**Permanent gate note.** ``config/configuration.jinja`` is a non-Python file, so testmon cannot
select this test: ``--impacted`` is blind to a template-only change. Run
``python scripts/qs/quality_gate.py --quick tests/test_config_jinja_rendering.py``
in any PR that touches the template. CI's whole-suite run is the durable net.
"""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jinja2
import pytest
import yaml
from homeassistant.components.template import validators as template_validators
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.script import Script
from homeassistant.helpers.template import Template
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import async_mock_service

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

DIRECTIONS = ("turn_on", "turn_off")

# The value each direction positively confirms on the mirror.
TARGET_OF = {"turn_on": "on", "turn_off": "off"}


class _CloudApiError(Exception):
    """Stand-in for pyatmo's ``ApiError``.

    Verified: ``ApiError``, ``ApiThrottlingError`` and ``ApiTooManyRequestError`` are plain
    ``Exception`` subclasses (``custom_components/netatmo/pyatmo/exceptions.py``), **not**
    ``HomeAssistantError``, so ``continue_on_error`` does not swallow them
    (``helpers/script.py`` -- "Only Home Assistant errors can be ignored", then ``raise``).
    Declared locally to keep this test hermetic.
    """


@dataclass(frozen=True)
class Row:
    """One scenario for the emitted ``state:`` template."""

    mirror: str
    link: str
    hub: str
    state: str
    why: str


# Rows are NAMED, not numbered, so expectations cannot drift.
#
# `R_link_down` and `R_hub_down` vary the two health sensors independently, which pins `hk_ok`
# as an AND -- an OR implementation would survive if only one were ever exercised.
#
# "None" is the *rendered* form of `{{ none }}`; HA literal-evals it back to Python None, which
# publishes state `unknown` while the entity stays AVAILABLE.
ROWS: dict[str, Row] = {
    "R_mirror_on_trusted": Row("on", "on", "on", "True", "the normal running state"),
    "R_mirror_off_trusted": Row(
        "off", "on", "on", "False", "the only case that may report a confident off"
    ),
    "R_mirror_unknown": Row("unknown", "on", "on", "None", "mirror not restored -> report nothing"),
    "R_mirror_unavailable": Row("unavailable", "on", "on", "None", "mirror invalid"),
    "R_link_down": Row("on", "off", "on", "None", "pins hk_ok as AND on the link sensor"),
    "R_hub_down": Row("on", "on", "off", "None", "pins hk_ok as AND on the hub sensor"),
    "R_health_down_mirror_off": Row(
        "off",
        "off",
        "off",
        "None",
        "THE incident class: never report a confident off when we cannot see",
    ),
}


# ============================================================================
# Layer 1: render the generator template
# ============================================================================


def _environment(loader: jinja2.BaseLoader) -> jinja2.Environment:
    """Build the generator-layer environment with the one global the template needs.

    ``StrictUndefined`` is a **test-harness** choice, not a production guarantee: this repo
    has no renderer, and HA's own Jinja runs with ``strict=None``. It is used here so a typo
    in the inventory fails loudly in tests rather than rendering an empty string.
    """
    env = jinja2.Environment(loader=loader, undefined=jinja2.StrictUndefined)
    env.globals["integration_entities"] = lambda domain: FIXTURE_NETATMO_ENTITIES
    return env


def render_config() -> str:
    """Render ``config/configuration.jinja`` with the generator-layer globals stubbed."""
    return _environment(jinja2.FileSystemLoader(CONFIG_DIR)).get_template(
        "configuration.jinja"
    ).render()


# The marker that closes the hand-edited device inventory.
_INVENTORY_END = "{## --- END EDIT --- ##}"


def render_config_with_extra_device(slug: str, name: str, cloud_entity: str | None) -> str:
    """Render the real template with one extra device appended to the inventory.

    The inventory is hardcoded in the template rather than injected, so exercising a
    HomeKit-only device (no ``cloud_entity``) or an awkward ``name`` means adding an entry.
    Done by source injection at the inventory's own end marker, so the block under test is
    the real one -- no forked copy of the facade definition to drift.
    """
    source = (CONFIG_DIR / "configuration.jinja").read_text()
    assert source.count(_INVENTORY_END) == 1, "inventory end marker moved"
    entry = {"slug": slug, "name": name}
    if cloud_entity is not None:
        entry["cloud_entity"] = cloud_entity
    injected = (
        "{%- set legrand_switches_to_map.switches = "
        f"legrand_switches_to_map.switches + [{entry!r}] %}}\n" + _INVENTORY_END
    )
    patched = source.replace(_INVENTORY_END, injected)
    return _environment(jinja2.DictLoader({"configuration.jinja": patched})).get_template(
        "configuration.jinja"
    ).render()


# ============================================================================
# Layer 2: parse the emitted YAML
# ============================================================================


class _TagTolerantLoader(yaml.SafeLoader):
    """SafeLoader that degrades unknown ``!tag`` nodes to plain strings.

    The rendered config uses ``!include``, which ``yaml.SafeLoader`` rejects. The
    multi-constructor is registered on this subclass rather than on ``yaml.SafeLoader``
    so every other test's ``yaml.safe_load`` stays strict.
    """


_TagTolerantLoader.add_multi_constructor("!", lambda loader, suffix, node: f"!{suffix}")


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


# --- step accessors -------------------------------------------------------------------
#
# Shape per direction, identical for both variants:
#   [0] input_boolean.turn_<dir>              (continue_on_error)
#   [1] parallel:
#         [0] sequence: [ switch.turn_<dir> ] -- ONLY when `cloud_entity` is defined
#         [-1] sequence: [ wait_template, choose(+default) ]
#
# The diagnostic is always the LAST parallel branch, so every accessor below is uniform
# across a cloud-backed and a HomeKit-only device. `cloud_entity` is optional: a device
# without one gets the same facade with no cloud branch emitted.
#
# The cloud write is isolated in its own branch so a plain-Exception failure cannot abort
# the diagnostic: `_async_step_parallel` gathers with `return_exceptions=True` and re-raises
# only after every branch has completed.


def boolean_step_of(blk: dict[str, Any], direction: str) -> dict[str, Any]:
    """Return the local command-boolean actuation step."""
    return blk[direction][0]


def parallel_of(blk: dict[str, Any], direction: str) -> list[dict[str, Any]]:
    """Return the parallel branches of a direction script."""
    return blk[direction][1]["parallel"]


def has_cloud(blk: dict[str, Any]) -> bool:
    """Whether this facade drives a Netatmo cloud channel."""
    counts = {len(parallel_of(blk, d)) for d in DIRECTIONS}
    assert counts in ({1}, {2}), f"inconsistent branch counts across directions: {counts}"
    return counts == {2}


def cloud_step_of(blk: dict[str, Any], direction: str) -> dict[str, Any]:
    """Return the Netatmo cloud actuation step, inside its own parallel branch."""
    assert has_cloud(blk), "this facade has no cloud channel"
    return parallel_of(blk, direction)[0]["sequence"][0]


def diagnostic_of(blk: dict[str, Any], direction: str) -> list[dict[str, Any]]:
    """Return the diagnostic branch's sequence: ``[wait_template, choose]``.

    Always the last parallel branch, so this is variant-independent.
    """
    return parallel_of(blk, direction)[-1]["sequence"]


def actuation_steps_of(blk: dict[str, Any], direction: str) -> list[dict[str, Any]]:
    """Return every actuation write: the boolean, plus the cloud when there is one."""
    steps = [boolean_step_of(blk, direction)]
    if has_cloud(blk):
        steps.append(cloud_step_of(blk, direction))
    return steps


def cloud_entity_of(blk: dict[str, Any]) -> str:
    """Return the Netatmo cloud entity id a facade block actuates."""
    entity_id = cloud_step_of(blk, "turn_on")["target"]["entity_id"]
    assert entity_id.startswith("switch."), entity_id
    return entity_id


def wait_of(blk: dict[str, Any], direction: str) -> str:
    """Return the ``wait_template`` of a facade block's direction script."""
    return diagnostic_of(blk, direction)[0]["wait_template"]


def wait_step_of(blk: dict[str, Any], direction: str) -> dict[str, Any]:
    """Return the whole wait step, for timeout assertions."""
    return diagnostic_of(blk, direction)[0]


def choose_of(blk: dict[str, Any], direction: str) -> dict[str, Any]:
    """Return the diagnostic ``choose`` step."""
    return diagnostic_of(blk, direction)[1]


def complain_of(blk: dict[str, Any], direction: str) -> str:
    """Return the diagnostic ``choose`` condition of a direction script."""
    return choose_of(blk, direction)["choose"][0]["conditions"]


def notification_of(blk: dict[str, Any], direction: str) -> dict[str, Any]:
    """Return the notification ``data`` mapping of a direction script."""
    return choose_of(blk, direction)["choose"][0]["sequence"][0]["data"]


def dismiss_ids_of(blk: dict[str, Any], direction: str) -> set[str]:
    """Return the notification ids the success path dismisses.

    Both directions are dismissed on either success path: a ``turn_off`` that timed out would
    otherwise leave its alarm standing through every later successful ``turn_on`` -- for hours
    on a solar-scheduled load -- showing a stale alarm for a device that just confirmed.
    """
    steps = choose_of(blk, direction)["default"]
    assert all(s["action"] == "persistent_notification.dismiss" for s in steps), steps
    return {s["data"]["notification_id"] for s in steps}


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


def sub_templates_of(blk: dict[str, Any]) -> dict[str, str]:
    """Return every template string a facade block evaluates at runtime.

    Includes the notification *message* templates: a broken template there raises at runtime
    on a step with no ``continue_on_error``, turning the diagnostic into a silent script
    failure -- exactly the class this task exists to eliminate.
    """
    out = {"state": blk["state"]}
    for direction in DIRECTIONS:
        out[f"{direction}.wait"] = wait_of(blk, direction)
        out[f"{direction}.conditions"] = complain_of(blk, direction)
        out[f"{direction}.message"] = notification_of(blk, direction)["message"]
    return out


# ============================================================================
# Layer 3: evaluate the emitted HA-runtime templates
# ============================================================================


def _negate(rendered: str) -> str:
    """Flip a rendered boolean string. Total over the two values it is used with."""
    return {"True": "False", "False": "True"}[rendered]


def eval_ha_template(text: str, states: dict[str, str]) -> str:
    """Render an emitted HA-runtime template with stubbed HA globals.

    Whitespace is collapsed rather than merely stripped, because ``state:`` is a ``>`` folded
    scalar whose more-indented body lines keep their newlines.
    """

    def _state_of(entity_id: str) -> str:
        if entity_id not in states:
            raise KeyError(f"template read unstubbed entity {entity_id!r}")
        return states[entity_id]

    class _FixedNow:
        """Stub for ``now()`` so message templates are evaluable off a real hass."""

        @staticmethod
        def isoformat() -> str:
            return "2026-07-30T12:00:00+02:00"

    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    env.globals["states"] = _state_of
    env.globals["is_state"] = lambda entity_id, value: _state_of(entity_id) == value
    env.globals["now"] = lambda: _FixedNow()
    return " ".join(env.from_string(text).render().split())


def trusted_states(slug: str, row: Row) -> dict[str, str]:
    """Build the ``states`` dict for one row.

    The cloud entity is **deliberately absent**. Every template under test must consult only
    the mirror and the two health sensors, so if a future edit reintroduces the cloud leg into
    ``state:``, a wait or a notification condition, the raising accessor turns it into a loud
    ``KeyError`` rather than a silent pass.
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

    This is what keeps every other assertion non-vacuous: it pins the set they quantify over.
    Do not prune it. There is no longer an HK-only variant -- that branch was dead code on the
    superseded design and was deleted, so ``cloud_entity`` is now effectively required and a
    missing one fails the render loudly under ``StrictUndefined``.
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
# Reporting (plan #02 section A1)
# ============================================================================


def test_facade_is_never_unavailable(blocks: dict[str, dict[str, Any]]) -> None:
    """No ``availability:`` template may exist on the facade.

    HA filters unavailable entities out of entity-service calls and only logs a warning, and
    ``switch.turn_on/off`` are entity services -- so an unavailable facade silently swallows
    every actuation quiet-solar issues, with no script run and therefore no diagnostic either.
    Both health sensors force themselves ``off`` for 120 s after an HA restart, which exceeds
    the ~70 s quiet-solar needs to give up on a command. Do not reintroduce this key.
    """
    for slug, blk in blocks.items():
        assert "availability" not in blk, f"{slug} would swallow actuation when untrusted"


@pytest.mark.parametrize("row_name", list(ROWS))
def test_facade_reports_only_what_the_mirror_can_prove(
    blocks: dict[str, dict[str, Any]], row_name: str
) -> None:
    """``state:`` is the mirror when trusted, and ``none`` when it is not."""
    row = ROWS[row_name]
    for slug, blk in blocks.items():
        states = trusted_states(slug, row)
        assert eval_ha_template(blk["state"], states) == row.state, f"{slug} / {row_name}"


def test_untrusted_state_emits_none_not_the_string_unknown(
    blocks: dict[str, dict[str, Any]],
) -> None:
    """The untrusted branch must emit ``none``, never the literal ``unknown``.

    ``cv.boolean`` is applied without ``none_on_unknown_unavailable``, so the string
    ``unknown`` fails validation, takes the error path and logs an ERROR on **every**
    evaluation. Only a real ``None`` is accepted cleanly.
    """
    for slug, blk in blocks.items():
        assert "unknown" not in blk["state"], f"{slug} still emits a literal state string"
        assert "unavailable" not in blk["state"], slug


async def test_untrusted_state_parses_to_none_without_error(
    hass: HomeAssistant, blocks: dict[str, dict[str, Any]]
) -> None:
    """End-to-end: HA parses the untrusted branch to ``None``, with no ERROR logged.

    Asserted through Home Assistant's own ``Template`` and the template platform's validator
    rather than the layer-3 stub, because this is the single most likely thing to be silently
    wrong. The companion *no-ERROR-logged* assertion lives in
    ``test_actuation_reaches_the_entity_while_untrusted``, which stands up the real platform --
    the validator is never invoked by rendering a ``Template`` directly, so asserting it here
    would be decorative.
    """
    for slug, blk in blocks.items():
        # Untrusted: HomeKit link down.
        hass.states.async_set(LINK_SENSOR, "off")
        hass.states.async_set(HUB_SENSOR, "on")
        hass.states.async_set(f"input_boolean.{slug}_state", "on")
        result = Template(blk["state"], hass).async_render(parse_result=True)
        assert result is None, f"{slug} rendered {result!r} instead of None"
        # This is the check the template platform actually applies to `state:`.
        assert template_validators.check_result_for_none(result) is True, slug

        # Trusted: the same template must still yield a real boolean.
        hass.states.async_set(LINK_SENSOR, "on")
        trusted = Template(blk["state"], hass).async_render(parse_result=True)
        assert trusted is True, f"{slug} rendered {trusted!r} instead of True"


async def test_actuation_reaches_the_entity_while_untrusted(
    hass: HomeAssistant, blocks: dict[str, dict[str, Any]]
) -> None:
    """The §A1 regression, closed by demonstration rather than by argument.

    Sets up the real facade through the ``template`` integration with the HomeKit link down, so
    the facade publishes ``unknown``, then calls ``switch.turn_on`` on it and asserts the local
    actuation actually executed. Had an ``availability:`` template made the entity unavailable,
    HA would have filtered it out of the service call with only a warning and this would fail --
    which is precisely how every actuation would be lost for 120 s after each HA restart.
    """
    slug, blk = next(iter(blocks.items()))
    mirror = f"input_boolean.{slug}_state"

    # Untrusted (link down) but the turn_on wait is immediately satisfied, so the script is fast.
    hass.states.async_set(LINK_SENSOR, "off")
    hass.states.async_set(HUB_SENSOR, "on")
    hass.states.async_set(mirror, "on")

    driven = async_mock_service(hass, "input_boolean", "turn_on")
    async_mock_service(hass, "persistent_notification", "dismiss")

    # C1: the real platform DOES invoke `template_validators.boolean` on `state:`, so this is
    # where a no-ERROR assertion is load-bearing. Under a string-`unknown` mutation the
    # platform logs "Received invalid switch state: unknown"; under `{{ none }}` it is silent.
    errors: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            errors.append(record)

    handler = _Capture(level=logging.ERROR)
    validators_logger = logging.getLogger(template_validators.__name__)
    validators_logger.addHandler(handler)

    # `async_setup_component` runs config validation, which mutates mappings in place
    # (template strings -> Template objects). `blocks` is module-scoped, so pass a copy or
    # later tests in this module see a contaminated fixture.
    config = {"template": [{"switch": [copy.deepcopy(blk)]}]}
    assert await async_setup_component(hass, "template", config)
    await hass.async_block_till_done()

    facade = next(e for e in hass.states.async_entity_ids("switch") if e.endswith("_facade"))
    assert hass.states.get(facade).state == "unknown", "expected the untrusted scenario"

    await hass.services.async_call(
        "switch", "turn_on", {"entity_id": facade}, blocking=True
    )
    await hass.async_block_till_done()

    validators_logger.removeHandler(handler)

    assert len(driven) == 1, "the untrusted facade swallowed the actuation"
    assert driven[0].data["entity_id"] == [f"input_boolean.{slug}_command"]
    assert not errors, f"the platform rejected the state template: {[r.getMessage() for r in errors]}"


def test_cloud_leg_cannot_move_the_facade(blocks: dict[str, dict[str, Any]]) -> None:
    """The cloud leg has no effect whatsoever on what the facade reports."""
    for slug, blk in blocks.items():
        cloud = cloud_entity_of(blk)
        for mirror in ("on", "off"):
            base = trusted_states(slug, Row(mirror, "on", "on", "", "probe"))
            results = {
                eval_ha_template(blk["state"], dict(base) | {cloud: value})
                for value in ("on", "off", "unavailable")
            }
            assert results == {"True" if mirror == "on" else "False"}, f"{slug}/{mirror}"


def test_no_sub_template_reads_last_changed(blocks: dict[str, dict[str, Any]]) -> None:
    """No ``last_changed`` in any evaluated template, including message text.

    A stale leg keeps an old timestamp, so ``last_changed`` is never a validity signal.
    Message text is included because a broken template there fails silently at runtime.
    """
    for slug, blk in blocks.items():
        for name, text in sub_templates_of(blk).items():
            assert "last_changed" not in text, f"{slug} / {name}"


def test_every_sub_template_evaluates(blocks: dict[str, dict[str, Any]]) -> None:
    """Every runtime template renders, including the notification messages.

    Substring checks alone would let a broken message template through; at runtime that raises
    on a step with no ``continue_on_error``, converting the diagnostic into a silent failure.
    """
    for slug, blk in blocks.items():
        states = trusted_states(slug, ROWS["R_mirror_on_trusted"])
        for name, text in sub_templates_of(blk).items():
            out = eval_ha_template(text, states)
            assert out, f"{slug} / {name} rendered empty"


def test_cloud_entity_confined_to_actuation(blocks: dict[str, dict[str, Any]]) -> None:
    """The cloud entity id appears only as an actuation target.

    It is no longer quoted in the notification message either: the fork writes state
    optimistically and ignores the API's return value, so ``cloud=`` echoed our own command (or
    a stale pre-command value) and could never tell the operator whether the write was accepted.
    """
    for slug, blk in blocks.items():
        cloud = cloud_entity_of(blk)
        assert cloud not in blk["state"], f"{slug}: cloud leaked into state"
        for direction in DIRECTIONS:
            assert cloud not in wait_of(blk, direction), f"{slug}/{direction} wait"
            assert cloud not in complain_of(blk, direction), f"{slug}/{direction} condition"
            message = notification_of(blk, direction)["message"]
            assert cloud not in message, f"{slug}/{direction} message quotes an inert leg"


# ============================================================================
# Actuation (plan #01 sections 1.3, 1.6; plan #02 B1)
# ============================================================================


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_both_channels_driven_unconditionally(
    blocks: dict[str, dict[str, Any]], direction: str
) -> None:
    """Both channels are driven every time, and nothing can suppress either write."""
    for slug, blk in blocks.items():
        steps = blk[direction]
        assert len(steps) == 2, f"{slug}/{direction}: expected boolean write + parallel"

        boolean_step = boolean_step_of(blk, direction)
        assert boolean_step.get("action") == f"input_boolean.{direction}", f"{slug}/{direction}"
        assert boolean_step["target"]["entity_id"] == f"input_boolean.{slug}_command"

        cloud_step = cloud_step_of(blk, direction)
        assert cloud_step.get("action") == f"switch.{direction}", f"{slug}/{direction}"
        assert cloud_step["target"]["entity_id"] == cloud_entity_of(blk)

        for index, step in enumerate(actuation_steps_of(blk, direction)):
            assert step.get("continue_on_error") is True, f"{slug}/{direction}/{index}"
            # HA honours `enabled: false` on a script action, which would silently disable an
            # actuation write while every other assertion still passed.
            assert "enabled" not in step, f"{slug}/{direction}: actuation {index} disabled"
            assert "variables" not in step, f"{slug}/{direction}: actuation {index} conditional"
            assert "choose" not in step, f"{slug}/{direction}: actuation {index} branches"

        # The cloud write is isolated so its exceptions cannot abort the diagnostic.
        branches = steps[1]["parallel"]
        assert len(branches) == 2, f"{slug}/{direction}"
        assert len(branches[0]["sequence"]) == 1, f"{slug}/{direction}: cloud branch not isolated"

        wait_step = wait_step_of(blk, direction)
        assert wait_step["timeout"] == "00:00:10", f"{slug}/{direction}"
        assert wait_step["continue_on_timeout"] is True, f"{slug}/{direction}"


@pytest.mark.parametrize("direction", DIRECTIONS)
async def test_notification_fires_when_cloud_write_raises(
    hass: HomeAssistant, blocks: dict[str, dict[str, Any]], direction: str
) -> None:
    """B1: a non-``HomeAssistantError`` from the cloud step must not suppress the diagnostic.

    ``continue_on_error`` only swallows ``HomeAssistantError`` subclasses, but pyatmo raises
    plain ``Exception`` subclasses on a 429/throttle or a dropped connection -- a documented
    operating condition for this fork. Before the ``parallel:`` split that aborted the script
    at the cloud step, so the single failure the diagnostic exists to report was the one that
    silenced it.

    Runs the real rendered sequence through HA's ``Script`` helper.
    """
    slug, blk = next(iter(blocks.items()))
    mirror = f"input_boolean.{slug}_state"

    # Mirror parked at the value that does NOT confirm this direction, so the wait must fail.
    hass.states.async_set(mirror, "on" if direction == "turn_off" else "off")

    async_mock_service(hass, "input_boolean", direction)
    created = async_mock_service(hass, "persistent_notification", "create")
    async_mock_service(hass, "persistent_notification", "dismiss")

    async def _raise(call: Any) -> None:
        raise _CloudApiError("429 throttled")

    hass.services.async_register("switch", direction, _raise)

    sequence = copy.deepcopy(blk[direction])
    # Shorten the wait so the test is fast; the 10 s value is asserted structurally above.
    diagnostic_of({direction: sequence}, direction)[0]["timeout"] = {"seconds": 0.05}

    script = Script(hass, cv.SCRIPT_SCHEMA(sequence), f"{slug} {direction}", "template")
    with pytest.raises(_CloudApiError):
        await script.async_run(context=Context())
    await hass.async_block_till_done()

    assert len(created) == 1, "the cloud exception suppressed the diagnostic notification"
    assert created[0].data["notification_id"] == f"{slug}_facade_{TARGET_OF[direction]}_unconfirmed"


# ============================================================================
# Diagnostics (plan #01 sections 1.3, 1.4)
# ============================================================================


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_wait_positively_confirms_the_target_value(
    blocks: dict[str, dict[str, Any]], direction: str
) -> None:
    """Each wait confirms the target value itself, never the absence of its opposite."""
    target = TARGET_OF[direction]
    for slug, blk in blocks.items():
        wait = wait_of(blk, direction)
        assert f"'{target}'" in wait, f"{slug}/{direction} does not mention {target!r}"
        assert not re.search(r"\bnot\s+is_state", wait), (
            f"{slug}/{direction} is expressed as a negation: {wait}"
        )

        for mirror in ("unknown", "unavailable"):
            states = trusted_states(slug, Row(mirror, "on", "on", "", "probe"))
            assert eval_ha_template(wait, states) == "False", f"{slug}/{direction}/{mirror}"


@pytest.mark.parametrize("row_name", list(ROWS))
def test_wait_and_complain_track_the_mirror_only(
    blocks: dict[str, dict[str, Any]], row_name: str
) -> None:
    """The wait is satisfied exactly when the mirror reads the target value.

    Deliberately **not** the old ``wait == not state`` identity. ``state:`` answers "is the load
    on?" and consults ``hk_ok``; the wait answers "did the mirror confirm the target?" and does
    not. The divergence is intentional -- do not "fix" it back.
    """
    row = ROWS[row_name]
    for slug, blk in blocks.items():
        states = trusted_states(slug, row)
        for direction in DIRECTIONS:
            expected = "True" if row.mirror == TARGET_OF[direction] else "False"
            wait = eval_ha_template(wait_of(blk, direction), states)
            assert wait == expected, f"{slug}/{row_name}/{direction}"
            complain = eval_ha_template(complain_of(blk, direction), states)
            assert complain == _negate(wait), f"{slug}/{row_name}/{direction} condition"


def test_notifications_have_per_direction_id(
    parsed: dict[str, Any], blocks: dict[str, dict[str, Any]]
) -> None:
    """Every facade notification carries a stable, per-direction ``notification_id``."""
    # "Mirror did not confirm" is absent on purpose: under the mirror-only design the message
    # legitimately says the mirror did not confirm.
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
            assert f"input_boolean.{slug}_state" in data["message"], f"{slug}/{direction}"


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_success_path_dismisses_the_notification(
    blocks: dict[str, dict[str, Any]], direction: str
) -> None:
    """A confirmed command clears any stale notification for that direction.

    Otherwise a transient failure leaves an alarm standing forever, even once the mirror has
    confirmed -- the same "a stale signal carries no information" problem as the log line this
    notification replaced.
    """
    for slug, blk in blocks.items():
        assert dismiss_ids_of(blk, direction) == {
            f"{slug}_facade_on_unconfirmed",
            f"{slug}_facade_off_unconfirmed",
        }, f"{slug}/{direction} must clear the other direction's stale alarm too"


# ============================================================================
# `cloud_entity` is optional (plan #03 section A2)
# ============================================================================


def test_homekit_only_device_gets_the_full_facade() -> None:
    """A device without ``cloud_entity`` gets the same facade, minus the cloud channel.

    Previously this was a duplicated ``{% else %}`` block with no wait, no notification and
    no dismiss, plus an ``availability:`` key that could take the entity offline -- which is
    why a mutation making it report a confident ``off`` passed the whole suite. There is now
    one code path, so a HomeKit-only device inherits every guarantee automatically.
    """
    parsed = parse_config(render_config_with_extra_device("lhk_hk_only", "hk only", None))
    blocks = facade_blocks(parsed)
    assert "lhk_hk_only" in blocks, "the HomeKit-only device produced no facade"
    blk = blocks["lhk_hk_only"]

    assert not has_cloud(blk)
    assert "availability" not in blk
    assert eval_ha_template(
        blk["state"], trusted_states("lhk_hk_only", ROWS["R_mirror_on_trusted"])
    ) == "True"

    for direction in DIRECTIONS:
        assert len(blk[direction]) == 2, direction
        assert len(parallel_of(blk, direction)) == 1, "a cloud branch was emitted anyway"
        assert boolean_step_of(blk, direction)["action"] == f"input_boolean.{direction}"
        # The full diagnostic apparatus must still be present.
        assert wait_step_of(blk, direction)["timeout"] == "00:00:10"
        assert wait_step_of(blk, direction)["continue_on_timeout"] is True
        assert complain_of(blk, direction)
        assert notification_of(blk, direction)["notification_id"] == (
            f"lhk_hk_only_facade_{TARGET_OF[direction]}_unconfirmed"
        )
        assert dismiss_ids_of(blk, direction) == {
            "lhk_hk_only_facade_on_unconfirmed",
            "lhk_hk_only_facade_off_unconfirmed",
        }


def test_homekit_only_device_emits_no_empty_entity_id() -> None:
    """No ``entity_id`` may render empty, ``null`` or ``'none'``.

    Under HA's own Jinja (``strict=None``, unlike this harness) an undefined ``cloud_entity``
    would render as an empty string, become ``entity_id: null``, and be coerced by
    ``cv.SCRIPT_SCHEMA`` to the string ``'none'`` -- which is ``ENTITY_MATCH_NONE``, so the
    call selects nothing **and logs nothing**. The cloud channel would vanish in silence.
    """
    rendered = render_config_with_extra_device("lhk_hk_only", "hk only", None)
    lines = rendered.splitlines()
    offenders = []
    for index, line in enumerate(lines):
        match = re.match(r"^(\s*)entity_id:\s*(.*)$", line)
        if not match:
            continue
        indent, value = match.group(1), match.group(2).strip()
        if value in ("null", "none", "'none'", '"none"', "~"):
            offenders.append(line.strip())
        elif not value:
            # A bare `entity_id:` is legitimate when a block list follows (several triggers do
            # this); it is an offender only when nothing does. A block sequence may be indented
            # at or beyond its key's column, so accept either.
            following = next((v for v in lines[index + 1 :] if v.strip()), "")
            is_list = following.lstrip().startswith("- ") and (
                len(following) - len(following.lstrip()) >= len(indent)
            )
            if not is_list:
                offenders.append(f"{line.strip()} (no value, no list)")
    assert not offenders, offenders

    # Belt and braces: the parsed form must not contain such a target either.
    for slug, blk in facade_blocks(parse_config(rendered)).items():
        for direction in DIRECTIONS:
            for step in actuation_steps_of(blk, direction):
                target = step["target"]["entity_id"]
                assert target and target not in ("none", "null"), f"{slug}/{direction}"


def test_homekit_only_notification_does_not_claim_two_channels() -> None:
    """The message must not say both channels were driven when there is only one.

    The wording is now reachable with an undriven cloud in three separate cases: the cloud
    write raised (which the ``parallel:`` split made reportable), the ``cloud=`` field was
    removed so the operator cannot tell which case they are in, and a HomeKit-only device has
    no second channel at all.
    """
    hk_only = facade_blocks(
        parse_config(render_config_with_extra_device("lhk_hk_only", "hk only", None))
    )["lhk_hk_only"]
    for direction in DIRECTIONS:
        message = notification_of(hk_only, direction)["message"]
        assert "no cloud channel" in message, message
        assert "Netatmo" not in message, message
        assert "Both channels" not in message, message


def test_cloud_backed_notification_names_both_channels(
    blocks: dict[str, dict[str, Any]],
) -> None:
    """A cloud-backed device's message says which channels were commanded."""
    for slug, blk in blocks.items():
        for direction in DIRECTIONS:
            message = notification_of(blk, direction)["message"]
            assert "Netatmo cloud" in message, f"{slug}/{direction}"
            assert "no cloud channel" not in message, f"{slug}/{direction}"
            assert "Both channels" not in message, f"{slug}/{direction}"


# ============================================================================
# Name escaping (plan #03 B1)
# ============================================================================


@pytest.mark.parametrize(
    "hostile_name",
    [
        'po"ol: pump',  # double quote inside a quoted scalar, plus a colon
        "pump: {weird}",  # leading brace after a colon
        "a'b",  # single quote
        "*anchor & alias",  # YAML indicators
        "trailing backslash \\",
    ],
)
def test_hostile_device_name_still_renders_valid_yaml(hostile_name: str) -> None:
    """Every ``s.name`` interpolation is escaped, so no name can break the render.

    There are seven interpolation sites: the two ``input_boolean`` helper names, the facade
    ``name:``, the two notification titles, and the two HomeKit ``entity_config`` names. A
    malformed render fails the load of **all** of HA's startup, not just the facade -- so a
    single unescaped site is a whole-instance outage triggered by a rename.
    """
    rendered = render_config_with_extra_device("lhk_hostile", hostile_name, "switch.hostile")
    parsed = parse_config(rendered)  # must not raise
    assert "lhk_hostile" in facade_blocks(parsed)
    assert parsed["input_boolean"]["lhk_hostile_state"]["name"] == (
        f"{hostile_name} - HK Mirror (read)"
    )
    assert parsed["input_boolean"]["lhk_hostile_command"]["name"] == (
        f"{hostile_name} - HK Command (write)"
    )
    entity_config = parsed["homekit"][0]["entity_config"]
    assert entity_config["input_boolean.lhk_hostile_state"]["name"] == f"LHK {hostile_name} Mirror"
    assert entity_config["input_boolean.lhk_hostile_command"]["name"] == f"LHK {hostile_name} Cmd"
    assert facade_blocks(parsed)["lhk_hostile"]["name"] == hostile_name


def test_state_stub_raises_on_unknown_entity() -> None:
    """The layer-3 accessor raises on an unstubbed entity id.

    This is what makes ``trusted_states``' omission of the cloud entity load-bearing: a
    template that reads it fails loudly instead of silently passing.
    """
    with pytest.raises(KeyError, match="unstubbed entity"):
        eval_ha_template("{{ states('sensor.never_stubbed') }}", {})
