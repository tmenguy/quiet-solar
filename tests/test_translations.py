"""Direct fixture tests for `strings.json` translation blocks.

These tests assert the structure of `custom_components/quiet_solar/strings.json`
directly, without going through HA's translation lookup. They tighten
contracts that would otherwise be verified only transitively (via
`get_select_translation_key()` + the quality-gate translations check).
"""

from __future__ import annotations

import json
from pathlib import Path

STRINGS_JSON = (
    Path(__file__).parent.parent
    / "custom_components"
    / "quiet_solar"
    / "strings.json"
)


def _load_strings() -> dict:
    with STRINGS_JSON.open(encoding="utf-8") as f:
        return json.load(f)


def test_water_boiler_mode_block() -> None:
    """`entity.select.water_boiler_mode` mirrors `on_off_mode` per AC-4b.

    Specifically:
    - Top-level `name` is "Water Boiler Mode".
    - `state` has the same 5 keys as `on_off_mode.state`.
    - Each state value matches the corresponding `on_off_mode` value
      (the user-visible label set is intentionally shared between the
      two modes — only the select's translation key is customised).
    """
    strings = _load_strings()
    select = strings["entity"]["select"]
    assert "water_boiler_mode" in select, (
        "entity.select.water_boiler_mode is missing from strings.json — "
        "AC-4b requires this block per the QS-194 story."
    )
    water_boiler_mode = select["water_boiler_mode"]
    on_off_mode = select["on_off_mode"]

    assert water_boiler_mode["name"] == "Water Boiler Mode"

    assert set(water_boiler_mode["state"].keys()) == set(on_off_mode["state"].keys()), (
        f"water_boiler_mode state keys must match on_off_mode; "
        f"diff: {set(water_boiler_mode['state']) ^ set(on_off_mode['state'])}"
    )

    for state_key, on_off_value in on_off_mode["state"].items():
        assert water_boiler_mode["state"][state_key] == on_off_value, (
            f"water_boiler_mode.state.{state_key!r} should match "
            f"on_off_mode.state.{state_key!r} ({on_off_value!r}); "
            f"got {water_boiler_mode['state'][state_key]!r}"
        )
