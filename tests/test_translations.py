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
TRANSLATIONS_EN_JSON = (
    Path(__file__).parent.parent
    / "custom_components"
    / "quiet_solar"
    / "translations"
    / "en.json"
)


def _load_strings() -> dict:
    with STRINGS_JSON.open(encoding="utf-8") as f:
        return json.load(f)


def _load_translations_en() -> dict:
    with TRANSLATIONS_EN_JSON.open(encoding="utf-8") as f:
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


def _assert_same_key_structure(strings: dict, en: dict, path: str = "") -> None:
    """Recursively assert two translation trees share the same key structure.

    Asserts the leaf-key sets match. Treats values:
    - Both ``dict``: recurse into matching keys.
    - One ``dict`` and the other not: structural mismatch — fail.
    - Both non-``dict``: leaf — values may differ (one is a key-reference
      placeholder like ``[%key:...%]`` and the other is the resolved
      translation string).
    """
    assert isinstance(strings, dict), f"{path}: expected dict in strings.json"
    assert isinstance(en, dict), f"{path}: expected dict in en.json"

    strings_keys = set(strings.keys())
    en_keys = set(en.keys())
    missing_in_en = strings_keys - en_keys
    missing_in_strings = en_keys - strings_keys
    assert not missing_in_en, (
        f"{path}: keys present in strings.json but missing in en.json: "
        f"{sorted(missing_in_en)}"
    )
    assert not missing_in_strings, (
        f"{path}: keys present in en.json but missing in strings.json: "
        f"{sorted(missing_in_strings)}"
    )

    for key in strings_keys:
        sub_strings = strings[key]
        sub_en = en[key]
        if isinstance(sub_strings, dict) or isinstance(sub_en, dict):
            assert isinstance(sub_strings, dict) and isinstance(sub_en, dict), (
                f"{path}.{key}: structural mismatch — one side is a dict, "
                f"the other is a leaf"
            )
            _assert_same_key_structure(sub_strings, sub_en, f"{path}.{key}")
        # else: both leaves — values may differ (key-ref placeholder vs.
        # resolved translation). Nothing to assert here.


def test_translations_en_matches_strings_structure() -> None:
    """`translations/en.json` shape matches `strings.json` shape (AC-6b).

    The generator script (`bash scripts/generate-translations.sh`)
    resolves `[%key:...%]` references but must preserve the key set. A
    drift between the two files would mean `en.json` is stale relative
    to `strings.json` — e.g., a developer edited `strings.json` and
    forgot to regenerate. The quality-gate translations check catches
    *missing* keys; this direct test catches *extra/orphaned* keys in
    either direction.
    """
    strings = _load_strings()
    en = _load_translations_en()
    _assert_same_key_structure(strings, en)
