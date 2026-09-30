import json
from pathlib import Path


def test_wallbox_fixture_reports_charging() -> None:
    data = json.loads((Path(__file__).parent / "fixtures" / "charger_state_wallbox.json").read_text())
    assert data["state"] == "charging"
