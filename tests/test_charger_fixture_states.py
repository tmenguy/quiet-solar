import json
from pathlib import Path


def test_wallbox_fixture_reports_charging() -> None:
    data = json.loads((Path(__file__).parent / "fixtures" / "Charger_State_Wallbox.json").read_text())
    assert data["state"] == "charging"
