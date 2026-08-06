"""QS-332 (B3): the 6 lane issue forms, one per lane, exact labels.

GitHub issue forms attach FIXED labels per template — a dropdown cannot
change labels — so exact lane labels by construction require one form
per lane (story D3, path 1).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = REPO_ROOT / ".github" / "ISSUE_TEMPLATE"

# The D3 table: form basename → exact fixed label set.
#
# Review-fix #03 (user decision) amends D3: the legacy `bug` /
# `enhancement` labels are NO LONGER attached. They created a label-set
# asymmetry between the two birth paths — the web forms emitted them
# while the agent path (`--labels "kind:bug,target:product,scale:task"`)
# did not, so the same lane produced two different label sets and skewed
# any filter keyed on the plain labels. Both paths now produce the
# axis-only set. Their only code consumer (`story_type`) was deleted by
# this same task.
EXPECTED_LABELS = {
    "bug-product": ["kind:bug", "target:product", "scale:task"],
    "feature-product": ["kind:feature", "target:product", "scale:task"],
    "bug-factory": ["kind:bug", "target:factory", "scale:task", "area:dev-pipeline"],
    "feature-factory": [
        "kind:feature",
        "target:factory",
        "scale:task",
        "area:dev-pipeline",
    ],
    "epic-product": ["scale:epic", "target:product"],
    "epic-factory": ["scale:epic", "target:factory", "area:dev-pipeline"],
}

# Review-fix #03: pinned as GONE from every form, not merely absent from
# the table above — a future edit re-adding one must fail loudly.
LEGACY_LABELS = ("bug", "enhancement")

TASK_FORMS = ("bug-product", "feature-product", "bug-factory", "feature-factory")
EPIC_FORMS = ("epic-product", "epic-factory")


def _load(form: str) -> dict:
    return yaml.safe_load((TEMPLATE_DIR / f"{form}.yml").read_text(encoding="utf-8"))


def _field_labels(form: str) -> list[str]:
    return [
        item["attributes"].get("label", "")
        for item in _load(form)["body"]
    ]


def test_exactly_the_six_lane_forms_plus_config_exist() -> None:
    names = sorted(p.name for p in TEMPLATE_DIR.iterdir() if p.suffix == ".yml")
    assert names == sorted([*(f"{f}.yml" for f in EXPECTED_LABELS), "config.yml"])


def test_legacy_templates_are_gone() -> None:
    assert not (TEMPLATE_DIR / "bug_report.yml").exists()
    assert not (TEMPLATE_DIR / "feature_request.yml").exists()


@pytest.mark.parametrize("form", sorted(EXPECTED_LABELS))
def test_form_carries_exactly_its_lane_labels(form: str) -> None:
    assert _load(form)["labels"] == EXPECTED_LABELS[form]


@pytest.mark.parametrize("form", sorted(EXPECTED_LABELS))
def test_no_form_attaches_a_legacy_label(form: str) -> None:
    """Review-fix #03: both birth paths must produce the SAME label set
    for a lane — the web forms no longer attach `bug`/`enhancement`."""
    attached = _load(form)["labels"]
    assert not set(LEGACY_LABELS) & set(attached), (
        f"{form}.yml still attaches a legacy label: "
        f"{sorted(set(LEGACY_LABELS) & set(attached))}. The agent birth "
        f"path emits axis labels only; the two paths must not disagree."
    )


@pytest.mark.parametrize("form", TASK_FORMS)
def test_task_forms_have_the_parent_epic_field(form: str) -> None:
    """The field label must be exactly "Parent epic": it renders as the
    `### Parent epic` heading `targets.parse_parent_epic` keys on."""
    assert "Parent epic" in _field_labels(form)
    # Optional by design — "leave empty if none".
    field = next(
        item
        for item in _load(form)["body"]
        if item["attributes"].get("label") == "Parent epic"
    )
    assert field.get("validations", {}).get("required") is not True


@pytest.mark.parametrize("form", EPIC_FORMS)
def test_epic_forms_have_no_parent_epic_field(form: str) -> None:
    """No epic nesting — and no kind label either (checked via the table)."""
    assert "Parent epic" not in _field_labels(form)


def test_blank_issues_are_disabled() -> None:
    config = yaml.safe_load((TEMPLATE_DIR / "config.yml").read_text(encoding="utf-8"))
    assert config["blank_issues_enabled"] is False
