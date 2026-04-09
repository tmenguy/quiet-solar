# Story feature-Github-#128-confirmation-popup-home-solar: Add confirmation popup for home and solar buttons in dashboard

issue: 128
branch: "QS_128"

Status: dev-complete

## Story
As a Quiet Solar user,
I want confirmation popups on home and solar action buttons in the dashboard,
so that I don't accidentally trigger destructive actions like history resets or forecast recomputations.

## Acceptance Criteria
1. Given a user viewing the dashboard, when they tap any home action button (serialize_for_debug, light_reset_history, recompute_people_historical_data, reset_history, generate_yaml_dashboard), then a confirmation popup appears with action-specific text before the action executes.
2. Given a user viewing the dashboard, when they tap any solar action button (recompute_forecast_scores, compute_dampening_1day, compute_dampening_7day, reset_dampening), then a confirmation popup appears with action-specific text before the action executes.
3. Given both template files (quiet_solar_dashboard_template.yaml.j2 and quiet_solar_dashboard_template_standard_ha.yaml.j2), when rendered, then the confirmation blocks follow the same tap_action/confirmation pattern used in the car section.
4. Given the rendered dashboard YAML, when parsed, then `confirmation:` text appears for all 9 action buttons in both templates.

## Tasks / Subtasks
- [x] Task 1: Add confirmation blocks to home buttons in BOTH templates (AC: #1, #3)
  - [x] qs_home_serialize_for_debug — standard_ha:395-399, custom:199-203
  - [x] qs_home_light_reset_history — standard_ha:400-404, custom:204-208
  - [x] qs_home_recompute_people_historical_data — standard_ha:405-409, custom:209-213
  - [x] qs_home_reset_history — standard_ha:410-414, custom:214-218
  - [x] qs_home_generate_yaml_dashboard — standard_ha:415-419, custom:219-223
- [x] Task 2: Add confirmation blocks to solar buttons in BOTH templates (AC: #2, #3)
  - [x] qs_solar_recompute_forecast_scores — standard_ha:465-469, custom:269-273
  - [x] qs_solar_compute_dampening_1day — standard_ha:470-474, custom:274-278
  - [x] qs_solar_compute_dampening_7day — standard_ha:475-479, custom:279-283
  - [x] qs_solar_reset_dampening — standard_ha:480-484, custom:284-288
- [x] Task 3: Add regression test for confirmation presence in rendered output (AC: #4)
- [x] Task 4: Run quality gates and verify all tests pass

## Dev Notes

### Insertion pattern

Insert 4 lines after the `name` line and before the `{%- endif %}` of each entity block. Example before/after for `qs_home_serialize_for_debug`:

**Before:**
```jinja2
              {%- set ha_entity = device.ha_entities.get("qs_home_serialize_for_debug") %}
              {%- if  ha_entity != None %}
              {{ "- entity: " + ha_entity.entity_id }}
              {{ "  name: " + "\'" + ha_entity.name + "\'" }}
              {%- endif %}
```

**After:**
```jinja2
              {%- set ha_entity = device.ha_entities.get("qs_home_serialize_for_debug") %}
              {%- if  ha_entity != None %}
              {{ "- entity: " + ha_entity.entity_id }}
              {{ "  name: " + "\'" + ha_entity.name + "\'" }}
              {{ "  tap_action:" }}
              {{ "    action: toggle" }}
              {{ "  confirmation:" }}
              {{ "    text: This will export home data to a debug file." }}
              {%- endif %}
```

Apply the same 4-line insertion to all 9 entity blocks in both files, using the confirmation text from the table below.

### Indentation

The tap_action and confirmation blocks must be indented at the same level as the entity's `name` property:
- `{{ "  tap_action:" }}` — 2-space indent (matches `name`)
- `{{ "    action: toggle" }}` — 4-space indent
- `{{ "  confirmation:" }}` — 2-space indent
- `{{ "    text: ..." }}` — 4-space indent

### No icon needed

The car section entities include custom `icon:` lines (e.g., `mdi:rabbit`). Home/solar action buttons do not need custom icons — only the `tap_action` + `confirmation` block is required.

### Template differences

The custom template (`quiet_solar_dashboard_template.yaml.j2`) has no existing confirmation pattern to reference — car/pool/climate use custom JS cards there. However, the home and solar sections in both templates use the identical `{{ "- entity: ..." }}` / `{{ "  name: ..." }}` Jinja2 string rendering, so the same tap_action/confirmation block applies directly.

### Confirmation Text per Button

| Entity key | Confirmation text |
|------------|-------------------|
| qs_home_serialize_for_debug | This will export home data to a debug file. |
| qs_home_light_reset_history | This will clear recent home history (keeps older data). |
| qs_home_recompute_people_historical_data | This will recompute all people historical data. |
| qs_home_reset_history | WARNING: This will permanently delete ALL home history data. |
| qs_home_generate_yaml_dashboard | This will regenerate the YAML dashboard. |
| qs_solar_recompute_forecast_scores | This will recompute all solar forecast scores. |
| qs_solar_compute_dampening_1day | This will compute solar dampening from 1 day of data. |
| qs_solar_compute_dampening_7day | This will compute solar dampening from 7 days of data. |
| qs_solar_reset_dampening | WARNING: This will reset ALL solar dampening values. |

### Scope note

This story is intentionally scoped to home and solar buttons per issue #128. Other device sections (climate, pool, on_off_duration) also have action buttons (e.g., `qs_device_clean_and_reset`) without confirmations — those are out of scope for this change.

### Test requirements

Add a regression test in `tests/test_dashboard_rendering.py` that renders both templates and asserts `confirmation:` text appears for all 9 action buttons. This guards against silent removal of confirmation blocks.

### Files to modify
- `custom_components/quiet_solar/ui/quiet_solar_dashboard_template_standard_ha.yaml.j2` — home section (lines 395-419), solar section (lines 465-484)
- `custom_components/quiet_solar/ui/quiet_solar_dashboard_template.yaml.j2` — home section (lines 199-223), solar section (lines 269-288)
- `tests/test_dashboard_rendering.py` — add confirmation presence test

### General notes
- Template changes only affect NEW dashboard generation — existing users must regenerate dashboards
- No Python source code changes needed — pure template + test modification

## Adversarial Review Notes

**Reviewers:** Critic, Concrete Planner, Dev Proxy
**Rounds:** 1

### Key findings incorporated:
- Task structure: merged "mirror" Task 3 into Tasks 1+2 so both files are edited together per button group (Critic + Concrete Planner)
- Added concrete before/after example to eliminate insertion ambiguity (Critic + Concrete Planner)
- Added dev notes on exact indentation, insert position, icon exclusion, template differences (Dev Proxy + Concrete Planner)
- Added regression test task for confirmation presence (all 3 reviewers)
- Rewrote confirmation texts in plain language with severity hints for destructive actions (Critic)
- Added scope note acknowledging other device sections are intentionally excluded (Critic)

### Decisions made:
- Plain language confirmation texts — Rationale: user chose user-friendly wording over technical descriptions
- All 9 action buttons get confirmations — Rationale: user confirmed all action buttons, not just destructive ones
- Scoped to home/solar only — Rationale: matches issue #128 scope; other sections can be a follow-up

### Known risks acknowledged:
- Template changes don't propagate to existing dashboards — users must regenerate
- HA Lovelace parser validation is not tested at runtime level (only YAML structure is validated by tests)
