---
story_key: "fix-58"
title: "Remove forecast sensors from home dashboard"
issue: 58
branch: "QS_58"
type: fix
status: ready
---

# Fix: Remove forecast sensors from home dashboard

## Description

Remove the `qs_no_control_forecast_XX` and `qs_solar_forecast_XX` sensors from the **home device** section of both dashboard Jinja templates. The solar device section must NOT be touched.

## Files to modify

- `custom_components/quiet_solar/ui/quiet_solar_dashboard_template.yaml.j2` — lines 218-223
- `custom_components/quiet_solar/ui/quiet_solar_dashboard_template_standard_ha.yaml.j2` — lines 420-425

## Tasks

- [ ] 1. Remove the for-loop block in the home device section of `quiet_solar_dashboard_template.yaml.j2` (the block that iterates `device.ha_entities.items()` and filters on `qs_no_control_forecast_` or `qs_solar_forecast_`)
- [ ] 2. Remove the same for-loop block in the home device section of `quiet_solar_dashboard_template_standard_ha.yaml.j2`
- [ ] 3. Update dashboard rendering tests/snapshots if affected

## Acceptance Criteria

- [ ] Neither `qs_no_control_forecast_*` nor `qs_solar_forecast_*` entities appear in the home device dashboard section
- [ ] The solar device section remains unchanged (still shows `qs_solar_forecast_ok`, `qs_solar_forecast_age`, `qs_solar_forecast_score_*`, `qs_solar_active_provider`)
- [ ] All quality gates pass
