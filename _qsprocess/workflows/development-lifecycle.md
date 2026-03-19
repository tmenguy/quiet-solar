# Quiet Solar — Development Lifecycle

This document defines the project-specific development lifecycle that augments the standard BMad `create-story` → `dev-story` pipeline. Every agent working on this project MUST follow these rules.

## Prerequisites

Before starting any development work:

1. `gh` CLI installed and authenticated (`gh auth status`)
2. Python venv active (`source venv/bin/activate`)
3. On `main` branch, up to date (`git checkout main && git pull`)

---

## Phase 1: Story Setup (augments `bmad-create-story`)

After `bmad-create-story` produces the story file, and BEFORE starting `bmad-dev-story`:

### 1a. Create GitHub Issue

```bash
gh issue create --title "{{story_title}}" --body "$(cat <<'EOF'
## Story {{story_key}}

{{story_description_from_story_file}}

---
Created from BMad story: {{story_file_path}}
EOF
)"
```

Extract the issue number from the output.

### 1b. Create Branch

Branch naming convention: `QS_{{github_issue_number}}`

```bash
git checkout main && git pull && git checkout -b QS_{{github_issue_number}}
```

Examples:
- Issue #10 → branch `QS_10`
- Issue #42 → branch `QS_42`

---

## Phase 2: Development (augments `bmad-dev-story` steps 5-8)

### Quality Gates — Mandatory After Every Task

The BMad `dev-story` step 7 says "run linting and code quality checks if configured." For this project, the specific commands are:

```bash
source venv/bin/activate

# 1. Tests with 100% coverage (MANDATORY — zero tolerance)
pytest tests/ --cov=custom_components/quiet_solar --cov-report=term-missing

# 2. Ruff lint (zero violations)
ruff check custom_components/quiet_solar/

# 3. Ruff format (all files formatted)
ruff format --check custom_components/quiet_solar/

# 4. MyPy type check (no issues)
mypy custom_components/quiet_solar/
```

ALL FOUR must pass before marking any task complete.

### 100% Coverage Enforcement

- Every code change MUST have corresponding test additions or updates
- If coverage drops below 100%, identify uncovered lines from the `--cov-report=term-missing` output
- Write tests to cover those lines before proceeding
- Repeat until 100% is achieved
- This is NOT optional — a task is NOT complete until coverage is 100%

### Test-First Development

Follow the red-green-refactor cycle from `bmad-dev-story` step 5:
1. Write failing tests for the new behavior
2. Implement minimal code to make tests pass
3. Refactor while keeping tests green
4. Run full quality gates

---

## Phase 3: Completion (augments `bmad-dev-story` step 9)

After all tasks are complete and `bmad-dev-story` marks the story as "review":

### 3a. Final Quality Gate

Run the full suite one last time to confirm:

```bash
source venv/bin/activate
pytest tests/ --cov=custom_components/quiet_solar --cov-report=term-missing
ruff check custom_components/quiet_solar/
ruff format --check custom_components/quiet_solar/
mypy custom_components/quiet_solar/
```

ALL must pass. If any fail, fix before proceeding.

### 3b. Commit Changes

Stage and commit all relevant files. Do NOT commit:
- `.DS_Store`, `.env`, credentials, IDE configs
- `venv/`, `config/`, `__pycache__/`

### 3c. Push and Create PR

```bash
git push -u origin QS_{{github_issue_number}}
```

Create the PR:

```bash
gh pr create --title "{{pr_title}}" --body "$(cat <<'EOF'
## Summary
{{one_to_three_bullet_summary}}

Fixes #{{github_issue_number}}

## Testing
- [x] Tests added/updated for new behavior
- [x] 100% coverage verified
- [x] No flaky tests introduced

## Code quality
- [x] Ruff passes (lint + format)
- [x] MyPy passes
- [x] No new `# type: ignore` or `noqa` without justification

## Risk assessment
- [ ] CRITICAL (solver, constraints, charger budgeting)
- [ ] HIGH (load base, constants, orchestration)
- [ ] MEDIUM (device-specific: car, person, battery, solar)
- [ ] LOW (platforms, UI, docs)

Mark the applicable risk level(s) with [x].
EOF
)"
```

PR title should be under 70 characters.

---

## Phase 3d: Merge PR

Merging is **manually triggered** by the user — never auto-merge.

When the user asks to merge a PR:

```bash
gh pr merge {{pr_number}} --merge --delete-branch
```

- Uses merge commit (no squash, no rebase)
- Deletes the remote branch after merge
- Do NOT merge unless the user explicitly asks

---

## Phase 4: Release

Releases are **manually triggered** by the user — never auto-release.

### Tag Format

`vYYYY.MM.DD.XX` where:
- `YYYY.MM.DD` is the release date
- `XX` is a zero-based index that increments for each release on the same day

Examples: `v2026.03.19.0` (first release on March 19), `v2026.03.19.1` (second release same day)

### Release Process

When the user asks to create a release:

1. **Determine the tag.** Find the latest existing tag for today and increment, or start at `.0`:

```bash
TODAY=$(date +%Y.%m.%d)
LAST=$(gh release list --limit 100 | grep "v${TODAY}" | head -1 | awk '{print $1}')
if [ -z "$LAST" ]; then
  TAG="v${TODAY}.0"
else
  INDEX=$(echo "$LAST" | sed "s/v${TODAY}\.//")
  TAG="v${TODAY}.$((INDEX + 1))"
fi
echo "Next tag: $TAG"
```

2. **Build the release notes.** Collect all merged PRs since the last release:

```bash
PREV_TAG=$(gh release list --limit 1 | awk '{print $1}')
gh pr list --state merged --search "merged:>=$(gh release view "$PREV_TAG" --json publishedAt -q .publishedAt | cut -dT -f1)" --json number,title --jq '.[] | "- #\(.number): \(.title)"'
```

3. **Create the release:**

```bash
gh release create "$TAG" --title "$TAG" --notes "$(cat <<EOF
## Changes since ${PREV_TAG}

{{merged_pr_list_from_step_2}}
EOF
)" --target main
```

The release title is the tag only. The description lists all merged PRs (which reference their issues via `Fixes #N`).

---

## Risk Categories

Determine from changed files:

| Category | Files |
|----------|-------|
| CRITICAL | `solver.py`, `constraints.py`, `charger.py`, `dynamic_group.py` |
| HIGH | `load.py`, `const.py`, `ha_model/home.py`, `ha_model/device.py` |
| MEDIUM | `ha_model/person.py`, `ha_model/car.py`, `ha_model/battery.py`, `ha_model/solar.py`, `config_flow.py` |
| LOW | `sensor.py`, `switch.py`, `number.py`, `select.py`, `button.py`, `ui/` |
