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

### 1b. Create Branch (Worktree by Default)

Branch naming convention: `QS_{{github_issue_number}}`

**Default — worktree** (used unless the user says "no worktree"):

```bash
git checkout main && git pull
bash scripts/worktree-setup.sh {{github_issue_number}}
```

This creates a worktree at `../<repo>-worktrees/QS_{{github_issue_number}}/` with:
- Branch `QS_{{github_issue_number}}`
- Symlinked `venv/` (shared with main — avoids ~GB duplicate)
- Symlinked `config/` (shared HA runtime config)
- Symlinked non-git `custom_components/*` (hacs, netatmo, etc.)

After setup, the agent works entirely inside the worktree directory. The main worktree stays on `main`.

**Opt-out — "no worktree"** (when the user explicitly asks):

```bash
git checkout main && git pull && git checkout -b QS_{{github_issue_number}}
```

The agent works in the main directory as before.

Examples:
- Issue #10 → worktree at `../quiet-solar-worktrees/QS_10/` (default)
- Issue #42, user says "no worktree" → branch `QS_42` in main dir

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

## Phase 3d: Code Review

After the PR is created, run `/bmad-code-review` on the story before considering it done.

- For best results, use a **different LLM** than the one that implemented the story (e.g., if Opus implemented, review with Sonnet or vice versa)
- The review catches blind spots the implementing agent missed
- This applies to **all** stories — even config-only or documentation stories

The agent MUST propose this step to the user after every dev-story completion.

---

## Phase 3e: Merge PR

Merging is **manually triggered** by the user — never auto-merge.

When the user asks to merge a PR:

```bash
gh pr merge {{pr_number}} --merge --delete-branch
```

- Uses merge commit (no squash, no rebase)
- Deletes the remote branch after merge
- Do NOT merge unless the user explicitly asks
- Do NOT merge until code review (Phase 3d) has been offered/completed

### 3f. Worktree Cleanup (after merge)

If a worktree was used for this story, clean it up after merge:

```bash
bash scripts/worktree-cleanup.sh {{github_issue_number}}
```

This removes the worktree directory and its symlinks. The targets (main venv, config, other custom_components) are NOT affected.

Then update main in the main worktree:

```bash
cd {{main_worktree_path}}
git checkout main && git pull
```

Skip this step if the story used "no worktree" mode.

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

2. **Update `manifest.json` version** to match the tag (without the `v` prefix):

```bash
VERSION="${TAG#v}"
sed -i '' "s/\"version\": \".*\"/\"version\": \"${VERSION}\"/" custom_components/quiet_solar/manifest.json
```

3. **Commit the version bump and push the tag:**

```bash
git add custom_components/quiet_solar/manifest.json
git commit -m "bump version to ${VERSION}"
git push origin main
git tag "$TAG"
git push origin "$TAG"
```

4. **GitHub Actions handles the rest.** The release pipeline (`.github/workflows/release.yml`) triggers on the tag push and:
   - Runs the full test suite with 100% coverage
   - Runs Ruff lint + format check and MyPy type check
   - Validates HACS compatibility
   - Checks that the tag version matches `manifest.json`
   - Creates the GitHub Release with auto-generated changelog

If any check fails, no release is created — fix the issue and re-tag.

---

## Risk Categories

Determine from changed files:

| Category | Files |
|----------|-------|
| CRITICAL | `solver.py`, `constraints.py`, `charger.py`, `dynamic_group.py` |
| HIGH | `load.py`, `const.py`, `ha_model/home.py`, `ha_model/device.py` |
| MEDIUM | `ha_model/person.py`, `ha_model/car.py`, `ha_model/battery.py`, `ha_model/solar.py`, `config_flow.py` |
| LOW | `sensor.py`, `switch.py`, `number.py`, `select.py`, `button.py`, `ui/` |

---

## Appendix: Worktree Reference

### Directory Layout

```
~/Developer/homeassistant/
  <repo>/                               # main worktree (always on main)
    venv/                               # real venv (~GB, installed once)
    config/                             # real HA config (runtime state, not in git)
    scripts/                            # worktree helper scripts
    custom_components/
      quiet_solar/                      # IN GIT — the project code
      hacs/                             # NOT in git — installed separately
      netatmo/                          # NOT in git
      ...
  <repo>-worktrees/                     # auto-created by worktree-setup.sh
    QS_42/                              # worktree for issue #42
      venv -> ../../<repo>/venv
      config -> ../../<repo>/config
      custom_components/
        quiet_solar/                    # IN GIT — this branch's version
        hacs -> ../../../<repo>/custom_components/hacs
        ...
```

The scripts derive `<repo>` from the main worktree's directory name automatically — no hardcoded paths.

### What Gets Symlinked and Why

| Item | Why | Needed for tests? | Needed for HA run? |
|------|-----|-------------------|---------------------|
| `venv/` | Avoid ~GB duplicate install | YES | YES |
| `config/` | Share HA runtime config | NO (tests use FakeHass) | YES |
| Non-git `custom_components/*` | Other integrations HA loads | NO | YES |

### Caveats

1. **No simultaneous HA runs**: Two HA instances cannot share the same `config/` (database locks). Run HA from one worktree at a time.
2. **Shared venv = shared deps**: `pip install` in one worktree affects all. Fine for normal dev; only matters if experimenting with different dep versions. If a branch needs different deps, create a real venv: `python3.14 -m venv venv && source venv/bin/activate && pip install -r requirements.txt -r requirements_test.txt`
3. **Quality gates work unchanged**: `source venv/bin/activate` resolves the symlink. pytest, ruff, mypy all run against the worktree's code, not the main worktree's.
