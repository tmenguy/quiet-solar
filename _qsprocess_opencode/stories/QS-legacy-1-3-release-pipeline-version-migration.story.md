# Story 1.3: Release Pipeline & Version Migration

Status: done

## Story

As TheDev,
I want pushing a version tag to automatically run the full test suite, HACS validation, tag/manifest version consistency check, and create a GitHub Release with auto-generated changelog,
So that shipping a release requires zero manual steps beyond pushing a tag.

## Acceptance Criteria

1. **Given** a version tag (`v*`) is pushed to main
   **When** the release pipeline triggers
   **Then** the full test suite runs with 100% coverage as a safety gate
   **And** Ruff lint + format check passes
   **And** MyPy type check passes

2. **Given** the quality gate passes
   **When** HACS validation runs
   **Then** `hacs/action@main` validates manifest.json, directory structure, and HACS compatibility

3. **Given** the tag is `v2026.03.19.0`
   **When** the version consistency check runs
   **Then** the tag version (without `v` prefix) is compared to `manifest.json` `"version"` field
   **And** the pipeline fails if they don't match

4. **Given** all checks pass
   **When** the release job runs
   **Then** a GitHub Release is created with the tag as title
   **And** the release body contains auto-generated changelog (commits or merged PRs since previous tag)

5. **Given** any check fails (tests, HACS, version mismatch)
   **When** the pipeline reports failure
   **Then** no GitHub Release is created
   **And** the failure reason is clearly reported in the Actions log

6. **Given** the release pipeline exists
   **When** `_qsprocess/workflows/development-lifecycle.md` Phase 4 is updated
   **Then** the manual release flow is replaced with: update `manifest.json` version, push tag, pipeline handles the rest
   **And** the tag format `vYYYY.MM.DD.XX` is documented

## Tasks / Subtasks

- [x] Task 1: Create `.github/workflows/release.yml` (AC: #1, #2, #3, #4, #5)
  - [x] 1.1 Trigger on `push: tags: ['v*']`
  - [x] 1.2 Job `quality-gate`: checkout, setup Python 3.14, install deps from `requirements.txt` + `requirements_test.txt`, run pytest with 100% coverage, ruff check + format, mypy
  - [x] 1.3 Job `hacs-validate`: run `hacs/action@main` (needs: quality-gate)
  - [x] 1.4 Job `version-check`: extract tag version (strip `v` prefix), read `manifest.json` version, compare, fail if mismatch (needs: quality-gate)
  - [x] 1.5 Job `release`: create GitHub Release with tag as title, auto-generated notes via `gh release create` (needs: quality-gate, hacs-validate, version-check)
- [x] Task 2: Sync `manifest.json` version to current release (AC: #3)
  - [x] 2.1 Update `manifest.json` `"version"` from `"2025.09.11"` to `"2026.03.19.0"` (matches current latest tag)
- [x] Task 3: Update `_qsprocess/workflows/development-lifecycle.md` Phase 4 (AC: #6)
  - [x] 3.1 Replace manual `gh release create` flow with: update `manifest.json` version, commit, push tag, pipeline handles release creation
  - [x] 3.2 Keep tag format `vYYYY.MM.DD.XX` documented
  - [x] 3.3 Document that pipeline creates the release — developer only pushes the tag

## Dev Notes

### Critical: manifest.json version is stale

`custom_components/quiet_solar/manifest.json` currently has `"version": "2025.09.11"`. The latest release is `v2026.03.19.0`. Task 2 must sync this. Going forward, the version-check job ensures they stay in sync.

### Version format

- Tag format: `vYYYY.MM.DD.XX` (e.g., `v2026.03.19.0`)
- manifest.json format: `YYYY.MM.DD.XX` (no `v` prefix, e.g., `2026.03.19.0`)
- The version-check job strips the `v` prefix from the tag before comparing

### GitHub Actions workflow structure

Architecture specifies Tier 3 in `.github/workflows/release.yml` [Source: architecture.md#CI/CD Pipeline Architecture, lines 489-497]:
- Trigger: `on: push` (tags: `v*`)
- Jobs: test (full suite), validate (HACS), release (GitHub Release), version-check (tag vs manifest)

### Python and dependency setup

- Python 3.14 (matches `pyproject.toml` target and HA 2026.2.1+)
- Install: `pip install -r requirements.txt -r requirements_test.txt`
- numpy/scipy must match HA's bundled versions (AR2) — `requirements.txt` already pins `numpy>=1.24.0`, `requirements_test.txt` has `scipy>=1.11.0`

### HACS validation

Use `hacs/action@main` — validates manifest.json structure, directory layout, HACS compatibility. This is a standard GitHub Action provided by HACS.

### Release notes generation

Use GitHub's built-in `--generate-notes` flag on `gh release create`, or `actions/create-release` with `generate_release_notes: true`. This generates changelog from commits/PRs since previous tag.

### Existing manual flow to update

`_qsprocess/workflows/development-lifecycle.md` Phase 4 (lines 169-219) currently has a manual flow:
1. Determine tag
2. Build release notes from merged PRs via `gh` CLI
3. Create release via `gh release create`

The new flow becomes:
1. Update `manifest.json` version to match planned tag
2. Commit the version bump
3. Push the tag: `git tag vYYYY.MM.DD.XX && git push origin vYYYY.MM.DD.XX`
4. Pipeline handles: tests, HACS validation, version check, release creation

### What NOT to do

- Do NOT create Tier 1 (PR Quality Gate) or Tier 2 (PR Merge Gate) workflows — those are Story 1.2
- Do NOT create PR templates, CODEOWNERS, labeler.yml, or issue templates — those are Story 1.4
- Do NOT add auto-review, issue-triage, or stale workflows — those are Story 1.4
- Do NOT modify any production Python code in `custom_components/quiet_solar/` (except `manifest.json` version)

### Previous story intelligence (Story 1.1)

- `pyproject.toml` already has Ruff + MyPy config — CI jobs should use the same config
- `requirements_test.txt` already has ruff>=0.9.0 and mypy>=1.13.0
- Quality gate commands are documented in `_qsprocess/rules/project-rules.md` lines 11-29
- The `_qsprocess/` directory is the canonical location for project process docs (shared between Claude and Cursor)

### Project Structure Notes

- `.github/workflows/release.yml` — new file, standard GitHub Actions location
- `custom_components/quiet_solar/manifest.json` — existing, version field update only
- `_qsprocess/workflows/development-lifecycle.md` — existing, Phase 4 update

### References

- [Source: _bmad-output/planning-artifacts/architecture.md#CI/CD Pipeline Architecture] — Tier 3 release pipeline spec (lines 489-497)
- [Source: _bmad-output/planning-artifacts/epics.md#Story 1.3] — acceptance criteria
- [Source: _bmad-output/planning-artifacts/prd.md#FR45] — automated releases with release notes
- [Source: _bmad-output/planning-artifacts/architecture.md#AR1] — HACS validation on every PR/release
- [Source: _bmad-output/planning-artifacts/architecture.md#AR2] — pin to HA's bundled dependency versions
- [Source: _qsprocess/workflows/development-lifecycle.md#Phase 4] — current manual release flow
- [Source: _bmad-output/implementation-artifacts/1-1-agentic-development-workflow.md] — Story 1.1 learnings

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

### Completion Notes List

- Created `.github/workflows/release.yml` with 4 jobs: quality-gate (pytest 100% + ruff + mypy), hacs-validate, version-check (tag vs manifest.json), release (gh release create --generate-notes)
- Updated `manifest.json` version from `2025.09.11` to `2026.03.19.0` to match current release tag
- Updated development-lifecycle.md Phase 4: replaced manual `gh release create` with automated pipeline (update manifest, push tag, pipeline handles rest)
- All quality gates pass: 3842 tests, 100% coverage, ruff clean, mypy clean

### Change Log

- 2026-03-19: Story 1.3 implemented — release pipeline, manifest version sync, lifecycle docs updated

### File List

New files:
- `.github/workflows/release.yml`

Modified files:
- `custom_components/quiet_solar/manifest.json` (version bump)
- `_qsprocess/workflows/development-lifecycle.md` (Phase 4 updated)
- `_bmad-output/implementation-artifacts/1-3-release-pipeline-version-migration.md` (story status + tasks)
