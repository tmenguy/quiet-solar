---
name: qs-release
description: >-
  Cut a release: bump manifest.json, commit, tag vYYYY.MM.DD.N, push.
  Runs on the main checkout. Independent of any task.
model: inherit
readonly: false
is_background: false
---

# qs-release — tag and ship

You cut a release from the main checkout. Independent of any task.

## Phase protocol

### 1. Confirm clean main
```bash
git checkout main
git pull
git status
```
`git status` must be clean. If not, STOP.

### 2. Dry run
```bash
python scripts/qs/release.py --dry-run
```
Show output. Ask "Proceed with this release?". Wait.

### 3. Real run
```bash
python scripts/qs/release.py
```

### 4. Report

```text
✅ Released {{tag}} ({{version}}).
GitHub Actions handles the release pipeline.
```

## Hard rules

- Always dry-run first. Get explicit confirmation.
- Refuse to run if `git status` not clean.
- Refuse to run if not on `main`.
