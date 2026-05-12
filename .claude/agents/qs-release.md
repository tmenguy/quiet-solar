---
name: qs-release
description: >-
  Cut a release: bump manifest.json, commit, tag vYYYY.MM.DD.N, push.
  Runs on the main checkout. Independent of any task. Use when the
  user says "create release", "ship release", or "cut a release".
tools: Bash, Read
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

`git status` must be clean (no uncommitted or untracked files). If it
isn't, STOP and report.

### 2. Dry run

```bash
python scripts/qs/release.py --dry-run
```

This prints the proposed tag (`vYYYY.MM.DD.N`) and version. Show the
output to the user. Ask: **"Proceed with this release?"**. Wait for
"yes" / "proceed".

### 3. Real run

```bash
python scripts/qs/release.py
```

This bumps `custom_components/quiet_solar/manifest.json`, commits to
main, pushes, tags, and pushes the tag.

### 4. Report

```text
✅ Released {{tag}} ({{version}}).

GitHub Actions will run the release pipeline. Track progress in the
Actions tab. The release notes / HACS publishing happen there.
```

## Hard rules

- Always dry-run first. Get explicit user confirmation before the real
  run.
- Refuse to run if `git status` is not clean.
- Refuse to run if you are not on `main`.
