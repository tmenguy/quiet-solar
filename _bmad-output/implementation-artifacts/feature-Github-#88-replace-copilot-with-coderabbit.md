---
title: "Replace GitHub Copilot with CodeRabbit in review workflow"
issue: 88
branch: "QS_88"
status: draft
story_type: feature
---

# Story: Replace GitHub Copilot with CodeRabbit in Review Workflow

## Description

The `/review-story` skill currently triggers GitHub Copilot for automated PR review, but the repo has no Copilot access. The user has installed CodeRabbit on GitHub instead. This story replaces all Copilot references with CodeRabbit in both the skill definition and the supporting Python script.

CodeRabbit is a GitHub App that automatically reviews PRs when they are opened or updated — it does not need to be triggered via API like Copilot. The script needs to be updated to wait for and fetch CodeRabbit comments instead of triggering Copilot.

## Acceptance Criteria

- [ ] `scripts/qs/review_pr.py` no longer references Copilot
- [ ] `scripts/qs/review_pr.py` can detect and fetch CodeRabbit review comments
- [ ] `_qsprocess/skills/review-story.md` references CodeRabbit instead of Copilot
- [ ] CLI flags updated: `--trigger-copilot` removed, `--wait-copilot` renamed to `--wait-coderabbit`
- [ ] A test PR confirms CodeRabbit activates and posts comments

## Tasks

### Task 1: Update `scripts/qs/review_pr.py`

1. Remove `trigger_copilot_review()` function (CodeRabbit triggers automatically on PR open/update)
2. Remove `--trigger-copilot` CLI flag
3. Rename `--wait-copilot` to `--wait-coderabbit`
4. Update `wait_for_copilot()` → `wait_for_coderabbit()`: look for author `coderabbitai[bot]` instead of `copilot`/`github-actions`
5. Update all JSON output keys from `copilot` → `coderabbit`

### Task 2: Update `_qsprocess/skills/review-story.md`

1. Step 1: Remove `--trigger-copilot` call (CodeRabbit auto-triggers)
2. Step 3: Change `--wait-copilot 60` to `--wait-coderabbit 120` (CodeRabbit may need more time)
3. Update all prose references from "Copilot" to "CodeRabbit"

### Task 3: Validate with a test PR

1. Create a small test PR on the repo
2. Verify CodeRabbit posts a review
3. Run the updated `review_pr.py --fetch-comments --wait-coderabbit 120` and confirm it picks up CodeRabbit comments

## Dev Notes

- CodeRabbit is a GitHub App, not a GitHub user. Its bot username is `coderabbitai[bot]`.
- CodeRabbit auto-reviews on PR creation and push events — no API trigger needed.
- The `trigger_copilot_review()` function used `POST requested_reviewers` which is Copilot-specific and can be fully removed.
