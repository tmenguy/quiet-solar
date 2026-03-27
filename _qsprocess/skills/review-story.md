# /review-story

Review a PR: run code review, trigger Copilot, process feedback.

## Input

- `--pr N` (required): PR number
- `--issue N` (optional): issue number for context

## Steps

### 1. Trigger Copilot review

```bash
python scripts/qs/review_pr.py {{pr_number}} --trigger-copilot
```

### 2. Run adversarial code review

Run the BMad adversarial code review skill:
```
/bmad-code-review
```

This executes parallel review layers (Blind Hunter, Edge Case Hunter, Acceptance Auditor) and triages findings into actionable categories. It is the primary local review mechanism — do NOT skip it.

When adversarial findings reveal impacts beyond the code fix (e.g., a missing acceptance criterion, an architecture gap, or a rule that should be added), flag the doc impact alongside the code finding so the user can choose "doc-update" when processing the comment.

Post significant findings as PR comments:
```bash
gh pr review {{pr_number}} --comment --body "{{review_body}}"
```

### 2.5. Inline doc-sync (continuous)

Throughout the review, watch for user direction that impacts the story specification or project docs. When detected, flag it, propose specific edits to the story artifact, and wait for approval before applying. For structural changes, also mention `architecture.md` or `_qsprocess/rules/project-rules.md` as separate proposals.

### 3. Fetch all review comments

Wait briefly for Copilot, then fetch:
```bash
python scripts/qs/review_pr.py {{pr_number}} --fetch-comments --wait-copilot 60
```

### 4. Process each unresolved comment

Process comments ONE AT A TIME. For each comment, present it to the user with file path, line, and diff context. Wait for the user's response before moving to the next. Do NOT batch multiple comments. For each one, ask the user to choose:

- **fix**: Implement the fix, run quality gates, commit, push
- **discuss**: Post a reply on the PR explaining the rationale
- **reject**: Post a rationale and resolve the thread
- **doc-update**: The comment reveals a spec gap, missing AC, architecture concern, or rule that should be documented. Propose specific edits to the story artifact (or other docs), wait for user approval, apply and commit if accepted.
- **skip**: Move on without action

After processing all comments, if any fixes were made, run:
```bash
python scripts/qs/quality_gate.py
```

### 4.5. Compound doc-sync

After processing all comments, review everything that happened during the review session:

1. **Scan**: all user direction, review findings (GitHub comments, adversarial review, Copilot feedback), and any implementation decisions made during fixes
2. **Propose**: a consolidated list of story artifact updates — AC adjustments, task modifications, dev notes. Present each proposed change clearly.
3. **Secondary docs**: if review revealed structural gaps, optionally propose updates to `architecture.md` or `_qsprocess/rules/project-rules.md`
4. **User approves/rejects** each item
5. **Apply and commit**: edit approved changes in place, commit and push

If inline doc-sync (Step 2.5) and doc-update actions already captured all changes, say so and move on.

### 5. Output finish command

```
Review complete. To finish and merge:
  /finish-story --pr {{pr_number}} [--story-key {{key}}]
```
