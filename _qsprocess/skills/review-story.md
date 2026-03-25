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

Post significant findings as PR comments:
```bash
gh pr review {{pr_number}} --comment --body "{{review_body}}"
```

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
- **skip**: Move on without action

After processing all comments, if any fixes were made, run:
```bash
python scripts/qs/quality_gate.py
```

### 5. Output finish command

```
Review complete. To finish and merge:
  /finish-story --pr {{pr_number}} [--story-key {{key}}]
```
