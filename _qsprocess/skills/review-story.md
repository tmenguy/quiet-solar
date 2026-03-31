# /review-story

Review a PR: run code review, wait for CodeRabbit, process feedback.

## Input

- `--pr N` (required): PR number
- `--issue N` (optional): issue number for context

## Steps

### 1. Run adversarial code review

Follow the **bmad-code-review** skill. This executes parallel review layers (Blind Hunter, Edge Case Hunter, Acceptance Auditor) and triages findings into actionable categories. It is the primary local review mechanism — do NOT skip it.

When adversarial findings reveal impacts beyond the code fix (e.g., a missing acceptance criterion, an architecture gap, or a rule that should be added), flag the doc impact alongside the code finding so the user can choose "doc-update" when processing the comment.

Post significant findings as PR comments:
```bash
gh pr review {{pr_number}} --comment --body "{{review_body}}"
```

### 1.5. Inline doc-sync (continuous)

Throughout the review, watch for user direction that impacts the story specification or project docs. When detected, flag it, propose specific edits to the story artifact, and wait for approval before applying. For structural changes, also mention `architecture.md` or `_qsprocess/rules/project-rules.md` as separate proposals.

### 2. Fetch all review comments

Wait for CodeRabbit (auto-triggers on PR creation/push), then fetch:
```bash
python scripts/qs/review_pr.py {{pr_number}} --fetch-comments --wait-coderabbit 120
```

### 3. Process each unresolved comment

Process comments ONE AT A TIME. For each comment, present it to the user with file path, line, and diff context. Wait for the user's response before moving to the next. Do NOT batch multiple comments. For each one, ask the user to choose:

- **fix**: Implement the fix, run quality gates, commit, push
- **discuss**: Post a reply on the PR explaining the rationale
- **reject**: Post a rationale and resolve the thread
- **doc-update**: The comment reveals a spec gap, missing AC, architecture concern, or rule that should be documented. Propose specific edits to the story artifact (or other docs), wait for user approval, apply and commit if accepted.
- **skip**: Move on without action

After processing all comments, if any fixes were made, run:
```bash
python scripts/qs/quality_gate.py --cache
```

### 3.5. Compound doc-sync

After processing all comments, review everything that happened during the review session:

1. **Scan**: all user direction, review findings (GitHub comments, adversarial review, CodeRabbit feedback), and any implementation decisions made during fixes
2. **Propose**: a consolidated list of story artifact updates — AC adjustments, task modifications, dev notes. Present each proposed change clearly.
3. **Secondary docs**: if review revealed structural gaps, optionally propose updates to `architecture.md` or `_qsprocess/rules/project-rules.md`
4. **User approves/rejects** each item
5. **Apply and commit**: edit approved changes in place, commit and push

If inline doc-sync (Step 1.5) and doc-update actions already captured all changes, say so and move on.

### 4. Output finish command

Run `next_step.py` to generate both command options:

```bash
python scripts/qs/next_step.py --skill finish-story --issue {{issue_number}} --pr {{pr_number}} --work-dir {{worktree_path}} --title "{{title}}"
```

Parse the JSON output (which includes `tool`, `same_context`, `new_context`) and tell the user:

```
Review complete.

**Option A — New context:**
  {{new_context}}

**Option B — Same context:**
  {{same_context}}
```

For Cursor users, `new_context` will be instructions to open the worktree as a new workspace. For Claude Code, it will be a sh launch script. The user can also continue in the same session with **Option B**.
