# /finish-story

Finish a story: doc-sync, quality gate, merge PR, cleanup — all automated.

## Input

No arguments required. The script auto-detects everything from the current branch (`QS_N` → issue number → story file via `find_story_file(N)`).

Optional overrides (rarely needed):
- `--pr N`: PR number
- `--issue N`: issue number
- `--skip-quality-gate`: skip local quality gate

## Steps

### 1. Mandatory doc-sync gate (agent judgment required)

Resolve the story file using the issue number from the current branch. Run:

```bash
python scripts/qs/doc_sync.py --issue {{issue_number}} --base-branch main --repo-path {{feature_worktree}}
```

If no story file is found, STOP and ask the user.

Review its output, then:

1. **For each discrepancy**: present it to the user with context
2. **User resolves each**: update the doc, or explain why the discrepancy is acceptable
3. **Apply approved changes** — they will be auto-committed by the script

Also do a manual read of the story artifact to catch anything the script can't — e.g., ACs whose intent doesn't match the implementation, or dev notes that are stale.

This gate is **mandatory** — do NOT proceed until all discrepancies are resolved.

### 2. Run the finish-story script

```bash
python scripts/qs/finish_story.py
```

The script handles everything automatically:
- Auto-detects issue number from branch, finds story file by issue
- Auto-commits pending changes and pushes
- Finds or creates the PR
- Runs the quality gate
- Checks CI status
- Ensures issue link in PR body
- Merges the PR (merge commit, not squash)
- Closes the GitHub issue
- Updates story artifact status to "done"
- Updates epics.md
- Cleans up the worktree
- Pulls main

### 3. Present report

The script outputs structured JSON. Present to the user:
- Success/failure status
- Any recovery instructions (if failed)
- Release: if `release.suggestion` is `"release"`, present both options from the release field:

```
**Option A — New context:**
  {{release.new_context}}

**Option B — Same context:**
  {{release.same_context}}
```

If `"no-release"`, tell the user no release is needed.

### 4. Commit epics update (if applicable)

If the script updated epics.md on main:
```bash
cd {{main_worktree}}
git add _bmad-output/planning-artifacts/epics.md
git commit -m "docs: mark story {{story_key}} as done"
git push origin main
```
