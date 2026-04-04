# /setup-task

Create a GitHub issue (if needed), set up a feature branch and worktree for parallel development — without touching main's checkout state.

**IMPORTANT: This skill must be fast and automatic.** Do NOT analyze, diagnose, or interpret the user's input (e.g., do not read log files to understand a bug, do not research the codebase). Just pass the text through to the GitHub issue as-is. Deep analysis belongs in `/create-plan`.

## Input

The user provides ONE of:
- A story key (e.g., "3.2") referencing an epic in `_bmad-output/planning-artifacts/epics.md`
- A feature description (free text, may include logs or error traces)
- A path to an external plan `.md` file (e.g., from Cursor) via `--plan /path/to/plan.md`
- An existing GitHub issue number via `--issue N` (e.g., `--issue 42`)

Optional flags:
- `--no-worktree`: create branch only, skip worktree creation. Work will happen in the main repo directory.

## Steps

### 1. Obtain GitHub issue

**If `--issue N` was provided** (existing issue):

```bash
python scripts/qs/fetch_issue.py --issue {{N}}
```

Capture `issue_number`, `title`, `body`, `labels`, and `story_type` from the JSON output. Do NOT create a new issue — use this one.

**Otherwise** (new issue — the default):

Derive title and body directly from the input — no codebase analysis, no log interpretation:
- **Story key**: Look up `_bmad-output/planning-artifacts/epics.md` for the story title and description. Title: `"Story {{story_key}}: {{title}}"`.
- **Plan file**: Read the plan, use its title as issue title. Body: the full plan text.
- **Free text**: Extract a short title (first sentence or first ~80 chars). Body: the full text verbatim (including any logs, traces, etc.).

```bash
python scripts/qs/create_issue.py --title "{{title}}" --body "{{body}}" [--labels "{{labels}}"]
```

Capture the `issue_number` from the JSON output.

### 2. Set up branch and worktree

```bash
python scripts/qs/setup_task.py {{issue_number}} --title "{{title}}" [--no-worktree] [--story-key {{key}}] [--plan {{path}}]
```

This script:
- Fetches latest from origin (without touching main's checkout)
- Creates worktree via `scripts/worktree-setup.sh` (or just creates the branch for `--no-worktree`)
- Builds next-step commands for `/create-plan`

### 3. Display results

From the JSON output, show the user:

```
Task #{{issue_number}} set up on branch QS_{{issue_number}}.
Worktree: {{worktree_path}}

**Option A — New context:**
  {{new_context}}

**Option B — Same context:**
  {{same_context}}
```

If the JSON includes `pycharm_context`, also show:

```
**Option C — PyCharm + clipboard:**
  {{pycharm_context}}
  (Opens PyCharm on the worktree. In PyCharm: ⌥F12 → ⌘V → Enter)
```

If the JSON includes `pycharm_applescript_context`, also show:

```
**Option D — PyCharm + auto-type (experimental):**
  {{pycharm_applescript_context}}
  (Like C but tries to auto-type via AppleScript. Needs Accessibility permissions.)
```

#### Tool-specific behavior

- **Claude Code**: `new_context` is a ready-to-paste `sh /tmp/...` command that opens a new Claude session in the worktree with `/create-plan` pre-loaded.
- **Cursor**: `new_context` is a human-readable instruction to open the worktree folder as a new Cursor workspace, then type the `/create-plan` command shown.

#### `--no-worktree` mode

Branch `QS_N` is created from `origin/main` but NOT checked out. The next step (`/create-plan`) will check it out when it runs. This prevents parallel story development but avoids worktree overhead.
