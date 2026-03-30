# /setup-story

Set up a worktree for story implementation and output the launch command.

## Input

The user provides: issue number (e.g., `42`).

Optional flags:
- `--no-worktree`: work directly on the branch in the main repo instead of creating a worktree.

## Steps

### 1. Run setup script

```bash
python scripts/qs/setup_worktree.py {{issue_number}} --title "{{title}}" [--no-worktree]
```

### 2. Display results

From the JSON output, show the user:

1. **Worktree path** — where the code lives
2. **Detected tool** — the `tool` field tells you whether the user is in Cursor or Claude Code
3. **Next-step commands** — parse the `same_context` and `new_context` fields and present both options:

```
Worktree ready at {{worktree_path}}

**Option A — New context:**
  {{new_context}}

**Option B — Same context:**
  {{same_context}}
```

#### Tool-specific behavior

- **Claude Code**: `new_context` is a ready-to-paste `sh /tmp/...` command that opens a new Claude session in the worktree with the implement skill pre-loaded.
- **Cursor**: `new_context` is a human-readable instruction to open the worktree folder as a new Cursor workspace. The user should: (1) File > Open Folder > select the worktree path, (2) type the skill command shown. Alternatively, the user can stay in the same window and use **Option B**.

#### `--no-worktree` mode

If the user prefers to stay in their current Cursor workspace (no context switch), suggest `--no-worktree`. This checks out the feature branch in the main repo directory instead of creating a separate worktree. Simpler for Cursor, but prevents parallel story development.

The user will choose one of these options. This skill does NOT start implementation itself.
