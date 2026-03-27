# /setup-story

Set up a worktree for story implementation and output the launch command.

## Input

The user provides: issue number (e.g., `42`) or story key (e.g., `3.2`).
If a story key is given, look up the issue number from the story file frontmatter or from `gh issue list`.

## Steps

### 1. Run setup script

```bash
python scripts/qs/setup_worktree.py {{issue_number}} --story-key "{{story_key}}" --title "{{title}}"
```

### 2. Display results

From the JSON output, show the user:

1. **Worktree path** — where the code lives
2. **Next-step commands** — parse the `same_context` and `new_context` fields from the JSON output and display both options:

```
Worktree ready at {{worktree_path}}

**Option A — New context (copy-paste this single command):**
  {{new_context}}

**Option B — Same context:**
  {{same_context}}
```

The `new_context` command opens a new Claude context AND immediately runs the implement skill (prompt embedded). The `same_context` is just the `/skill --args` string to type in a current session.

The user will copy one of these commands. This skill does NOT start implementation itself.
