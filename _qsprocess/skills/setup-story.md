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
2. **Launch command** — use the `launch_command` field from the JSON output (it already includes terminal title, `CLAUDE_LAUNCH_OPTS`, and `--name`)
3. **First prompt** — what to type in the new context:
   ```
   /implement-story --issue {{issue_number}} --story-file {{story_file}}
   ```

The user will copy these commands and launch a separate terminal/context for implementation. This skill does NOT start implementation itself.
