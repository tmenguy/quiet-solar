# Story: Unified Next-Step Command Output for Skill Transitions

issue: 52
branch: "QS_52"
Status: ready-for-dev

## Story

As TheDev,
I want each skill to present its "next step" as two ready-to-use commands (one for a new context with the prompt embedded, one for the same context),
so that I never have to copy-paste two separate things and the transition output is consistent across all skills.

## Acceptance Criteria

1. **Given** a skill finishes and needs to hand off to the next skill (setup-story -> implement-story, implement-story -> review-story, review-story -> finish-story)
   **When** it displays the next-step commands
   **Then** it shows exactly two options:
   - **Option A (new context)**: a single copy-pasteable command that opens a new Claude context AND immediately runs the next skill (prompt embedded as positional arg)
   - **Option B (same context)**: just the `/skill --args` string to type in the current session

2. **Given** the next-step commands are generated
   **When** any skill calls the generation logic
   **Then** the commands are produced by `python scripts/qs/next_step.py` (centralized, not hand-built in each skill markdown)

3. **Given** `claude_launch_command()` in `utils.py` is called
   **When** a `prompt` parameter is provided
   **Then** the prompt is appended as a quoted positional argument to the `claude` invocation

4. **Given** `/setup-story` completes
   **When** it displays the next step
   **Then** it runs `next_step.py --skill implement-story --issue N [--story-file PATH] --work-dir DIR --title TITLE`
   **And** shows both options from the JSON output

5. **Given** `/implement-story` completes (PR created)
   **When** it displays the next step
   **Then** it runs `next_step.py --skill review-story --issue N --pr PR --work-dir DIR --title TITLE`
   **And** shows both options from the JSON output

6. **Given** `/review-story` completes
   **When** it displays the next step
   **Then** it runs `next_step.py --skill finish-story --pr N [--story-key KEY] --work-dir DIR --title TITLE`
   **And** shows both options from the JSON output

## Tasks / Subtasks

- [ ] Task 1: Add `prompt` parameter to `claude_launch_command()` in `utils.py` (AC: #3)
  - [ ] 1.1 Add optional `prompt: str | None = None` kwarg
  - [ ] 1.2 When provided, append `shlex.quote(prompt)` as positional arg to the claude command
  - [ ] 1.3 Existing callers (no prompt) remain unchanged

- [ ] Task 2: Create `scripts/qs/next_step.py` (AC: #2)
  - [ ] 2.1 Accept args: `--skill` (required), `--issue`, `--pr`, `--story-file`, `--story-key`, `--work-dir` (required), `--title` (required)
  - [ ] 2.2 Build the skill prompt string (e.g., `/review-story --pr 5 --issue 42`)
  - [ ] 2.3 Build the full new-context command via `claude_launch_command(work_dir, issue, title, prompt=skill_prompt)`
  - [ ] 2.4 Output JSON with keys: `same_context`, `new_context`

- [ ] Task 3: Update `setup_worktree.py` to output both command options (AC: #4)
  - [ ] 3.1 Replace current `launch_command` + `first_prompt` + `instructions` with `same_context` and `new_context` fields
  - [ ] 3.2 Use `claude_launch_command(..., prompt=implement_prompt)` for `new_context`
  - [ ] 3.3 Keep `implement_prompt` as `same_context`

- [ ] Task 4: Update `setup-story.md` skill to use new output format (AC: #4)
  - [ ] 4.1 Step 2: parse `same_context` and `new_context` from JSON
  - [ ] 4.2 Display both options with clear labels
  - [ ] 4.3 Enforce: the skill MUST run `next_step.py` or parse the equivalent fields from `setup_worktree.py` — never hand-build commands

- [ ] Task 5: Update `implement-story.md` step 6 to use `next_step.py` (AC: #5)
  - [ ] 5.1 After PR creation, run `python scripts/qs/next_step.py --skill review-story --issue N --pr PR --work-dir DIR --title TITLE`
  - [ ] 5.2 Display both options from JSON output
  - [ ] 5.3 Remove the old "launch_command" + "Then type:" two-step pattern

- [ ] Task 6: Update `review-story.md` step 5 to use `next_step.py` (AC: #6)
  - [ ] 6.1 Run `python scripts/qs/next_step.py --skill finish-story --pr N [--story-key KEY] --work-dir DIR --title TITLE`
  - [ ] 6.2 Display both options from JSON output

## Dev Notes

### Architecture Constraints

- **Script pattern**: all scripts in `scripts/qs/` use `argparse` + `output_json()` from `utils.py`. Follow this exact pattern for `next_step.py`.
- **Import style**: `from utils import ...` (scripts run from `scripts/qs/` dir). Do NOT use absolute imports.
- **Existing callers**: `setup_worktree.py` already calls `claude_launch_command()`. Update it in-place — do not break its current JSON contract without updating the skill markdown that reads it.
- **`CLAUDE_LAUNCH_OPTS`**: defined in `utils.py` as a constant. Never duplicate or hardcode this value elsewhere.

### Key Files to Modify

| File | Change |
|------|--------|
| `scripts/qs/utils.py` | Add `prompt` kwarg to `claude_launch_command()` |
| `scripts/qs/next_step.py` | NEW — centralized next-step command builder |
| `scripts/qs/setup_worktree.py` | Replace `launch_command`/`first_prompt`/`instructions` with `same_context`/`new_context` |
| `_qsprocess/skills/setup-story.md` | Update step 2 display to show both options |
| `_qsprocess/skills/implement-story.md` | Update step 6 to call `next_step.py`, show both options |
| `_qsprocess/skills/review-story.md` | Update step 5 to call `next_step.py`, show both options |

### Output Format Convention

All skill transitions MUST show next steps in this exact format:

```
{{completion_message}}

**Option A — New context (copy-paste this single command):**
  {{new_context}}

**Option B — Same context:**
  {{same_context}}
```

### What NOT to Do

- Do NOT add `next_step.py` logic inside `setup_worktree.py` — keep them separate. `setup_worktree.py` handles worktree setup; `next_step.py` handles command generation. `setup_worktree.py` can call `claude_launch_command()` directly since it already does.
- Do NOT change `/create-story` output — it already outputs a simple same-context `/setup-story` command which is fine (no new-context needed there since create-story runs in main repo).
- Do NOT change `/finish-story` output — it just suggests `/release` which is a same-context command.
- Do NOT break the `setup_worktree.py` JSON output without updating `setup-story.md` in the same task.

### Previous Story Intelligence

Stories 1.7 (worktrees) and 1.8 (AI-assisted PR review) established the current `scripts/qs/` patterns. Story 1.11 (doc-sync) added `doc_sync.py` following the same pattern. All use `argparse` + `output_json()`.

The `claude_launch_command()` function was introduced in story 1.7 and is currently used by `setup_worktree.py` only.

### Testing

This is a dev-tooling change (skill markdown + Python scripts). Testing approach:
- Unit test `claude_launch_command()` with and without `prompt` param
- Unit test `next_step.py` argument parsing and JSON output
- Manual verification: run each skill transition and confirm both commands work

### Project Structure Notes

- Scripts: `scripts/qs/next_step.py` (new)
- Skills: `_qsprocess/skills/{setup-story,implement-story,review-story}.md` (modified)
- Utils: `scripts/qs/utils.py` (modified)

### References

- [Source: scripts/qs/utils.py#claude_launch_command] — current launch command builder
- [Source: scripts/qs/setup_worktree.py] — current worktree setup with launch_command output
- [Source: _qsprocess/skills/implement-story.md#Step 6] — current review command output
- [Source: _qsprocess/skills/setup-story.md#Step 2] — current implement command output
- [Source: _qsprocess/skills/review-story.md#Step 5] — current finish command output

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### Change Log

### File List
