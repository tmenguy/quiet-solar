# /create-plan

Create a story implementation artifact and commit it on the feature branch. Runs inside the worktree (or on the QS_N branch in --no-worktree mode).

## Input

- `--issue N` (required): GitHub issue number. Branch is `QS_N`.
- `--story-key X` (optional): Epic story key (e.g., "3.2") for looking up `epics.md`.
- `--plan /path` (optional): Path to an external plan `.md` file.

If neither `--story-key` nor `--plan` is provided, the issue title and body are used as the primary source of requirements.

## Prerequisites

You should be inside the worktree directory (`../quiet-solar-worktrees/QS_N/`) or on branch `QS_N` (if `--no-worktree` was used in `/setup-task`).

If not on `QS_N`, check out the branch:
```bash
git checkout QS_{{issue_number}}
```

Verify: `git branch --show-current` should return `QS_N`.

## Steps

### 1. Fetch issue details

```bash
python scripts/qs/fetch_issue.py --issue {{N}}
```

Capture `title`, `body`, `labels`, and `story_type` from the JSON output.

### 2. Build the story key with GitHub issue number

Construct the story key that bmad-create-story will use as the output filename. Always embed `Github-#N`:

- Epic stories: `{{epic}}-{{story}}-Github-#{{issue_number}}-{{slug}}`
  Example: `1-14-Github-#60-robust-story-naming-retrieval`
- Bug fixes: `bug-Github-#{{issue_number}}-{{slug}}`
- Other: `{{prefix}}-Github-#{{issue_number}}-{{slug}}`

When using `--issue N` without `--story-key`, the `story_type` from `fetch_issue.py` determines the prefix:
- `"bug"` → use `bug-Github-#{{issue_number}}-{{slug}}`
- `"feature"` → use `feature-Github-#{{issue_number}}-{{slug}}` (or epic-style if a story key is also provided)

This key becomes the filename: `_bmad-output/implementation-artifacts/{{story_key}}.md`

### 3. Write the story file using BMad

Follow the **bmad-create-story** skill to generate the story file.

When bmad-create-story asks which story, provide the **full story key from step 2** (e.g., `1-14-Github-#60-robust-story-naming-retrieval`). This ensures the output file already has `Github-#N` in its name — no rename needed.

Also provide the user's story key or feature description so bmad-create-story knows what to write about.

#### When `--issue N` is the only input

**CRITICAL**: The issue's `title`, `body`, and `labels` from step 1 are the **primary source of requirements**. You MUST:

1. **Pass the full issue `body`** (not just the title) as the feature description to bmad-create-story. The body contains the user's actual requirements, reproduction steps, expected behavior, and context — do not paraphrase or replace it with your own analysis.
2. **Use `labels`** to inform story framing: `bug` label → frame as bug fix; other labels (e.g., `area:car`, `area:ui`) indicate affected areas to investigate.
3. **Use `story_type`** to set the story key prefix (step 2) and the story tone.

Do NOT skip the issue body in favor of your own codebase exploration. The body is the user's intent — your analysis supplements it, not replaces it.

#### When `--plan` is provided

If the user passed an external plan `.md` file:

1. **Read the external plan** first to understand its analysis, root causes, and fix plan.
2. **Run bmad-create-story** as usual, feeding it both the story key and a summary of the plan content.
3. **Synthesize** the bmad output with the external plan:
   - The external plan is the **primary authority** — follow its structure, root cause analysis, and fix plan as closely as possible.
   - Only diverge from the external plan if bmad-create-story or your own analysis reveals something clearly missing or incorrect.
   - Preserve the external plan's fix ordering, priorities, and file/line references.
   - Add standard story structure (Story, Acceptance Criteria, Tasks) around the plan content if the plan doesn't already have them.
4. **Write the final story file** to the standard path `_bmad-output/implementation-artifacts/{{story_key}}.md` — this is a synthesis, not a copy.

#### When `--story-key` is provided

Look up the story in `_bmad-output/planning-artifacts/epics.md` and use it as context for bmad-create-story.

### 4. Enrich story frontmatter

After bmad-create-story finishes, add issue and branch info to the story file frontmatter (after the title line):

```yaml
issue: {{issue_number}}
branch: "QS_{{issue_number}}"
```

### 5. Commit and push

```bash
git add _bmad-output/
git commit -m "story: create {{story_key}} (Github-#{{issue_number}})"
git push -u origin QS_{{issue_number}}
```

### 6. Output next-step command

Run `next_step.py` to generate both command options:

```bash
python scripts/qs/next_step.py --skill implement-story --issue {{issue_number}} --work-dir "$(pwd)" --title "{{title}}"
```

Parse the JSON output and tell the user:

```
Story created on branch QS_{{issue_number}}.

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
  (Like C but tries to auto-type the command via AppleScript. Needs Accessibility permissions.)
```
