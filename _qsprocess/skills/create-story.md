# /create-story

Create a story implementation artifact and commit it on a feature branch.

## Input

The user provides ONE of:
- A story key (e.g., "3.2") referencing an epic in `_bmad-output/planning-artifacts/epics.md`
- A feature description (free text)
- A path to an external plan `.md` file (e.g., from Cursor) via `--plan /path/to/plan.md`
- An existing GitHub issue number via `--issue N` (e.g., `--issue 42`)

## Steps

### 1. Obtain GitHub issue

**If `--issue N` was provided** (existing issue):

```bash
python scripts/qs/fetch_issue.py --issue {{N}}
```

Capture `issue_number`, `title`, `body`, `labels`, and `story_type` (`"bug"` or `"feature"`) from the JSON output. Do NOT create a new issue — use this one. The `story_type` drives the story key prefix in step 3 (`bug-` for bugs, epic-style or generic for features).

**Otherwise** (new issue — the default):

```bash
python scripts/qs/create_issue.py --title "Story {{story_key}}: {{title}}"
```

Capture the `issue_number` from the JSON output.

### 2. Create feature branch

```bash
git checkout main && git pull
git checkout -b QS_{{issue_number}}
```

### 3. Build the story key with GitHub issue number

Construct the story key that bmad-create-story will use as the output filename. Always embed `Github-#N`:

- Epic stories: `{{epic}}-{{story}}-Github-#{{issue_number}}-{{slug}}`
  Example: `1-14-Github-#60-robust-story-naming-retrieval`
- Bug fixes: `bug-Github-#{{issue_number}}-{{slug}}`
- Other: `{{prefix}}-Github-#{{issue_number}}-{{slug}}`

When using `--issue N`, the `story_type` from `fetch_issue.py` determines the prefix:
- `"bug"` → use `bug-Github-#{{issue_number}}-{{slug}}`
- `"feature"` → use `feature-Github-#{{issue_number}}-{{slug}}` (or epic-style if a story key is also provided)

This key becomes the filename: `_bmad-output/implementation-artifacts/{{story_key}}.md`

### 4. Write the story file using BMad

Follow the **bmad-create-story** skill to generate the story file.

When bmad-create-story asks which story, provide the **full story key from step 3** (e.g., `1-14-Github-#60-robust-story-naming-retrieval`). This ensures the output file already has `Github-#N` in its name — no rename needed.

Also provide the user's story key or feature description so bmad-create-story knows what to write about.

#### When `--issue N` is provided

**CRITICAL**: The issue's `title`, `body`, and `labels` from step 1 are the **primary source of requirements**. You MUST:

1. **Pass the full issue `body`** (not just the title) as the feature description to bmad-create-story. The body contains the user's actual requirements, reproduction steps, expected behavior, and context — do not paraphrase or replace it with your own analysis.
2. **Use `labels`** to inform story framing: `bug` label → frame as bug fix; other labels (e.g., `area:car`, `area:ui`) indicate affected areas to investigate.
3. **Use `story_type`** to set the story key prefix (step 3) and the story tone.

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
4. **Write the final story file** to the standard path `_bmad-output/implementation-artifacts/{{story_key}}.md` — this is a synthesis, not a copy. The result should read as a coherent story artifact that happens to closely follow the external plan.

### 5. Enrich story frontmatter

After bmad-create-story finishes, add issue and branch info to the story file frontmatter (after the title line):

```yaml
issue: {{issue_number}}
branch: "QS_{{issue_number}}"
```

### 6. Commit and push

```bash
git add _bmad-output/
git commit -m "story: create {{story_key}} (Github-#{{issue_number}})"
git push -u origin QS_{{issue_number}}
```

### 7. Output setup command

Run `next_step.py` to generate both command options:

```bash
python scripts/qs/next_step.py --skill setup-story --issue {{issue_number}} --work-dir "$(pwd)" --title "{{title}}"
```

Parse the JSON output and tell the user:

```
Story created on branch QS_{{issue_number}}.

**Option A — New context:**
  {{new_context}}

**Option B — Same context:**
  {{same_context}}
```
