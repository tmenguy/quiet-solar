# /create-story

Create a story implementation artifact and commit it on a feature branch.

## Input

The user provides ONE of:
- A story key (e.g., "3.2") referencing an epic in `_bmad-output/planning-artifacts/epics.md`
- A feature description (free text)
- A path to an external plan `.md` file (e.g., from Cursor) via `--plan /path/to/plan.md`

## Steps

### 1. Create GitHub issue

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

This key becomes the filename: `_bmad-output/implementation-artifacts/{{story_key}}.md`

### 4. Write the story file using BMad

Follow the **bmad-create-story** skill to generate the story file.

When bmad-create-story asks which story, provide the **full story key from step 3** (e.g., `1-14-Github-#60-robust-story-naming-retrieval`). This ensures the output file already has `Github-#N` in its name — no rename needed.

Also provide the user's story key or feature description so bmad-create-story knows what to write about.

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

Tell the user:
```
Story created on branch QS_{{issue_number}}.
To set up for implementation, run:
  /setup-story {{issue_number}}
```
