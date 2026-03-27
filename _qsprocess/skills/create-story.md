# /create-story

Create a story implementation artifact and commit it on a feature branch.

## Input

The user provides ONE of:
- A story key (e.g., "3.2") referencing an epic in `_bmad-output/planning-artifacts/epics.md`
- A feature description (free text)

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

Run the BMad create-story skill:
```
/bmad-create-story
```

When bmad-create-story asks which story, provide the **full story key from step 3** (e.g., `1-14-Github-#60-robust-story-naming-retrieval`). This ensures the output file already has `Github-#N` in its name — no rename needed.

Also provide the user's story key or feature description so bmad-create-story knows what to write about.

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
