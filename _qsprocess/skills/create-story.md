# /create-story

Create a story implementation artifact and commit it on a feature branch.

## Input

The user provides ONE of:
- A story key (e.g., "3.2") referencing an epic in `_bmad-output/planning-artifacts/epics.md`
- A feature description (free text)

## Steps

### 1. Create GitHub issue

```bash
python scripts/qs/create_issue.py --title "Story {{story_key}}: {{title}}" --story-key "{{story_key}}"
```

Capture the `issue_number` from the JSON output.

### 2. Create feature branch

```bash
git checkout main && git pull
git checkout -b QS_{{issue_number}}
```

### 3. Write the story file using BMad

Run the BMad create-story skill to generate the comprehensive story file:
```
/bmad-create-story
```

This skill performs exhaustive analysis of epics, architecture, previous stories, git history, and latest tech specifics to create a developer-ready story file. Do NOT attempt to write the story file manually — BMad does this far better.

When bmad-create-story asks which story, provide the story key or feature description from the user.

### 4. Enrich story frontmatter

After bmad-create-story finishes, add issue and branch info to the story file frontmatter:

```yaml
issue: {{issue_number}}
branch: "QS_{{issue_number}}"
```

### 5. Commit and push

```bash
git add _bmad-output/
git commit -m "story: create {{story_key}}"
git push -u origin QS_{{issue_number}}
```

### 6. Output setup command

Tell the user:
```
Story created on branch QS_{{issue_number}}.
To set up for implementation, run:
  /setup-story {{issue_number}}
```
