---
description: >-
  Create a release — determine the next version tag, bump version files,
  tag and push, create a GitHub Release. Runs on main, independent of any
  task or worktree.
agent: qs-release
subtask: true
---

Delegate to the `qs-release` subagent. Pass the user's full request
(version hint, bump type, or flags) verbatim. The subagent owns the full
release protocol.

Expected outcome:
- Quality gate passes on clean main.
- Version bumped in manifest files.
- Git tag created and pushed.
- GitHub Release created with auto-generated notes.

This is one of the **two static slash commands** in the OpenCode pipeline
(the other is `/setup-task`). Release is independent of any task — it
operates on whatever is currently merged to main.
