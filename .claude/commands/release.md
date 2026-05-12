---
description: Cut a release — bump manifest.json, commit, tag vYYYY.MM.DD.N, push. Runs on main. Always dry-runs first.
---

Use the **qs-release** subagent to handle this.

Expected outcome:
- Confirms clean main; refuses otherwise.
- Dry-runs `python scripts/qs/release.py --dry-run` and asks the user
  to authorize the proposed tag.
- On confirmation, runs `python scripts/qs/release.py` (bumps manifest,
  commits, pushes, tags).
- GitHub Actions handles the release pipeline from there.

User request:
$ARGUMENTS
