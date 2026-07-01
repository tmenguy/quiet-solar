---
description: Cut a release — bump manifest.json, commit, tag vYYYY.MM.DD.N, push. Runs on main. Always dry-runs first.
---

> **Preferred entry**: open a fresh terminal in the main checkout and
> run `claude --agent qs-release` (interactive session — you can review
> the dry-run output and authorize the proposed tag mid-flight).
>
> **This slash command is the degraded fallback** — kept for Claude
> Desktop and any chat without a CLI launcher. It spawns a one-shot
> non-interactive `Agent`-tool sub-process; the persona runs to
> completion and returns a final summary, and you cannot interject. This
> is the broken-by-design UX that QS-175 mitigates — we keep the slash
> command **only as a fallback**, not as the primary flow.

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
