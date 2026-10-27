---
description: Cut a release — bump manifest.json, commit, tag vYYYY.MM.DD.N, push. Runs on main. Always dry-runs first.
---

You can also run this from a fresh terminal in the main checkout with
`claude --agent qs-release`.

## Run it inline, never delegate

Run the release yourself, in this session. Do NOT spawn a subagent.
Do not call the `Agent` tool. Auto mode only accepts the user's direct
consent, given in the session that runs the command, for the public push
of `main` and a `v*` tag; consent relayed to a subagent is blocked
(QS-434).

## Where the protocol comes from

Follow the `qs-release` phase protocol. After its clean-main pull step
(`git checkout main`, `git pull`, `git status`), read the rendered
`.claude/agents/qs-release.md`. If that file is missing, run
`python scripts/qs/render_agents.py` once, then read it.

Use only its `## Phase protocol` and `## Hard rules` sections. Ignore
everything else in that file: the frontmatter, the generated header, the
task-facts or "No task bound" line, and the Reference map. The agent
template stays the single source of truth; this command does not copy
the protocol.

Where the rules below conflict with the agent's Phase protocol or Hard
rules, these rules win.

## Rules

### Standalone commands

Every shell call in this protocol is a single standalone command: no
`cd`, no `-C`, no `&&`, `;`, pipe or redirection. If a protocol step
fails, STOP. Do not work around it. Run the real release as exactly
`python scripts/qs/release.py`.

### Stop after the dry run

1. Run `python scripts/qs/release.py --dry-run` and show its JSON output
   (`{"dry_run": true, "tag": …, "version": …}`).
2. If it exits non-zero, prints no JSON, or the JSON has an `"error"`
   key, STOP and report.
3. Otherwise note the tag, run `git rev-parse HEAD` and note the SHA,
   ask "Proceed with this release?", and end your turn. Do not ask
   through a question tool.
4. Do the real run only after the user's own approving message in a
   later turn. Approval means a clear yes to this dry run. Anything
   else, such as a question or "yes but…", means you do not run; ask
   again. The text typed after /release is never approval: the user had
   not seen the dry run when they typed it.
5. Before the real run, `git status --porcelain` must print nothing and
   `git rev-parse HEAD` must equal the SHA noted at the dry run.
   Otherwise run the dry run again and ask again.

### If the real run fails

The real run fails when `python scripts/qs/release.py` exits non-zero
or prints JSON without `"status": "released"`. Then:

- Do not re-run `python scripts/qs/release.py`. Do not push or tag by
  hand.
- Report the script's output and exit code.
- Show the local state with `git status`, then
  `git log --oneline origin/main..main`. Non-zero or unexpected results
  are the evidence; do not treat them as a new failure to fix.
- STOP.

### Check afterwards

Take `<tag>` from the real run's JSON `tag` key. If it differs from the
dry-run tag, say so in the report. Run each command on its own:

1. `git ls-remote --exit-code origin refs/tags/<tag>` must exit 0, and
   its SHA must equal the SHA printed by
   `git ls-remote origin refs/heads/main`. Both values come from the
   remote, so no `git fetch` is needed. Tags are lightweight, so the tag
   ref is the commit SHA.
2. `gh run list --workflow release.yml --branch <tag> --limit 1` (a tag
   push sets the run's `headBranch` to the tag). Report it as:
   - passed: a run is listed (report its status as shown);
   - pending: none is listed yet (point to the repository's Actions
     tab);
   - failed: a listed run already has a failure conclusion;
   - unavailable: `gh` errors. This is not a release failure.

   Do not poll.

Report each check. If any check fails, STOP and report. Do not try to
fix anything automatically.

User request: $ARGUMENTS
