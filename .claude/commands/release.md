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
(`git checkout main`, `git pull`, `git status`), always run
`python scripts/qs/render_agents.py` (the pull may have brought a newer
template), then read the rendered `.claude/agents/qs-release.md`.

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
3. The dry run has pulled, so `origin/main` is fresh:
   `git log --oneline origin/main..main` must print nothing. Otherwise
   STOP and report: the real run would publish those unpushed commits.
4. Otherwise note the tag, run `git rev-parse HEAD` and note the SHA,
   ask "Proceed with this release?", and end your turn. Do not ask
   through a question tool.
5. Do the real run only after the user's own approving message in a
   later turn. Approval means a clear yes to this dry run. Anything
   else, such as a question or "yes but…", means you do not run; ask
   again. The text typed after /release is never approval: the user had
   not seen the dry run when they typed it.
6. Right before the real run, re-check, each command on its own:
   - `git status --porcelain` must print nothing. If the tree is not
     clean, STOP and report; do not loop back to the dry run.
   - Run `python scripts/qs/release.py --dry-run` again (it pulls). Its
     tag must equal the approved tag, `git rev-parse HEAD` must equal
     the SHA noted at the approved dry run, and
     `git log --oneline origin/main..main` must still print nothing.
     If the tag or the SHA differs, show the new dry run and ask again
     (step 4). If unpushed commits appear, STOP and report.

### If the real run fails

The real run fails when `python scripts/qs/release.py` exits non-zero
or prints JSON without `"status": "released"`. Then:

- Do not re-run `python scripts/qs/release.py`. Do not push or tag by
  hand.
- Report the script's output and exit code.
- Show the state, each command on its own: `git status`,
  `git log --oneline origin/main..main`, `git tag --list <dry-run tag>`,
  and `git ls-remote origin refs/tags/<dry-run tag>`. These show a
  partial release, such as main pushed but the tag not. Non-zero or
  unexpected results are the evidence; do not treat them as a new
  failure to fix.
- STOP.

### Check afterwards

Take `<tag>` from the real run's JSON `tag` key. If it differs from the
dry-run tag, say so in the report. Run each command on its own:

1. `git ls-remote --exit-code origin refs/tags/<tag>` must exit 0, and
   its SHA must equal local `git rev-parse HEAD` (the commit the script
   just tagged). Tags are lightweight, so the tag ref is the commit SHA.
2. `git ls-remote origin refs/heads/main` should print that same SHA.
   If it differs, report that main has moved on since the release; that
   alone is not a release failure.
3. `gh run list --workflow release.yml --branch <tag> --limit 1` (a tag
   push sets the run's `headBranch` to the tag). Report the first label
   that matches:
   - unavailable: `gh` errors. This is not a release failure.
   - pending: nothing is listed yet, or the listed run is queued or in
     progress (point to the repository's Actions tab);
   - passed: the listed run completed with success;
   - failed: the listed run completed with any other conclusion
     (including cancelled).

   Do not poll.

Report each check. If any check fails, STOP and report. Do not try to
fix anything automatically.

User request: $ARGUMENTS
