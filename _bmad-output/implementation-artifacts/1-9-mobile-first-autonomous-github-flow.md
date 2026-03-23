# Story 1.9: Mobile-First Autonomous GitHub Flow

Status: ready-for-dev

## Story

As TheDev,
I want to create a tagged GitHub issue from my phone, have a cloud-based automation pick it up, run the full BMad workflow autonomously (YOLO mode), and present the result as a PR for review -- which I can review, approve, and merge from my phone, triggering the full release pipeline,
So that the entire development lifecycle from idea to release can be driven from a mobile device with zero local setup.

## Acceptance Criteria

1. **Given** TheDev creates a GitHub issue from mobile and applies the `auto-bmad` label
   **When** the GitHub Actions workflow detects the labeled issue
   **Then** it picks up the issue and starts the full BMad workflow autonomously
   **And** it creates a branch, implements the fix or feature following all project rules
   **And** it runs quality gates (tests, lint, type check, coverage) and iterates until passing
   **And** it creates a PR linking to the issue with full template and risk assessment

2. **Given** the autonomous PR is ready
   **When** TheDev reviews from the GitHub mobile app
   **Then** the PR includes a clear summary of what was done, why, and what changed
   **And** TheDev can approve or request changes from the GitHub mobile app

3. **Given** TheDev approves and merges the PR from mobile
   **When** the merge completes
   **Then** the release pipeline triggers automatically (per Story 1.3)
   **And** the full flow from issue creation to release completes without any local terminal access

4. **Given** the autonomous workflow encounters a problem it cannot resolve
   **When** the failure is detected (quality gates fail after max retries, scope too large, ambiguous intent)
   **Then** it posts a comment on the issue explaining what went wrong
   **And** it labels the issue `auto-bmad-failed` so TheDev can intervene
   **And** no PR is created for incomplete or broken work

5. **Given** the autonomous workflow runs
   **When** execution completes (success or failure)
   **Then** a cost/usage summary is posted as an issue comment (tokens used, duration, retries)
   **And** the workflow respects a configurable timeout (default: 30 minutes)

## Tasks / Subtasks

- [ ] Task 0: Evaluate execution environment (AC: #1)
  - [ ] 0.1 Investigate Claude Code for Web / remote agent capabilities (repo access, shell commands, PR creation)
  - [ ] 0.2 Decide: Option A (GitHub Actions + CLI) vs Option B (Claude Code remote agent)
  - [ ] 0.3 If Option B: verify auth model (GitHub App, OAuth, deploy key?) and quality gate execution
  - [ ] 0.4 Document decision rationale in this story's completion notes

- [ ] Task 1: Create `auto-bmad.yml` GitHub Actions workflow (AC: #1)
  - [ ] 1.1 Trigger on `issues.labeled` event where label is `auto-bmad`
  - [ ] 1.2 **If Option A:** Checkout repo, set up Python venv, install Claude Code CLI, authenticate via `ANTHROPIC_API_KEY` secret
  - [ ] 1.3 **If Option B:** Call Claude Code remote agent API / trigger instead of local CLI
  - [ ] 1.4 Parse issue title and body to determine intent (bug vs feature)
  - [ ] 1.5 Run Claude Code with the appropriate BMad workflow command
  - [ ] 1.6 Configure timeout (30 min default, configurable via repo variable)

- [ ] Task 2: Implement intent routing and autonomous execution (AC: #1, #2)
  - [ ] 2.1 Create a prompt template that feeds issue context to Claude Code
  - [ ] 2.2 Route `bug` labeled issues to quick-dev flow, others to feature flow
  - [ ] 2.3 Ensure the agent follows all project rules (`_qsprocess/rules/project-rules.md`)
  - [ ] 2.4 Agent creates branch (`QS_<issue_number>`), implements, runs quality gates
  - [ ] 2.5 Agent creates PR with full template, risk assessment, and link to issue

- [ ] Task 3: Implement guardrails and failure handling (AC: #4, #5)
  - [ ] 3.1 Add workflow timeout (GitHub Actions `timeout-minutes`)
  - [ ] 3.2 On failure: post diagnostic comment on issue, add `auto-bmad-failed` label
  - [ ] 3.3 On success: post summary comment with cost/usage metrics on the issue
  - [ ] 3.4 Add `auto-bmad-running` label while workflow is active (remove on completion)
  - [ ] 3.5 Prevent duplicate runs: skip if `auto-bmad-running` label is already present
  - [ ] 3.6 Create `auto-bmad` and `auto-bmad-failed` and `auto-bmad-running` labels in the repo

- [ ] Task 4: Configure repository secrets and variables (AC: #1)
  - [ ] 4.1 **If Option A:** Document required secret: `ANTHROPIC_API_KEY`
  - [ ] 4.2 **If Option B:** Document Claude Code platform auth setup (may replace API key secret)
  - [ ] 4.3 Document optional repo variable: `AUTO_BMAD_TIMEOUT_MINUTES` (default 30)
  - [ ] 4.4 Document optional repo variable: `AUTO_BMAD_MAX_RETRIES` (default 3, for quality gate retry loops)

- [ ] Task 5: Test end-to-end flow (AC: #1, #2, #3, #4)
  - [ ] 5.1 Create a test issue with `auto-bmad` label (small bug fix or trivial feature)
  - [ ] 5.2 Verify workflow triggers, agent runs, PR is created
  - [ ] 5.3 Verify PR can be reviewed and merged from GitHub mobile
  - [ ] 5.4 Verify release pipeline triggers after merge
  - [ ] 5.5 Test failure path: create an issue with ambiguous/impossible scope, verify failure handling

- [ ] Task 6: Document the autonomous workflow (AC: #1, #2, #3, #4, #5)
  - [ ] 6.1 Add autonomous flow section to `_qsprocess/workflows/development-lifecycle.md`
  - [ ] 6.2 Update `_qsprocess/rules/project-rules.md` workflow routing table
  - [ ] 6.3 Document issue authoring guidelines for mobile (what makes a good auto-bmad issue)

## Dev Notes

### This is an infrastructure/CI story -- no production Python code changes

All changes are in `.github/workflows/`, documentation, and repo configuration. No changes to `custom_components/quiet_solar/`.

### Execution Environment: Two Options

Evaluate both options at implementation time. Option B is preferred if it meets all requirements, since it eliminates infrastructure setup entirely.

#### Option A: GitHub Actions + Claude Code CLI

The workflow runs on `ubuntu-latest` GitHub Actions runners with Claude Code CLI installed.

**Runner setup sequence:**
1. Checkout repo
2. Set up Python (match HA version from `pr-quality.yml`)
3. Create and activate venv, install dependencies (`pip install -r requirements_test.txt`)
4. Install Claude Code CLI (`npm install -g @anthropic-ai/claude-code` or via the official install script)
5. Authenticate via `ANTHROPIC_API_KEY` environment variable
6. Run Claude Code with the autonomous prompt

**Invocation pattern:**
```bash
claude --print --dangerously-skip-permissions \
  "You are working on GitHub issue #${ISSUE_NUMBER}: ${ISSUE_TITLE}
${ISSUE_BODY}
Follow all rules in _qsprocess/rules/project-rules.md.
Route this as a ${INTENT_TYPE} (bug fix or feature).
Create branch QS_${ISSUE_NUMBER}, implement the fix/feature,
run quality gates until passing, then create a PR linking to issue #${ISSUE_NUMBER}.
Do NOT commit story files -- work directly from the issue description."
```

**Key flags:**
- `--print` -- non-interactive output mode
- `--dangerously-skip-permissions` -- required for autonomous execution (no human to approve tool calls)

**Pros:** Full control over runner environment, Python/venv available for quality gates.
**Cons:** Requires CLI installation in CI, API key as GitHub Secret, GitHub Actions minutes cost.

#### Option B: Claude Code for Web (Remote Agent)

Claude Code has web-based remote execution capabilities. Instead of running CLI in a GitHub runner, trigger a Claude Code remote agent that connects to the repo directly.

**How it would work:**
1. GitHub Actions workflow triggers on `auto-bmad` label (same trigger)
2. Instead of installing CLI, the workflow calls the Claude Code remote agent API (or uses the `/schedule` trigger mechanism)
3. The remote agent runs in Anthropic's cloud infrastructure with full repo access
4. It performs the same workflow: branch, implement, quality gates, PR creation

**Pros:** No CLI installation, no API key in GitHub Secrets (auth handled by Claude Code platform), simpler workflow YAML, potentially better agent performance (native environment), built-in cost tracking.
**Cons:** Depends on Claude Code web/remote agent availability and GitHub integration capabilities. Verify at implementation time whether remote agents can: (1) clone/push to private repos, (2) run arbitrary shell commands (pytest, ruff, mypy), (3) create PRs via gh CLI.

**Investigation at implementation time:**
- Check Claude Code remote trigger / schedule capabilities for GitHub repo integration
- Verify whether remote agents have shell access for quality gates
- Determine authentication model (GitHub App, OAuth, deploy key?)
- If Option B covers all requirements, use it and simplify the GitHub Actions workflow to just the trigger + status reporting

### Intent Routing Logic

The workflow (or remote agent prompt) parses the issue to determine intent:
- If the issue has the `bug` label -> route to bug fix flow (quick-dev)
- Otherwise -> route to feature flow (create-story -> dev-story)
- The issue title and body become the agent's task description

### Guardrails

| Guardrail | Implementation |
|-----------|---------------|
| **Timeout** | `timeout-minutes: 30` on the job (configurable via `AUTO_BMAD_TIMEOUT_MINUTES` repo variable) |
| **Duplicate prevention** | Check for `auto-bmad-running` label before starting; add it on start, remove on finish |
| **Failure reporting** | On any non-zero exit: post diagnostic comment, add `auto-bmad-failed` label |
| **Scope limit** | The agent operates within the issue scope only -- prompt explicitly constrains it |
| **Cost visibility** | Post token usage summary as issue comment on completion |
| **No force push** | Agent cannot force-push (project rules already prevent this) |
| **Quality gates required** | Agent must pass all quality gates before PR creation (same as local workflow) |

### Workflow File Structure

```yaml
# .github/workflows/auto-bmad.yml
name: Auto BMad

on:
  issues:
    types: [labeled]

jobs:
  auto-bmad:
    if: github.event.label.name == 'auto-bmad'
    runs-on: ubuntu-latest
    timeout-minutes: ${{ vars.AUTO_BMAD_TIMEOUT_MINUTES || 30 }}
    permissions:
      contents: write
      issues: write
      pull-requests: write
    steps:
      # ... setup, install, run, cleanup
```

### Existing CI/CD Infrastructure to Leverage

| Existing workflow | Role in this story |
|------------------|-------------------|
| `pr-quality.yml` | Validates the PR created by the autonomous agent (same gates as manual PRs) |
| `release.yml` | Triggers automatically when TheDev merges from mobile |
| `issue-triage.yml` | Auto-labels issues by keywords (helps intent routing) |
| `auto-label.yml` | Auto-labels PRs by changed file paths |

### Security Considerations

**Option A (GitHub Actions + CLI):**
- `ANTHROPIC_API_KEY` stored as a GitHub repository secret -- never logged, never exposed in PR output
- The agent runs with `contents: write` permission scoped to this repo only
- `--dangerously-skip-permissions` is required but safe in this context: the runner is ephemeral, the agent is constrained by project rules, and all output goes through PR review

**Option B (Claude Code for Web):**
- Authentication handled by the Claude Code platform (no API key in GitHub Secrets)
- Repo access managed through Claude Code's GitHub integration (GitHub App or OAuth)
- Potentially simpler security model since credentials are managed by Anthropic's platform

**Both options:**
- The agent cannot access other secrets or external services beyond what the project already uses
- All code changes still go through PR review before merge (TheDev must approve)
- No auto-merge -- human approval is always required

### Issue Authoring Guidelines (for mobile)

Good `auto-bmad` issues should:
- Have a clear, specific title (e.g., "Bug: solver ignores off-peak constraints after midnight")
- Include steps to reproduce (for bugs) or clear scope (for features)
- Be self-contained -- the agent has no additional context beyond the issue and codebase
- Be appropriately scoped -- single bug fix or small feature, not multi-story epics

Bad examples (will likely fail):
- "Improve the solver" (too vague)
- "Refactor the entire charger module" (too large)
- "Add support for a new charger protocol" (multi-file, needs design decisions)

### What NOT to Do

- Do NOT modify any production Python code in `custom_components/quiet_solar/`
- Do NOT modify existing CI workflows (`pr-quality.yml`, `release.yml`, etc.) -- the new workflow is additive
- Do NOT store the API key anywhere except GitHub Secrets
- Do NOT enable auto-merge -- TheDev must explicitly approve and merge from mobile
- Do NOT run the agent on issues without the `auto-bmad` label (prevent accidental triggers)
- Do NOT use self-hosted runners in the first iteration -- start with GitHub-hosted runners

### Previous Story Intelligence

**Story 1.1 (Agentic Workflow):** Established the local agentic dev workflow. This story extends it to run headlessly in CI. Same BMad skills, same project rules, same quality gates -- just no human in the loop during execution.

**Story 1.2 (PR Quality Gate):** The `pr-quality.yml` workflow validates PRs. The autonomous PR will go through the same gates. No changes needed.

**Story 1.3 (Release Pipeline):** `release.yml` triggers on version tags. After TheDev merges from mobile, the release flow is unchanged.

**Story 1.7 (Worktrees):** Worktrees are for local parallel development. In CI, each run gets a fresh checkout -- no worktrees needed. The agent uses standard `git checkout -b`.

**Story 1.10 (Lazy Logging):** Demonstrates a mechanical, well-scoped story that could be a good candidate for autonomous execution. The autonomous workflow should handle this type of story well.

### Git Intelligence (Recent Commits)

Recent work pattern: stories are committed to `_bmad-output/implementation-artifacts/` on main, then a worktree is created for implementation. In CI, the agent skips story file creation (works from issue description directly) and creates the branch in the runner's checkout.

### Project Structure Notes

```
.github/workflows/
  auto-bmad.yml              (NEW -- the autonomous workflow)
  pr-quality.yml             (EXISTING -- validates autonomous PRs)
  release.yml                (EXISTING -- triggers on merge)
  issue-triage.yml           (EXISTING -- helps with intent routing)

_qsprocess/workflows/
  development-lifecycle.md   (MODIFIED -- add autonomous flow section)

_qsprocess/rules/
  project-rules.md           (MODIFIED -- add autonomous flow to routing table)
```

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 1.9] -- story definition and open questions
- [Source: _bmad-output/planning-artifacts/architecture.md#CI/CD Pipeline Architecture] -- existing pipeline design
- [Source: _qsprocess/rules/project-rules.md#Workflow Routing] -- routing table to update
- [Source: _qsprocess/workflows/development-lifecycle.md] -- lifecycle phases to document
- [Source: .github/workflows/pr-quality.yml] -- existing PR quality gates
- [Source: .github/workflows/issue-triage.yml] -- existing issue triage (intent routing aid)
- [Source: _bmad-output/project-context.md#Development Lifecycle] -- project rules pointer

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
