# Auto BMad — Mobile-First Autonomous Development

Auto BMad lets you drive the full development lifecycle from your phone: create a GitHub issue, label it, and a cloud-based AI agent implements the fix, runs quality gates, and opens a PR for your review.

## How It Works

```
Phone: create issue + apply "auto-bmad" label
  --> GitHub Actions detects the label
  --> Runner installs Python, Claude Code CLI
  --> Claude Code reads the issue, creates a branch, implements the change
  --> Runs tests (100% coverage), ruff, mypy — retries if needed
  --> Creates a PR linking to the issue
Phone: review PR, approve, merge
  --> Release pipeline triggers automatically
```

No local terminal, no IDE, no laptop required.

## Setup

### 1. Get an Anthropic API Key

1. Go to [console.anthropic.com](https://console.anthropic.com/)
2. Sign in or create an account
3. Navigate to **API Keys** in the left sidebar
4. Click **Create Key**, name it (e.g., "quiet-solar-auto-bmad")
5. Copy the key (starts with `sk-ant-...`) — you only see it once

### 2. Add the Secret to GitHub

1. Go to your repository on GitHub
2. **Settings** > **Secrets and variables** > **Actions**
3. Click **New repository secret**
4. Name: `ANTHROPIC_API_KEY`
5. Value: paste the API key
6. Click **Add secret**

The key is encrypted by GitHub and never exposed in logs or PR output.

### 3. Optional Configuration

You can customize behavior via **Settings > Secrets and variables > Actions > Variables** (repository variables, not secrets):

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTO_BMAD_TIMEOUT_MINUTES` | 30 | Maximum time the agent can run before being killed |
| `AUTO_BMAD_MAX_RETRIES` | 3 | How many times the agent retries failing quality gates |

## Usage

### Creating an Issue for Auto BMad

From the GitHub mobile app (or any GitHub client):

1. Create a new issue
2. Write a clear title and description
3. Apply the `auto-bmad` label

The workflow triggers automatically and posts status updates as issue comments.

### Writing Good Issues

The agent has no context beyond the issue description and the codebase. Be specific:

**Good examples:**
- "Bug: solver ignores off-peak constraints when they span midnight"
- "Add a sensor entity that exposes the current solar surplus in watts"
- "Fix: car charge target resets to 100% when person disconnects"

**Bad examples (will likely fail):**
- "Improve the solver" (too vague)
- "Refactor the entire charger module" (too large)
- "Add support for a new charger protocol" (needs design decisions)

### Labeling for Intent Routing

The agent routes automatically based on labels:
- Issue has `bug` label -> bug fix flow (quick, focused)
- No `bug` label -> feature flow (broader implementation)

The `issue-triage.yml` workflow auto-labels issues by keywords (e.g., "charger" gets `area:charger`), so you only need to add `auto-bmad` and optionally `bug`.

### Monitoring a Run

While the agent works:
- The issue gets an `auto-bmad-running` label (prevents duplicate runs)
- When done, a comment is posted with the outcome and duration
- On success: a PR is created for your review
- On failure: a diagnostic comment explains what went wrong, and `auto-bmad-failed` is added

### Reviewing and Merging

1. Open the PR from the GitHub mobile app
2. Review the changes, check the summary
3. The PR goes through the same quality gates as any manual PR (`pr-quality.yml`)
4. Approve and merge when satisfied
5. The release pipeline triggers automatically on merge + version tag

### Retrying After Failure

If the agent fails:
1. Read the failure comment for clues
2. Check the [workflow run logs](../../actions) for details
3. Narrow the issue scope or add clarifying details if needed
4. Re-apply the `auto-bmad` label to trigger another run

## Guardrails

| Protection | How It Works |
|------------|-------------|
| **Timeout** | Agent is killed after 30 minutes (configurable) |
| **Duplicate prevention** | `auto-bmad-running` label blocks concurrent runs |
| **Quality gates** | Same as local dev: 100% test coverage, ruff, mypy |
| **No auto-merge** | You must always review and approve manually |
| **No force push** | Agent follows project rules — cannot force-push or amend |
| **Scope containment** | Agent only works on files related to the issue |
| **API key security** | Encrypted GitHub secret, never logged or exposed |

## Cost

Each autonomous run uses Anthropic API credits. Typical costs:
- Simple bug fix: $1-5
- Small feature: $3-10
- Complex changes that hit retry limits: $10-20

Monitor usage at [console.anthropic.com/usage](https://console.anthropic.com/usage).

## Limitations

- **Token usage not reported in comments** — the Claude CLI does not expose token metrics in its output. Duration is reported instead.
- **GitHub Actions minutes consumed** — each run uses runner minutes from your GitHub plan.
- **GITHUB_TOKEN PRs** — PRs created by the workflow use `GITHUB_TOKEN`, which cannot trigger other `pull_request`-triggered workflows (GitHub platform limitation). The `pr-quality.yml` gates run on the PR's push events, so this typically works, but be aware of this constraint.
- **Not for large or ambiguous tasks** — the agent works best on well-scoped, single-purpose changes. Multi-story epics or architectural decisions need human guidance.
