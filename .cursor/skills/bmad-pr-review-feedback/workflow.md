---
main_config: '{project-root}/_bmad/bmm/config.yaml'
---

# PR Review Feedback Workflow

**Goal:** Pull PR review comments from GitHub into the local workflow, present them with diff context, and process each interactively — fix, discuss, or reject. Optionally bridge local `/bmad-code-review` findings to GitHub first.

**Your Role:** You are the PR review feedback handler. You fetch review comments, present them clearly, and execute the user's chosen action for each. You never auto-decide — every comment action is the user's choice.

---

## INITIALIZATION

### Configuration Loading

Load config from `{main_config}` and resolve:

- `project_name`, `user_name`
- `communication_language`, `document_output_language`
- `date` as system-generated current datetime
- `project_context` = `**/project-context.md` (load if exists)

### Prerequisites

- `gh` CLI installed and authenticated (`gh auth status`)
- Current directory is inside a git repository on a feature branch (not `main`)
- A PR exists for the current branch

---

## EXECUTION

<workflow>

<step n="1" goal="Detect open PR for current branch">
  <action>Run: `gh pr view --json number,url,title,state,reviewDecision,headRefName`</action>

  <check if="command fails or no PR found">
    <output>No open PR found for the current branch.</output>
    <ask>Would you like to:
1. Create a PR first (follow the development lifecycle Phase 3c)
2. Specify a PR number manually
3. Cancel</ask>

    <check if="user provides PR number">
      <action>Run: `gh pr view {{pr_number}} --json number,url,title,state,reviewDecision,headRefName`</action>
    </check>
    <check if="user chooses cancel">
      <action>HALT</action>
    </check>
  </check>

  <action>Store {{pr_number}}, {{pr_url}}, {{pr_title}}, {{pr_state}}</action>

  <check if="pr_state is not OPEN">
    <output>PR #{{pr_number}} is {{pr_state}}, not OPEN. Cannot process review feedback on a closed/merged PR.</output>
    <action>HALT</action>
  </check>

  <output>PR #{{pr_number}}: {{pr_title}}
URL: {{pr_url}}
Review decision: {{reviewDecision}}</output>

  <action>Extract repo owner and name from: `gh repo view --json owner,name --jq '.owner.login + "/" + .name'`</action>
  <action>Store {{repo_owner}}, {{repo_name}}</action>
</step>

<step n="2" goal="Offer to post local code review findings to GitHub">
  <ask>Would you like to:
1. **Post local review** -- Run `/bmad-code-review` and post findings as PR comments (Phase A)
2. **Process existing feedback** -- Pull and process review comments already on the PR (Phase B)
3. **Both** -- Run local review, post findings, then process all feedback interactively</ask>

  <check if="user chooses 1 or 3 (post local review)">
    <output>Run `/bmad-code-review` first, then return here to post findings.</output>
    <action>HALT with message: "Run `/bmad-code-review` now. When complete, run `/bmad-pr-review-feedback` again and choose option 2 to process the feedback. To post the code review findings to GitHub as PR comments, copy the patch findings from the code review output and paste them when prompted."</action>
  </check>

  <check if="user chooses 2">
    <goto step="3" />
  </check>
</step>

<step n="3" goal="Pull unresolved review comments from GitHub">
  <critical>Fetch ALL review threads, filter to unresolved only</critical>

  <action>Fetch review threads using GraphQL to get resolution status:

```bash
gh api graphql -f query='
query($owner: String!, $repo: String!, $pr: Int!) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $pr) {
      reviewThreads(first: 100) {
        nodes {
          id
          isResolved
          isOutdated
          path
          line
          startLine
          diffSide
          comments(first: 50) {
            nodes {
              id
              body
              author { login }
              createdAt
              url
            }
          }
        }
      }
    }
  }
}' -f owner='{{repo_owner}}' -f repo='{{repo_name}}' -F pr={{pr_number}}
```
  </action>

  <action>Parse the response and filter to threads where `isResolved == false`</action>
  <action>For each unresolved thread, extract:
    - `thread_id` (GraphQL node ID, needed for resolve mutation)
    - `path` (file path)
    - `line` / `startLine` (line numbers in the diff)
    - `is_outdated` (whether the comment is on outdated code)
    - `comments` array (author, body, timestamp, URL for each comment in thread)
  </action>

  <action>Store as {{unresolved_threads}} list</action>

  <check if="no unresolved threads found">
    <output>No unresolved review comments on PR #{{pr_number}}. Nothing to process.</output>
    <action>HALT</action>
  </check>

  <output>Found {{thread_count}} unresolved review thread(s) on PR #{{pr_number}}.</output>
</step>

<step n="4" goal="Present each comment with diff context and prompt for action">
  <critical>Process threads one at a time. For each thread, show context and wait for user decision.</critical>

  <action>Initialize counters: {{fix_count}} = 0, {{discuss_count}} = 0, {{reject_count}} = 0</action>
  <action>Initialize {{fixes_made}} = false</action>

  <action>For each thread in {{unresolved_threads}}:</action>

  <!-- Show context -->
  <action>Display the file path and line number</action>
  <action>Show the relevant code context by reading the file at the specified path and lines:
    - Read ~5 lines before and after the comment's line range
    - If the file or lines don't exist (outdated comment), note this
  </action>
  <action>Display all comments in the thread (author, body, timestamp)</action>
  <action>If thread is outdated, note: "(this comment is on outdated code -- the file has changed since the review)"</action>

  <output>
**Thread {{current_index}}/{{thread_count}}** -- `{{path}}:{{line}}`
{{outdated_notice}}

```
{{code_context}}
```

**Review comment** by @{{author}} ({{timestamp}}):
> {{comment_body}}

{{additional_replies_if_any}}

**Actions:** [F]ix -- implement the suggestion | [D]iscuss -- reply on the PR | [R]eject -- dismiss with rationale | [S]kip -- defer to later
  </output>

  <ask>Choose action for this thread:</ask>

  <!-- Handle FIX -->
  <check if="user chooses Fix">
    <action>Understand the reviewer's suggestion from the comment</action>
    <action>Implement the requested change in the codebase</action>
    <action>Run quality gates:
```bash
source venv/bin/activate
pytest tests/ --cov=custom_components/quiet_solar --cov-report=term-missing
ruff check custom_components/quiet_solar/
ruff format --check custom_components/quiet_solar/
mypy custom_components/quiet_solar/
```
    </action>
    <action>If quality gates fail, fix until they pass</action>
    <action>Stage and commit the fix:
```bash
git add -A
git commit -m "fix: address review comment on {{path}}:{{line}}"
```
    </action>
    <action>Push the commit: `git push`</action>
    <action>Post reply on the PR thread:
```bash
gh api graphql -f query='
mutation($threadId: ID!, $body: String!) {
  addPullRequestReviewThread(input: {pullRequestReviewThreadId: $threadId, body: $body}) {
    comment { id }
  }
}' -f threadId='{{thread_id}}' -f body='Fixed in latest push.'
```
    </action>
    <action>Resolve the thread:
```bash
gh api graphql -f query='
mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) {
    thread { isResolved }
  }
}' -f threadId='{{thread_id}}'
```
    </action>
    <action>Increment {{fix_count}}, set {{fixes_made}} = true</action>
  </check>

  <!-- Handle DISCUSS -->
  <check if="user chooses Discuss">
    <ask>What would you like to reply?</ask>
    <action>Post the reply on the PR thread:
```bash
gh api graphql -f query='
mutation($threadId: ID!, $body: String!) {
  addPullRequestReviewThread(input: {pullRequestReviewThreadId: $threadId, body: $body}) {
    comment { id }
  }
}' -f threadId='{{thread_id}}' -f body='{{user_reply}}'
```
    </action>
    <action>Thread remains open for further discussion</action>
    <action>Increment {{discuss_count}}</action>
  </check>

  <!-- Handle REJECT -->
  <check if="user chooses Reject">
    <ask>Provide rationale for dismissing this comment:</ask>
    <action>Post the rationale on the PR thread:
```bash
gh api graphql -f query='
mutation($threadId: ID!, $body: String!) {
  addPullRequestReviewThread(input: {pullRequestReviewThreadId: $threadId, body: $body}) {
    comment { id }
  }
}' -f threadId='{{thread_id}}' -f body='Dismissed: {{user_rationale}}'
```
    </action>
    <action>Resolve the thread:
```bash
gh api graphql -f query='
mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) {
    thread { isResolved }
  }
}' -f threadId='{{thread_id}}'
```
    </action>
    <action>Increment {{reject_count}}</action>
  </check>

  <!-- Handle SKIP -->
  <check if="user chooses Skip">
    <action>Move to next thread without action</action>
  </check>

  <action>Continue to next thread</action>
</step>

<step n="5" goal="Summary report and final quality gates">
  <output>
## Review Feedback Summary -- PR #{{pr_number}}

| Action   | Count |
|----------|-------|
| Fixed    | {{fix_count}} |
| Discussed | {{discuss_count}} |
| Rejected | {{reject_count}} |
| Skipped  | {{skip_count}} |
| **Total** | **{{thread_count}}** |
  </output>

  <check if="fixes_made == true">
    <output>Fixes were made -- re-running quality gates...</output>
    <action>Run full quality gate suite:
```bash
source venv/bin/activate
pytest tests/ --cov=custom_components/quiet_solar --cov-report=term-missing
ruff check custom_components/quiet_solar/
ruff format --check custom_components/quiet_solar/
mypy custom_components/quiet_solar/
```
    </action>

    <check if="quality gates pass">
      <output>All quality gates pass. Fixes are clean.</output>
    </check>

    <check if="quality gates fail">
      <output>Quality gates failed after fixes. Please address the failures before the PR can be merged.</output>
      <action>Show failing gate output for user to review</action>
    </check>
  </check>

  <check if="fixes_made == false">
    <output>No code changes were made -- quality gates not re-run.</output>
  </check>

  <output>
**Next steps:**
- If discussions are open, wait for reviewer replies and run this skill again
- If all threads are resolved, the PR is ready for merge
- Run `/bmad-code-review` with a different LLM for additional coverage
  </output>
</step>

</workflow>
