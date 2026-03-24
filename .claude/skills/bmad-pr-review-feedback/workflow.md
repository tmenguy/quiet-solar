---
main_config: '{project-root}/_bmad/bmm/config.yaml'
---

# PR Review Feedback Workflow

**Goal:** Pull PR review comments from GitHub into the local workflow, present them with diff context, and process each interactively — fix, discuss, or reject. Optionally bridge local `/bmad-code-review` findings to GitHub first.

**Your Role:** You are the PR review feedback handler. You fetch review comments, present them clearly, and execute the user's chosen action for each. You never auto-decide — every comment action is the user's choice. You never auto-commit — the user confirms every commit.

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
  <!-- P11: Active branch check — prevent running from main -->
  <action>Check current branch: `git branch --show-current`</action>
  <check if="current branch is main or master">
    <output>You are on the `main` branch. This skill must run from a feature branch with an open PR.</output>
    <ask>Switch to a feature branch first, or provide a PR number to check out its branch.</ask>
    <action>HALT</action>
  </check>

  <action>Run: `gh pr view --json number,url,title,state,reviewDecision,headRefName`</action>

  <check if="command fails or no PR found">
    <output>No open PR found for the current branch.</output>
    <ask>Would you like to:
1. Create a PR first (run Phase 3c from development-lifecycle.md, then re-run this skill)
2. Specify a PR number manually
3. Cancel</ask>

    <check if="user chooses 1">
      <output>Create the PR first following Phase 3c in development-lifecycle.md, then re-run `/bmad-pr-review-feedback`.</output>
      <action>HALT</action>
    </check>
    <check if="user provides PR number">
      <action>Run: `gh pr view {{pr_number}} --json number,url,title,state,reviewDecision,headRefName`</action>
      <!-- Verify PR branch matches current branch -->
      <check if="headRefName does not match current branch">
        <output>Warning: PR #{{pr_number}} is for branch `{{headRefName}}` but you are on `{{current_branch}}`. Fixes would be committed to the wrong branch.</output>
        <ask>Continue anyway, or switch to branch `{{headRefName}}` first?</ask>
      </check>
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

<!-- P5/P6: Simplified Phase A — removed dead "Both" option, honest about Phase A scope -->
<step n="2" goal="Choose workflow phase">
  <ask>Would you like to:
1. **Process existing feedback** — Pull and process review comments already on the PR
2. **Run local review first** — Run `/bmad-code-review`, then come back to process feedback</ask>

  <check if="user chooses 2 (run local review first)">
    <output>Run `/bmad-code-review` now. When the review is complete, re-run `/bmad-pr-review-feedback` and choose option 1 to process the feedback interactively.</output>
    <action>HALT</action>
  </check>

  <check if="user chooses 1">
    <goto step="3" />
  </check>
</step>

<step n="3" goal="Pull unresolved review comments from GitHub">
  <critical>Fetch ALL review threads and top-level reviews, filter to unresolved/actionable only</critical>

  <!-- D1: Pagination support — fetch with pageInfo, loop if needed -->
  <!-- D2: Also fetch top-level review bodies and PR conversation comments -->
  <action>Fetch review threads, top-level reviews, and PR comments using GraphQL:

```bash
gh api graphql -f query='
query($owner: String!, $repo: String!, $pr: Int!) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $pr) {
      reviewThreads(first: 100) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id
          isResolved
          isOutdated
          path
          line
          startLine
          diffSide
          comments(first: 50) {
            pageInfo { hasNextPage endCursor }
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
      reviews(last: 20) {
        nodes {
          body
          state
          author { login }
          createdAt
        }
      }
      comments(last: 20) {
        nodes {
          body
          author { login }
          createdAt
        }
      }
    }
  }
}' -f owner='{{repo_owner}}' -f repo='{{repo_name}}' -F pr={{pr_number}}
```
  </action>

  <!-- P7: Error handling for GraphQL calls -->
  <check if="GraphQL call fails or returns errors">
    <output>GitHub API call failed: {{error_message}}</output>
    <output>Possible causes: authentication expired (`gh auth status`), rate limit, network issue, or insufficient permissions.</output>
    <action>HALT</action>
  </check>

  <!-- D1: Pagination — if hasNextPage, warn and fetch more -->
  <check if="reviewThreads.pageInfo.hasNextPage is true">
    <output>Note: This PR has more than 100 review threads. Fetching additional pages...</output>
    <action>Fetch remaining pages using the endCursor until hasNextPage is false</action>
  </check>

  <action>Parse the response and filter to threads where `isResolved == false`</action>
  <action>For each unresolved thread, extract:
    - `thread_id` (GraphQL node ID, needed for resolve mutation)
    - `path` (file path)
    - `line` / `startLine` (line numbers — these are file-relative on the side indicated by `diffSide`)
    - `is_outdated` (whether the comment is on outdated code)
    - `comments` array (author, body, timestamp, URL for each comment in thread)
  </action>

  <!-- D2: Surface top-level review bodies with actionable content -->
  <action>Check top-level reviews for non-empty body text (especially CHANGES_REQUESTED or COMMENT reviews). If any have substantive body text, present them separately as "Top-level review feedback" before the inline threads.</action>

  <action>Store as {{unresolved_threads}} list</action>

  <check if="no unresolved threads found AND no actionable top-level reviews">
    <output>No unresolved review comments on PR #{{pr_number}}. Nothing to process.</output>
    <action>HALT</action>
  </check>

  <output>Found {{thread_count}} unresolved review thread(s) on PR #{{pr_number}}.</output>
  <check if="top-level reviews have actionable content">
    <output>Also found {{review_count}} top-level review comment(s) with substantive feedback.</output>
  </check>
</step>

<step n="4" goal="Present each comment with diff context and prompt for action">
  <critical>Process threads one at a time. For each thread, show context and wait for user decision.</critical>
  <critical>NEVER commit or push without explicit user confirmation.</critical>

  <!-- P2: Initialize ALL counters including skip_count -->
  <action>Initialize counters: {{fix_count}} = 0, {{discuss_count}} = 0, {{reject_count}} = 0, {{skip_count}} = 0</action>
  <action>Initialize {{fixes_made}} = false, {{fix_attempts}} = 0</action>

  <!-- D2: Present top-level review feedback first if any -->
  <check if="top-level reviews have actionable content">
    <output>**Top-level review feedback** (not tied to specific lines):</output>
    <action>For each top-level review with non-empty body:
      Display: **@{{author}}** ({{state}}, {{timestamp}}):
      > {{body}}
    </action>
    <output>These are informational. Inline thread processing follows.</output>
  </check>

  <action>For each thread in {{unresolved_threads}}:</action>

  <!-- P12: Handle null line and deleted file edge cases -->
  <action>Display the file path and line number (if available)</action>
  <action>Show the relevant code context:
    - If `line` is not null AND the file at `path` exists: read ~5 lines before and after the comment's line range
    - If `line` is null: this is a file-level comment — show the first ~10 lines of the file or note "file-level comment, no specific line"
    - If the file at `path` does not exist: check if renamed (`git log --diff-filter=R --find-renames -- {{path}}`), show the renamed path if found, or note "file was deleted or renamed"
    - If the thread is outdated: note that the code has changed since the review and show current file content at the approximate location
  </action>
  <!-- Display ALL comments in thread with proper attribution -->
  <action>Display every comment in the thread with individual attribution:
    For each comment in thread.comments:
      **@{{comment.author.login}}** ({{comment.createdAt}}):
      > {{comment.body}}
  </action>
  <action>If thread is outdated, note: "(this comment is on outdated code — the file has changed since the review)"</action>

  <output>
**Thread {{current_index}}/{{thread_count}}** — `{{path}}{{line_display}}`
{{outdated_notice}}

```
{{code_context}}
```

{{all_comments_with_attribution}}

**Actions:** [F]ix — implement the suggestion | [D]iscuss — reply on the PR | [R]eject — dismiss with rationale | [S]kip — defer to later
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

    <!-- P8: Bounded fix loop with escape hatch (max 3 attempts) -->
    <action>Set {{fix_attempts}} = 1</action>
    <check if="quality gates fail">
      <action>Fix the failures and re-run quality gates</action>
      <action>Increment {{fix_attempts}}</action>
      <check if="fix_attempts > 3 AND quality gates still fail">
        <output>Quality gates still failing after 3 fix attempts.</output>
        <ask>Would you like to:
1. Continue trying to fix
2. Revert the fix and switch to Discuss or Reject for this thread
3. Skip this thread for now</ask>
        <check if="user chooses 2">
          <action>Run: `git checkout -- .` to revert changes</action>
          <action>Re-prompt for Discuss or Reject action</action>
        </check>
        <check if="user chooses 3">
          <action>Run: `git checkout -- .` to revert changes</action>
          <action>Increment {{skip_count}}</action>
          <action>Continue to next thread</action>
        </check>
      </check>
    </check>

    <!-- P3/P4: NO auto-commit — stage specific files and ask user to confirm -->
    <action>Stage only the files that were changed for this fix: `git add {{changed_files}}`</action>
    <output>Fix implemented and quality gates pass. Changes staged:</output>
    <action>Show `git diff --cached --stat` to the user</action>
    <ask>Commit and push this fix? [Y/N]</ask>
    <check if="user confirms">
      <action>Commit:
```bash
git commit -m "fix: address review comment on {{path}}:{{line}}"
```
      </action>
      <action>Push: `git push`</action>
      <!-- P1: Correct mutation — addPullRequestReviewThreadReply (not addPullRequestReviewThread) -->
      <!-- P10: Use heredoc to avoid shell metacharacter issues in body text -->
      <action>Post reply on the PR thread:
```bash
gh api graphql -f query='
mutation($threadId: ID!, $body: String!) {
  addPullRequestReviewThreadReply(input: {pullRequestReviewThreadId: $threadId, body: $body}) {
    comment { id }
  }
}' -f threadId='{{thread_id}}' -f body='Fixed in latest push.'
```
      </action>
      <!-- P7: Check for errors on the reply mutation -->
      <check if="reply mutation fails">
        <output>Warning: Failed to post reply on GitHub ({{error}}). The fix was pushed but the comment was not replied to. You can reply manually.</output>
      </check>
      <!-- D3: Resolve thread — may fail if branch protection requires reviewer to resolve -->
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
      <check if="resolve mutation fails">
        <output>Warning: Could not resolve thread (may require reviewer to resolve per branch protection rules). The fix was pushed and reply posted — the reviewer can resolve the thread.</output>
      </check>
    </check>
    <check if="user declines commit">
      <output>Changes are staged but not committed. You can commit manually later.</output>
    </check>
    <action>Increment {{fix_count}}, set {{fixes_made}} = true</action>
  </check>

  <!-- Handle DISCUSS -->
  <check if="user chooses Discuss">
    <ask>What would you like to reply?</ask>
    <!-- P1: Correct mutation — addPullRequestReviewThreadReply -->
    <!-- P10: Pass body via -f which handles escaping; for complex text, agent should use a temp file -->
    <action>Post the reply on the PR thread. For replies containing special characters (quotes, backticks, $), write the body to a temp file and use `--field body=@/tmp/reply.txt`:
```bash
gh api graphql -f query='
mutation($threadId: ID!, $body: String!) {
  addPullRequestReviewThreadReply(input: {pullRequestReviewThreadId: $threadId, body: $body}) {
    comment { id }
  }
}' -f threadId='{{thread_id}}' -f body='{{user_reply}}'
```
    </action>
    <!-- P7: Error handling -->
    <check if="reply mutation fails">
      <output>Warning: Failed to post reply on GitHub ({{error}}). You can reply manually on the PR.</output>
    </check>
    <action>Thread remains open for further discussion</action>
    <action>Increment {{discuss_count}}</action>
  </check>

  <!-- Handle REJECT -->
  <check if="user chooses Reject">
    <ask>Provide rationale for dismissing this comment:</ask>
    <!-- P1: Correct mutation — addPullRequestReviewThreadReply -->
    <action>Post the rationale on the PR thread:
```bash
gh api graphql -f query='
mutation($threadId: ID!, $body: String!) {
  addPullRequestReviewThreadReply(input: {pullRequestReviewThreadId: $threadId, body: $body}) {
    comment { id }
  }
}' -f threadId='{{thread_id}}' -f body='Dismissed: {{user_rationale}}'
```
    </action>
    <!-- P7: Error handling -->
    <check if="reply mutation fails">
      <output>Warning: Failed to post rationale on GitHub ({{error}}). You can reply manually on the PR.</output>
    </check>
    <!-- D3: Resolve — may be blocked by branch protection -->
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
    <check if="resolve mutation fails">
      <output>Warning: Could not resolve thread (may require reviewer to resolve per branch protection rules). Rationale was posted — the reviewer can resolve the thread.</output>
    </check>
    <action>Increment {{reject_count}}</action>
  </check>

  <!-- Handle SKIP — P2: properly increment skip_count -->
  <check if="user chooses Skip">
    <action>Increment {{skip_count}}</action>
    <action>Move to next thread without action</action>
  </check>

  <action>Continue to next thread</action>
</step>

<step n="5" goal="Summary report and final quality gates">
  <output>
## Review Feedback Summary — PR #{{pr_number}}

| Action    | Count |
|-----------|-------|
| Fixed     | {{fix_count}} |
| Discussed | {{discuss_count}} |
| Rejected  | {{reject_count}} |
| Skipped   | {{skip_count}} |
| **Total** | **{{thread_count}}** |
  </output>

  <check if="fixes_made == true">
    <output>Fixes were made — running final quality gates...</output>
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
    <output>No code changes were made — quality gates not re-run.</output>
  </check>

  <output>
**Next steps:**
- If discussions are open, wait for reviewer replies and run this skill again
- If all threads are resolved, the PR is ready for merge
- Run `/bmad-code-review` with a different LLM for additional coverage
  </output>
</step>

</workflow>