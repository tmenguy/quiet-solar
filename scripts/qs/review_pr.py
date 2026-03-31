#!/usr/bin/env python3
"""Fetch PR review comments and optionally wait for CodeRabbit review.

Usage:
    python scripts/qs/review_pr.py <pr_number> [--fetch-comments] [--wait-coderabbit N]

Output: JSON with structured review comments.
"""

from __future__ import annotations

import argparse
import json
import time

from utils import output_json, run_gh


def get_repo_owner_name() -> tuple[str, str]:
    """Get the owner and repo name from gh CLI."""
    result = run_gh(["repo", "view", "--json", "owner,name", "--jq", ".owner.login + \"/\" + .name"])
    parts = result.stdout.strip().split("/")
    return parts[0], parts[1]


def fetch_pr_comments(pr_number: int) -> list[dict]:
    """Fetch all unresolved review threads on a PR."""
    owner, repo = get_repo_owner_name()

    # Use gh api graphql with properly resolved owner/repo
    query = """query($owner: String!, $repo: String!, $pr: Int!) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $pr) {
      reviewThreads(first: 100) {
        nodes {
          isResolved
          comments(first: 10) {
            nodes {
              author { login }
              body
              path
              line
              diffHunk
            }
          }
        }
      }
    }
  }
}"""

    result = run_gh([
        "api", "graphql",
        "-F", f"owner={owner}",
        "-F", f"repo={repo}",
        "-F", f"pr={pr_number}",
        "-f", f"query={query}",
    ], check=False)

    if result.returncode != 0:
        return fetch_pr_comments_simple(pr_number)

    try:
        data = json.loads(result.stdout)
        pr_data = data["data"]["repository"]["pullRequest"]
        comments = []

        for thread in pr_data.get("reviewThreads", {}).get("nodes", []):
            if thread["isResolved"]:
                continue
            nodes = thread.get("comments", {}).get("nodes", [])
            if not nodes:
                continue
            first = nodes[0]
            replies = [n["body"] for n in nodes[1:]]
            comments.append({
                "path": first.get("path", ""),
                "line": first.get("line"),
                "author": (first.get("author") or {}).get("login", "unknown"),
                "body": first["body"],
                "diff_context": (first.get("diffHunk", "") or "")[-200:],
                "replies": replies,
                "resolved": False,
            })

        return comments
    except (json.JSONDecodeError, KeyError):
        return fetch_pr_comments_simple(pr_number)


def fetch_pr_comments_simple(pr_number: int) -> list[dict]:
    """Fallback: fetch comments using REST API."""
    result = run_gh([
        "api", f"repos/{{owner}}/{{repo}}/pulls/{pr_number}/comments",
        "--jq", ".[] | {path: .path, line: .line, author: .user.login, body: .body}",
    ], check=False)

    if result.returncode != 0:
        return []

    comments = []
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            try:
                c = json.loads(line)
                c["resolved"] = False
                comments.append(c)
            except json.JSONDecodeError:
                comments.append({"body": line.strip(), "resolved": False})
    return comments


def wait_for_coderabbit(pr_number: int, timeout: int = 120) -> list[dict]:
    """Wait for CodeRabbit review to appear, then fetch its comments."""
    start = time.time()
    while time.time() - start < timeout:
        comments = fetch_pr_comments(pr_number)
        coderabbit_comments = [c for c in comments if c.get("author", "").startswith("coderabbitai")]
        if coderabbit_comments:
            return coderabbit_comments
        time.sleep(10)
    return []


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Review PR")
    parser.add_argument("pr_number", type=int, help="PR number")
    parser.add_argument("--fetch-comments", action="store_true", help="Fetch existing comments")
    parser.add_argument("--wait-coderabbit", type=int, default=0, help="Wait N seconds for CodeRabbit review")
    args = parser.parse_args(argv)

    result_data: dict = {"pr_number": args.pr_number}

    if args.wait_coderabbit > 0:
        result_data["coderabbit_comments"] = wait_for_coderabbit(args.pr_number, args.wait_coderabbit)

    if args.fetch_comments:
        all_comments = fetch_pr_comments(args.pr_number)
        result_data["comments"] = all_comments
        result_data["unresolved_count"] = len([c for c in all_comments if not c.get("resolved")])

    output_json(result_data)


if __name__ == "__main__":
    main()
