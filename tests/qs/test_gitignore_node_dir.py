"""QS-424 AC7: a node's worktree-local files (`.qs-node/`) are ignored by the repository's `.gitignore`.

Unlike `test_gitignore_claude_local_settings.py`, this pin asks git itself
(`git check-ignore -v`), and then asserts the matching rule's **source**: the
repository's `.gitignore` with the `/.qs-node/` pattern. A global excludes file
or `.git/info/exclude` holding the same pattern therefore cannot make it pass.
`check-ignore` needs no existing path.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NODE_FILES = (".qs-node/plan.md", ".qs-node/digest.json", ".qs-node/binding.json", ".qs-node/round.json")


@pytest.mark.parametrize("path", NODE_FILES)
def test_a_node_file_is_ignored_by_the_repository_gitignore(path: str) -> None:
    res = subprocess.run(
        ["git", "check-ignore", "-v", path], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert res.returncode == 0, f"{path} is not ignored: {res.stderr or res.stdout}"
    rule, _, matched = res.stdout.rstrip("\n").partition("\t")
    source, _line, pattern = rule.split(":", 2)
    assert (source, pattern, matched) == (".gitignore", "/.qs-node/", path)
