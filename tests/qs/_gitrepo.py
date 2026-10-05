"""A real-git fixture shared by the QS-400 tests (worktree-setup.sh, integrate_item.py).

A bare ``origin`` (``init --bare -b main``) plus a clone holding a copy of
``scripts/worktree-setup.sh`` and a tracked ``custom_components/quiet_solar``;
an isolated git config (``GIT_CONFIG_GLOBAL``, ``GIT_CONFIG_NOSYSTEM``);
untracked seed sources ``venv/``, ``config/<x>``, ``custom_components/<other>/``,
``.mypy_cache/`` and ``.testmondata``. The seed ``venv/bin/python`` is a fake
gate whose behaviour is chosen by ``QS_FAKE_GATE``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "worktree-setup.sh"
BASH = "/bin/bash" if Path("/bin/bash").exists() else (shutil.which("bash") or "bash")

FAKE_PYTHON = """#!/bin/sh
# Fake gate interpreter for the QS-400 tests: behaviour chosen by QS_FAKE_GATE.
case "${QS_FAKE_GATE:-green}" in
    green) echo "gate ok"; exit 0 ;;
    red) i=0; while [ $i -lt 50 ]; do echo "red line $i"; i=$((i+1)); done; exit 1 ;;
    commit) git commit -q --allow-empty -m "gate commit"; exit 0 ;;
    touch) echo changed >> README.md; exit 0 ;;
    sleep) echo $$ > "$QS_FAKE_DIR/gate.pids"; sleep 1000 & echo $! >> "$QS_FAKE_DIR/gate.pids"; wait; exit 0 ;;
    killparent) kill -9 $PPID; exit 0 ;;
esac
exit 0
"""


@dataclass
class Repo:
    root: Path
    origin: Path
    clone: Path
    env: dict[str, str]

    @property
    def worktrees(self) -> Path:
        return self.clone.parent / f"{self.clone.name}-worktrees"

    def git(self, *args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd or self.clone),
            env=self.env,
            capture_output=True,
            text=True,
            check=check,
        )

    def rev(self, ref: str, cwd: Path | None = None) -> str:
        return self.git("rev-parse", ref, cwd=cwd).stdout.strip()

    def setup(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [BASH, str(self.clone / "scripts" / "worktree-setup.sh"), *args],
            cwd=str(self.clone),
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )


def make_repo(tmp_path: Path) -> Repo:
    """Bare origin + clone; tracked script and quiet_solar; untracked seed sources."""
    root = tmp_path.resolve()
    gitconfig = root / "gitconfig"
    gitconfig.write_text(
        "[user]\n\tname = QS Test\n\temail = qs@example.invalid\n"
        "[init]\n\tdefaultBranch = main\n[advice]\n\tdetachedHead = false\n"
    )
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": str(gitconfig),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
        env.pop(key, None)
    origin = root / "origin.git"
    clone = root / "repo"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], env=env, check=True)
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], env=env, check=True, capture_output=True)
    repo = Repo(root=root, origin=origin, clone=clone, env=env)

    (clone / "scripts").mkdir()
    shutil.copy2(SCRIPT, clone / "scripts" / "worktree-setup.sh")
    (clone / "custom_components" / "quiet_solar").mkdir(parents=True)
    (clone / "custom_components" / "quiet_solar" / "__init__.py").write_text('"""qs."""\n')
    (clone / ".gitignore").write_text(
        "venv\nvenv/\nconfig/\n.mypy_cache/\n.testmondata\ncustom_components/other\n__pycache__/\n"
    )
    (clone / "README.md").write_text("base\n")
    repo.git("add", "-A")
    repo.git("commit", "-q", "-m", "base")
    repo.git("push", "-q", "-u", "origin", "main")

    # Untracked seed sources.
    (clone / "venv" / "bin").mkdir(parents=True)
    fake_python = clone / "venv" / "bin" / "python"
    fake_python.write_text(FAKE_PYTHON)
    fake_python.chmod(0o755)
    (clone / "config").mkdir()
    (clone / "config" / "secrets.yaml").write_text("x: 1\n")
    (clone / "custom_components" / "other").mkdir()
    (clone / "custom_components" / "other" / "__init__.py").write_text("")
    (clone / ".mypy_cache").mkdir()
    (clone / ".mypy_cache" / "cache.json").write_text("{}")
    (clone / ".testmondata").write_text("testmon")
    return repo
