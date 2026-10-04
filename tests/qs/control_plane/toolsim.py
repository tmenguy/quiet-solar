"""A simulated world for the tool tests: the wrapped scripts, ``gh``, ``git`` and ``claude --bg``.

Every rule answers through ``FakeRunner``, so each call is counted, and the
probes read the same simulated state the effects changed.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from control_plane.runner import RunResult

from .conftest import Call, FakeClaude, FakeRunner, agent

_MARKER = re.compile(r"<!-- qs-cp-key: [^>]+ -->")


@dataclass
class Sim:
    runner: FakeRunner
    claude: FakeClaude
    main: Path
    wt: Path
    issues: list[dict[str, Any]] = field(default_factory=list)
    prs: list[dict[str, Any]] = field(default_factory=list)
    head: str = "a" * 40
    remote: str | None = None
    pr_state: str = "OPEN"
    registered: set[str] = field(default_factory=set)
    cleanup_status: str = "removed"
    gate_exit: int = 0
    launch_exit: int = 0
    list_on_launch: bool = True
    hold: threading.Event | None = None
    entered: threading.Event = field(default_factory=threading.Event)
    session_seq: int = 0

    def install(self) -> Sim:
        r = self.runner
        r.on(["setup_task.py"], self._setup)
        r.on(["cleanup_worktree.py"], self._cleanup)
        r.on(["git", "-C", str(self.main), "worktree", "list", "--porcelain"], self._worktree_list)
        r.on(["git", "-C", str(self.main), "worktree", "prune"], self._prune)
        r.on(["quality_gate.py"], self._gate)
        r.on(["create_issue.py"], self._create_issue)
        r.on(["gh", "issue", "list"], lambda c: RunResult(0, json.dumps(self.issues), ""))
        r.on(["create_pr.py"], self._create_pr)
        r.on(["gh", "pr", "list"], lambda c: RunResult(0, json.dumps(self.prs), ""))
        r.on(["git", "rev-parse", "HEAD"], lambda c: RunResult(0, self.head + "\n", ""))
        r.on(
            ["git", "ls-remote"],
            lambda c: RunResult(0, f"{self.remote}\trefs/heads/QS_11\n" if self.remote else "", ""),
        )
        r.on(["git", "push"], self._push)
        r.on(["gh", "pr", "view"], self._pr_view)
        r.on(["gh", "pr", "merge"], self._merge)
        r.on(["claude", "--bg"], self._claude_bg)
        return self

    def _block(self) -> None:
        self.entered.set()
        if self.hold is not None:
            assert self.hold.wait(timeout=5)

    created: int = 0

    def _setup(self, call: Call) -> RunResult:
        self._block()
        if not self.wt.exists():
            self.created += 1  # the script is idempotent: a second run creates nothing
        self.wt.mkdir(exist_ok=True)
        self.registered.add(str(self.wt.resolve()))
        return RunResult(0, json.dumps({"worktree_path": str(self.wt), "branch": "QS_11", "issue_number": 11}), "")

    def _cleanup(self, call: Call) -> RunResult:
        self._block()
        if self.cleanup_status == "removed":
            for p in sorted(self.wt.rglob("*"), reverse=True):
                p.unlink() if p.is_file() else p.rmdir()
            self.wt.rmdir()
            self.registered.discard(str(self.wt.resolve()))
        return RunResult(0, json.dumps({"status": self.cleanup_status}), "")

    def _worktree_list(self, call: Call) -> RunResult:
        return RunResult(
            0, "".join(f"worktree {p}\nHEAD x\n\n" for p in [str(self.main), *sorted(self.registered)]), ""
        )

    def _prune(self, call: Call) -> RunResult:
        self.registered = {p for p in self.registered if Path(p).exists()}
        return RunResult(0, "", "")

    def _gate(self, call: Call) -> RunResult:
        self._block()
        return RunResult(self.gate_exit, "\n".join(f"line {i}" for i in range(80)), "")

    def _create_issue(self, call: Call) -> RunResult:
        self._block()
        body = call.argv[call.argv.index("--body") + 1]
        number = 100 + len(self.issues)
        self.issues.append({"number": number, "url": f"https://x/issues/{number}", "body": body})
        return RunResult(0, json.dumps({"issue_number": number, "url": f"https://x/issues/{number}"}), "")

    def _create_pr(self, call: Call) -> RunResult:
        self._block()
        body = call.argv[call.argv.index("--summary") + 1]
        number = 200 + len(self.prs)
        self.prs.append({"number": number, "url": f"https://x/pull/{number}", "body": body, "state": "OPEN"})
        self.remote = self.head
        return RunResult(0, json.dumps({"pr_number": number, "url": f"https://x/pull/{number}"}), "")

    def _push(self, call: Call) -> RunResult:
        self._block()
        self.remote = self.head
        return RunResult(0, "", "")

    def _pr_view(self, call: Call) -> RunResult:
        fields = call.argv[call.argv.index("--json") + 1]
        if fields == "headRefOid":
            return RunResult(0, json.dumps({"headRefOid": self.head}), "")
        merge_commit = {"oid": "m" * 40} if self.pr_state == "MERGED" else None
        return RunResult(0, json.dumps({"state": self.pr_state, "mergeCommit": merge_commit}), "")

    def _merge(self, call: Call) -> RunResult:
        self._block()
        self.pr_state = "MERGED"
        return RunResult(0, "", "")

    def _claude_bg(self, call: Call) -> RunResult:
        self._block()
        if self.launch_exit:
            return RunResult(self.launch_exit, "", "boom")
        if "--resume" in call.argv:
            sid = call.argv[call.argv.index("--resume") + 1]
            name = next((a.name for a in (self.claude.listing or []) if a.session_id == sid), None)
            if name is None:  # resumed from a reaped node: the session keeps its name
                name = self.resumed_name
            self.claude.listing = [a for a in (self.claude.listing or []) if a.name != name]
            new_sid = sid if not getattr(self, "new_session_on_resume", False) else f"{sid}-2"
            if self.list_on_launch:
                self.claude.listing.append(agent(new_sid, name, id="short-r", started_at_ms=9_999_999_999_999))
            return RunResult(0, "", "")
        name = call.argv[call.argv.index("-n") + 1]
        self.session_seq += 1
        if self.list_on_launch:
            self.claude.listing = [
                *(self.claude.listing or []),
                agent(f"S-{name}", name, id=f"short-{self.session_seq}", started_at_ms=9_999_999_999_999),
            ]
        return RunResult(0, "", "")

    resumed_name: str | None = None

    def effects(self, *pattern: str) -> int:
        return len(self.runner.matching(*pattern))


def strip_markers(text: str) -> str:
    return _MARKER.sub("", text)
