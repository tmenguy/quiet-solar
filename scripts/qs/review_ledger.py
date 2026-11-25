#!/usr/bin/env python3
"""Deterministic review-finding ledger (QS-453 D2).

The code-review orchestrators (``qs-review-task``, ``qs-verify-task``)
record every finding, decision and state change of a PR's review rounds
in ``docs/stories/QS-<N>.review_ledger.json`` through this script — the
single writer. Subcommands: ``start-round``, ``add``, ``set``,
``check-round``, ``show`` (all take ``--issue N [--root PATH]``).

Round lifecycle = git: a *finished* round has the ledger clean vs
``HEAD``; a *started* round has it dirty (or untracked) with >= 1 round.

Exit codes: 0 ok; 1 invalid input / transition / unknown ID / not a repo
/ wrong lifecycle state (argparse errors too); 2 corrupt ledger. On exit
1/2 stdout carries ``{"error": "...", ...}``.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from utils import get_repo_root  # type: ignore[import-not-found]

VERSION = 1
DIFF_CAP = 200
SEVERITIES = ("must-fix", "should-fix", "nice-to-have")
_SEV_RANK = {"nice-to-have": 0, "should-fix": 1, "must-fix": 2}
VERDICTS = ("CONFIRMED", "PLAUSIBLE", "REFUTED")
SEVERITY_ACTIONS = ("keep", "demote", "promote")
RELATION_KINDS = ("NEW", "DUPLICATE", "REOPENS", "FLIP-FLOP")
CHECK_STATUSES = ("RESOLVED", "STILL-PRESENT")
MODES = ("verified", "fallback")
REJECTED_BY = ("user", "default-skip", "verifier")
DOC_MAINTENANCE = "orchestrator:doc-maintenance"
SCENARIO_EXEMPT = frozenset({"qs-review-acceptance-auditor", "qs-review-regression-proof", DOC_MAINTENANCE})
_TOP_KEYS = frozenset({"mode", "checks", "findings"})
_CHECK_KEYS = frozenset({"ref", "status", "evidence"})
_FINDING_KEYS = frozenset(
    {
        "key",
        "source",
        "severity",
        "where",
        "title",
        "verdict",
        "evidence",
        "reproduces_when",
        "severity_action",
        "relation",
        "note",
    }
)
_RELATION_KEYS = frozenset({"kind", "ref"})
_RETARGET = (
    "reported_severity",
    "severity_action",
    "severity",
    "where",
    "verdict",
    "evidence",
    "reproduces_when",
    "note",
)
_WHERE_RE = re.compile(r"^(?P<path>[^\s:]+)(:\d+(-\d+)?)?$")
_LIVE = ("accepted", "resolved")


class LedgerError(Exception):
    """An exit-1/2 condition; ``payload`` is printed as JSON."""

    def __init__(self, code: int, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.code = code
        self.payload = {"error": message, **extra}


def _fail(message: str, **extra: Any) -> NoReturn:
    raise LedgerError(1, message, **extra)


@dataclass
class _Ctx:
    root: Path
    issue: int

    @property
    def rel(self) -> str:
        return f"docs/stories/QS-{self.issue}.review_ledger.json"

    @property
    def fix_glob(self) -> str:
        return f"docs/stories/QS-{self.issue}.story_review_fix_*"

    @property
    def path(self) -> Path:
        return self.root / self.rel

    @property
    def excludes(self) -> list[str]:
        return [f":(exclude){self.rel}", f":(exclude){self.fix_glob}"]


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True, check=False)


def resolve_root(arg: str | None) -> Path:
    try:
        base = Path(arg) if arg is not None else get_repo_root()
        proc = subprocess.run(
            ["git", "-C", str(base), "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        raise LedgerError(1, "not a git repo") from exc
    return Path(proc.stdout.strip())


# --- storage seam -----------------------------------------------------------


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": VERSION, "rounds": [], "findings": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise LedgerError(2, "corrupt ledger: invalid JSON", path=str(path)) from exc
    if not isinstance(data, dict) or data.get("version") != VERSION:
        raise LedgerError(2, "corrupt ledger: missing or unsupported version", path=str(path))
    if not isinstance(data.get("rounds"), list) or not isinstance(data.get("findings"), list):
        raise LedgerError(2, "corrupt ledger: rounds/findings must be lists", path=str(path))
    return data


def save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# --- lifecycle ----------------------------------------------------------------


def _dirty(ctx: _Ctx) -> bool:
    return bool(_git(ctx.root, "status", "--porcelain", "--untracked-files=all", "--", ctx.rel).stdout.strip())


def _committed_unfinished(data: dict[str, Any]) -> list[str]:
    """IDs still ``open`` in a ledger whose last round was added (QS-453 fix #01 F-003)."""
    if not data["rounds"] or not data["rounds"][-1].get("added"):
        return []
    return [f["id"] for f in data["findings"] if f["state"] == "open"]


def _require_started(ctx: _Ctx, data: dict[str, Any], *, added: bool, recover: bool = False) -> dict[str, Any]:
    """The started round. ``recover`` (``set`` only) also accepts a *committed
    unfinished* round — a clean ledger whose last round still has ``open``
    findings — so the CLI itself can decide them (fix #01 F-003)."""
    if not data["rounds"] or not (_dirty(ctx) or (recover and _committed_unfinished(data))):
        _fail("no started round (run start-round first)")
    last = data["rounds"][-1]
    if bool(last.get("added")) is not added:
        state = "already added" if last.get("added") else "not added yet"
        _fail(f"findings {state} for round {last['n']}")
    return last


def _note(fid_round: int, state: str | None, to: str, note: str) -> dict[str, Any]:
    return {"round": fid_round, "from": state, "to": to, "note": note}


# --- start-round --------------------------------------------------------------


def _commit_sha(ctx: _Ctx, rev: str) -> str:
    """Full SHA of ``rev`` — exit 1 on an unknown commit (never passed raw to git diff)."""
    proc = _git(ctx.root, "rev-parse", "--verify", "--quiet", "--end-of-options", f"{rev}^{{commit}}")
    if proc.returncode != 0:
        _fail("unknown commit", head=rev)
    return proc.stdout.strip()


def cmd_start_round(ctx: _Ctx, args: argparse.Namespace) -> dict[str, Any]:
    head = _commit_sha(ctx, args.head)
    discarded = False
    if _dirty(ctx):
        if _git(ctx.root, "cat-file", "-e", f"HEAD:{ctx.rel}").returncode == 0:
            restore = _git(ctx.root, "checkout", "HEAD", "--", ctx.rel)
        else:
            restore = _git(ctx.root, "rm", "-q", "--cached", "--ignore-unmatch", "--", ctx.rel)
            if restore.returncode == 0:
                ctx.path.unlink(missing_ok=True)
        if restore.returncode != 0 or _dirty(ctx):
            _fail("could not discard the unfinished round", stderr=restore.stderr.strip())
        discarded = True
    stale = _git(ctx.root, "ls-files", "--others", "--exclude-standard", "--", ctx.fix_glob).stdout.splitlines()
    data = load(ctx.path)
    rounds = data["rounds"]
    common = {"discarded": discarded, "stale_fix_plans": stale}
    if rounds:
        open_ids = [f["id"] for f in data["findings"] if f["state"] == "open"]
        if open_ids:
            _fail(
                f"round {rounds[-1]['n']} committed unfinished",
                open=open_ids,
                hint="decide each open ID with `review_ledger.py set`, run check-round, commit the ledger, "
                "then start-round again",
                **common,
            )
    if not rounds:
        rounds.append({"n": 1, "head": head, "added": False})
        save(ctx.path, data)
        return {"status": "new", **common, "round": 1, "mode": "full", "base": None, "reason": None}
    last = rounds[-1]
    changed = _git(ctx.root, "diff", "--quiet", last["head"], head, "--", ".", *ctx.excludes).returncode != 0
    if not changed and not args.force_full:
        return {"status": "no-change", **common, "round": last["n"], "mode": None, "base": None, "reason": None}
    n = last["n"] + 1
    mode, base, reason = "full", None, None
    if args.force_full:
        reason = "forced"
    elif n >= 3:
        anc = _git(ctx.root, "merge-base", "--is-ancestor", last["head"], head).returncode
        if anc == 1:
            reason = "base not an ancestor"
        elif anc != 0:
            reason = "base unknown"
        elif _git(ctx.root, "rev-list", "--merges", f"{last['head']}..{head}").stdout.strip():
            reason = "merge commit since base"
        else:
            mode, base = "delta", last["head"]
    rounds.append({"n": n, "head": head, "added": False})
    save(ctx.path, data)
    return {"status": "new", **common, "round": n, "mode": mode, "base": base, "reason": reason}


# --- add: parsing and validation ---------------------------------------------


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            _fail("duplicate key", key=key)
        out[key] = value
    return out


def _keys(obj: Any, allowed: frozenset[str], where: str) -> None:
    if not isinstance(obj, dict):
        _fail(f"{where}: expected an object")
    missing = sorted(allowed - obj.keys())
    if missing:
        _fail(f"{where}: missing key", key=missing[0])
    unknown = sorted(obj.keys() - allowed)
    if unknown:
        _fail(f"{where}: unknown key", key=unknown[0])


def _enum(value: Any, allowed: tuple[str, ...], where: str, key: str, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str) or value not in allowed:
        _fail(f"{where}: bad value for {key}", key=key, value=value)


def _text(value: Any, where: str, key: str, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str) or not value.strip():
        _fail(f"{where}: {key} must be a non-empty string", key=key)


def _validate(obj: Any, snapshot: dict[str, dict[str, Any]]) -> None:
    _keys(obj, _TOP_KEYS, "object")
    mode = obj["mode"]
    _enum(mode, MODES, "object", "mode")
    for key in ("checks", "findings"):
        if not isinstance(obj[key], list):
            _fail(f"object: {key} must be a list", key=key)
    accepted = {fid for fid, f in snapshot.items() if f["state"] == "accepted"}
    seen_refs: set[str] = set()
    for i, check in enumerate(obj["checks"]):
        where = f"checks[{i}]"
        _keys(check, _CHECK_KEYS, where)
        _text(check["ref"], where, "ref")
        if check["ref"] not in accepted:
            _fail(f"{where}: ref is not an accepted finding", key="ref", id=check["ref"])
        if check["ref"] in seen_refs:
            _fail(f"{where}: duplicate check ref", key="ref", id=check["ref"])
        seen_refs.add(check["ref"])
        _enum(check["status"], CHECK_STATUSES, where, "status")
        _text(check["evidence"], where, "evidence")
    if mode == "verified" and seen_refs != accepted:
        _fail("checks must cover every accepted finding", key="checks", missing=sorted(accepted - seen_refs))
    seen_keys: set[str] = set()
    for i, f in enumerate(obj["findings"]):
        where = f"findings[{i}]"
        _keys(f, _FINDING_KEYS, where)
        for key in ("key", "source", "where", "title"):
            _text(f[key], where, key)
        if f["key"] in seen_keys:
            _fail(f"{where}: duplicate finding key", key="key", value=f["key"])
        seen_keys.add(f["key"])
        _enum(f["severity"], SEVERITIES, where, "severity")
        _enum(f["verdict"], VERDICTS, where, "verdict", nullable=True)
        for key in ("evidence", "reproduces_when", "note"):
            _text(f[key], where, key, nullable=True)
        _enum(f["severity_action"], SEVERITY_ACTIONS, where, "severity_action")
        rel = f["relation"]
        _keys(rel, _RELATION_KEYS, f"{where}.relation")
        _enum(rel["kind"], RELATION_KINDS, f"{where}.relation", "kind")
        if rel["kind"] != "NEW":
            _text(rel["ref"], f"{where}.relation", "ref")
        verified = f["verdict"] is not None
        if verified != (f["evidence"] is not None):
            _fail(f"{where}: evidence must be non-null iff verdict is non-null", key="evidence")
        action, sev = f["severity_action"], f["severity"]
        if action == "promote" and (
            sev != "nice-to-have" or f["verdict"] in (None, "REFUTED") or f["reproduces_when"] is None
        ):
            _fail(
                f"{where}: promote needs a verified (not refuted) nice-to-have with a scenario", key="severity_action"
            )
        if action == "demote" and sev == "nice-to-have":
            _fail(f"{where}: demote on a nice-to-have", key="severity_action")
        if mode == "fallback" and verified:
            _fail(f"{where}: fallback findings are never verified", key="verdict")
        if mode == "verified" and sev != "nice-to-have":
            if not verified and f["source"] != DOC_MAINTENANCE:
                _fail(f"{where}: a must/should finding needs a verdict", key="verdict")
            if verified and f["source"] not in SCENARIO_EXEMPT and f["reproduces_when"] is None and action != "demote":
                _fail(f"{where}: verified finding without reproduces_when must be demoted", key="severity_action")


# --- add: reconciliation ------------------------------------------------------


def _effective(el: dict[str, Any]) -> str:
    return {"keep": el["severity"], "demote": "nice-to-have", "promote": "should-fix"}[el["severity_action"]]


def _id_num(fid: str) -> int:
    return int(fid[2:])


def _superseder(fid: str, snapshot: dict[str, dict[str, Any]]) -> str | None:
    """The live head of ``fid``'s supersede chain, or ``None`` (fix #01 F-002).

    Each hop is the newest FLIP-FLOP finding that refs the current one and
    was accepted (live or itself superseded since); the walk stops at the
    first live one. A cycle or a dead end yields ``None``.
    """
    seen = {fid}
    current = fid
    while True:
        cands = [
            f["id"]
            for f in snapshot.values()
            if f.get("relation")
            and f["relation"]["kind"] == "FLIP-FLOP"
            and f["relation"]["ref"] == current
            and f["state"] in (*_LIVE, "superseded")
        ]
        if not cands:
            return None
        current = max(cands, key=_id_num)
        if current in seen:
            return None
        if snapshot[current]["state"] in _LIVE:
            return current
        seen.add(current)


def _resolve(el: dict[str, Any], snapshot: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
    """Plan for one element: ``(create, effect, relation, note)``, ``(duplicate, ref)`` or ``(target, ref, kind)``."""
    kind, ref = el["relation"]["kind"], el["relation"]["ref"]
    if kind == "NEW":
        return ("create", "new", None, None)

    def down(why: str) -> tuple[Any, ...]:
        return ("create", "new-downgraded", None, f"relation-invalid: {kind} {ref}: {why}")

    target = snapshot.get(ref)
    if target is None:
        return down("unknown ref")
    state = target["state"]
    if kind == "DUPLICATE":
        if state == "open":
            return down("ref is open")
        if _SEV_RANK[_effective(el)] > _SEV_RANK[target["severity"]]:
            return down("severity escalated")
        return ("duplicate", ref)
    if state == "superseded":
        sup = _superseder(ref, snapshot)
        if sup is None:
            return down("superseded ref has no live superseder")
        return (
            "create",
            "flip-flop",
            {"kind": "FLIP-FLOP", "ref": sup},
            f"redirected: {kind} {ref} is superseded by {sup}",
        )
    if kind == "REOPENS":
        return ("target", ref, "REOPENS") if state in _LIVE else down(f"ref is {state}")
    if state in _LIVE:
        return ("create", "flip-flop", {"kind": "FLIP-FLOP", "ref": ref}, None)
    if state == "rejected" and target["rejected_by"] == "user":
        return ("target", ref, "FLIP-FLOP")
    return down(f"ref is {state} (rejected_by {target['rejected_by']})")


def _retarget(live: dict[str, Any], el: dict[str, Any]) -> str:
    previous = {k: live[k] for k in _RETARGET}
    live.update(
        {
            "reported_severity": el["severity"],
            "severity_action": el["severity_action"],
            "severity": _effective(el),
            "where": el["where"],
            "verdict": el["verdict"],
            "evidence": el["evidence"],
            "reproduces_when": el["reproduces_when"],
            "note": el["note"],
        }
    )
    return json.dumps(previous, ensure_ascii=False)


def _reconcile(
    data: dict[str, Any], obj: dict[str, Any], snapshot: dict[str, dict[str, Any]], n: int
) -> dict[str, Any]:
    findings = data["findings"]
    live = {f["id"]: f for f in findings}
    results: dict[str, Any] = {el["key"]: None for el in obj["findings"]}
    creates: list[tuple[dict[str, Any], str, Any, str | None]] = []
    targets: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for el in obj["findings"]:
        plan = _resolve(el, snapshot)
        if plan[0] == "create":
            creates.append((el, plan[1], plan[2], plan[3]))
        elif plan[0] == "duplicate":
            fx = live[plan[1]]
            fx["history"].append(_note(n, fx["state"], fx["state"], f"duplicate: {el['source']} ({el['key']})"))
            results[el["key"]] = {"id": plan[1], "effect": "duplicate"}
        else:
            targets.setdefault(plan[1], []).append((plan[2], el))
    for el, effect, relation, note in creates:
        nxt = max((_id_num(f["id"]) for f in findings), default=0) + 1
        fid = f"F-{nxt:03d}"
        history = [_note(n, None, "open", "added")]
        if note:
            history.append(_note(n, "open", "open", note))
        findings.append(
            {
                "id": fid,
                "round": n,
                "source": el["source"],
                "reported_severity": el["severity"],
                "severity_action": el["severity_action"],
                "severity": _effective(el),
                "where": el["where"],
                "title": el["title"],
                "note": el["note"],
                "verdict": el["verdict"],
                "evidence": el["evidence"],
                "reproduces_when": el["reproduces_when"],
                "relation": relation,
                "state": "open",
                "rejected_by": None,
                "decided_round": None,
                "history": history,
            }
        )
        results[el["key"]] = {"id": fid, "effect": effect}
    checks = {c["ref"]: c for c in obj["checks"]}
    check_results: dict[str, str] = {}
    rank = {"REOPENS": 0, "FLIP-FLOP": 1, "STILL-PRESENT": 2, "RESOLVED": 3}
    for fid in [f for f in snapshot if f in targets or f in checks]:
        fx = live[fid]
        effects: list[tuple[str, dict[str, Any] | None, dict[str, Any] | None]] = [
            (kind, el, None) for kind, el in targets.get(fid, [])
        ]
        if fid in checks:
            effects.append((checks[fid]["status"], None, checks[fid]))
        effects.sort(key=lambda e: rank[e[0]])
        (kind, el, chk), losers = effects[0], effects[1:]
        before = fx["state"]
        if el is not None:
            previous = _retarget(fx, el)
            fx["state"], fx["rejected_by"] = "open", None
            label = "reopened" if kind == "REOPENS" else "challenged"
            fx["history"].append(
                _note(n, before, "open", f"{label}: {el['source']} ({el['key']}); previous: {previous}")
            )
            results[el["key"]] = {"id": fid, "effect": label}
        else:
            assert chk is not None
            fx["state"] = "open" if kind == "STILL-PRESENT" else "resolved"
            fx["history"].append(_note(n, before, fx["state"], f"check {kind}: {chk['evidence']}"))
            check_results[fid] = "reopened" if kind == "STILL-PRESENT" else "resolved"
        for lkind, lel, _lchk in losers:
            who = f"{lel['source']} ({lel['key']})" if lel is not None else "check"
            fx["history"].append(_note(n, fx["state"], fx["state"], f"merged: {lkind} from {who}, {kind} applied"))
            if lel is not None:
                results[lel["key"]] = {"id": fid, "effect": "merged"}
            else:
                check_results[fid] = "merged"
    return {"findings": results, "checks": check_results}


def cmd_add(ctx: _Ctx, args: argparse.Namespace) -> dict[str, Any]:
    data = load(ctx.path)
    rnd = _require_started(ctx, data, added=False)
    try:
        obj = json.loads(sys.stdin.read(), object_pairs_hook=_pairs)
    except ValueError as exc:
        raise LedgerError(1, f"not JSON: {exc}") from exc
    snapshot = copy.deepcopy({f["id"]: f for f in data["findings"]})
    _validate(obj, snapshot)
    result = _reconcile(data, obj, snapshot, rnd["n"])
    rnd["added"] = True
    save(ctx.path, data)
    return result


# --- set / check-round / show -------------------------------------------------


def cmd_set(ctx: _Ctx, args: argparse.Namespace) -> dict[str, Any]:
    data = load(ctx.path)
    rnd = _require_started(ctx, data, added=True, recover=True)
    live = {f["id"]: f for f in data["findings"]}
    f = live.get(args.id)
    if f is None:
        _fail("unknown ID", id=args.id)
    if f["state"] != "open":
        _fail(f"{args.id} is {f['state']}; only open findings can be decided", id=args.id)
    if args.to == "rejected" and args.by is None:
        _fail("--by is required for --to rejected", id=args.id)
    if args.to == "accepted" and args.by is not None:
        _fail("--by is not allowed for --to accepted", id=args.id)
    n = rnd["n"]
    label = args.by or "user"
    text = args.note or ("fix" if args.to == "accepted" else "skip")
    f["state"], f["rejected_by"], f["decided_round"] = args.to, args.by, n
    f["history"].append(_note(n, "open", args.to, f"{label}: {text}"))
    superseded = None
    rel = f.get("relation")
    if args.to == "accepted" and rel and rel["kind"] == "FLIP-FLOP":
        fx = live.get(rel["ref"])
        # ``open`` too (fix #01 F-001): F-x reopened this round must not stay
        # decidable once its undo is accepted — order-independent.
        if fx is not None and fx["state"] in (*_LIVE, "open"):
            fx["history"].append(_note(n, fx["state"], "superseded", f"superseded by {args.id}"))
            fx["state"] = "superseded"
            superseded = fx["id"]
    save(ctx.path, data)
    return {"id": args.id, "state": args.to, "superseded": superseded}


def cmd_check_round(ctx: _Ctx, args: argparse.Namespace) -> dict[str, Any]:
    data = load(ctx.path)
    open_ids = [f["id"] for f in data["findings"] if f["state"] == "open"]
    if open_ids:
        _fail("open findings remain", open=open_ids)
    n = data["rounds"][-1]["n"] if data["rounds"] else 0
    unchecked = [f["id"] for f in data["findings"] if f["state"] == "accepted" and (f["decided_round"] or 0) < n]
    return {"accepted_unchecked": unchecked}


def _file_changed(root: Path, where: str, sha: str | None, head: str) -> tuple[bool | None, str | None]:
    m = _WHERE_RE.match(where)
    if m is None or sha is None:
        return None, None
    path = m.group("path")
    if all(_git(root, "cat-file", "-e", f"{rev}:{path}").returncode != 0 for rev in (sha, head)):
        return None, None
    rc = _git(root, "diff", "--quiet", sha, head, "--", path).returncode
    if rc == 0:
        return False, None
    if rc != 1:
        return None, None
    lines = _git(root, "diff", sha, head, "--", path).stdout.splitlines()
    if len(lines) > DIFF_CAP:
        lines = lines[: DIFF_CAP - 1] + [f"... [truncated: {len(lines) - DIFF_CAP + 1} more lines]"]
    return True, "\n".join(lines)


def _md(data: dict[str, Any]) -> str:
    rows = [
        "| ID | Round | Source | Sev | Where | Title | State | By | Relation |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for f in data["findings"]:
        rel = f.get("relation")
        cells = [
            f["id"],
            str(f["round"]),
            f["source"],
            f["severity"],
            f["where"],
            f["title"],
            f["state"],
            f.get("rejected_by") or "",
            f"{rel['kind']} {rel['ref']}" if rel else "",
        ]
        rows.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    return "\n".join(rows)


def cmd_show(ctx: _Ctx, args: argparse.Namespace) -> Any:
    data = load(ctx.path)
    if args.format == "md":
        return _md(data)
    if args.head:
        head = _commit_sha(ctx, args.head)
        heads = {r["n"]: r["head"] for r in data["rounds"]}
        for f in data["findings"]:
            if f.get("decided_round") is None:
                continue
            f["file_changed"], f["diff_since_decision"] = _file_changed(
                ctx.root, f["where"], heads.get(f["decided_round"]), head
            )
    return data


# --- CLI ----------------------------------------------------------------------


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise LedgerError(1, f"usage: {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="review_ledger.py", description="QS review-finding ledger (QS-453)")
    sub = parser.add_subparsers(dest="cmd", required=True, parser_class=_Parser)

    def common(name: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name)
        sp.add_argument("--issue", type=int, required=True)
        sp.add_argument("--root")
        return sp

    sr = common("start-round")
    sr.add_argument("--head", required=True)
    sr.add_argument("--force-full", action="store_true")
    common("add")
    st = common("set")
    st.add_argument("--id", required=True)
    st.add_argument("--to", required=True, choices=("accepted", "rejected"))
    st.add_argument("--by", choices=REJECTED_BY)
    st.add_argument("--note")
    common("check-round")
    sh = common("show")
    sh.add_argument("--format", choices=("json", "md"), default="json")
    sh.add_argument("--head")
    return parser


_COMMANDS = {
    "start-round": cmd_start_round,
    "add": cmd_add,
    "set": cmd_set,
    "check-round": cmd_check_round,
    "show": cmd_show,
}


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        ctx = _Ctx(resolve_root(args.root), args.issue)
        out = _COMMANDS[args.cmd](ctx, args)
    except LedgerError as exc:
        print(json.dumps(exc.payload, indent=2, ensure_ascii=False))
        return exc.code
    print(out if isinstance(out, str) else json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
