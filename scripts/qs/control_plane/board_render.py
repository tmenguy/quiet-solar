"""The board's HTML (#428 D6, D7, D8): ``render(model) -> str``, pure.

Everything it needs is in the model dict ``board.model`` builds; it imports nothing from ``board``.

**Security.** Every value is escaped (``html.escape(..., quote=True)``); LLM-written text goes into
element content only. There is no inline event handler: the one fixed ``SCRIPT`` attaches its handlers
with ``addEventListener`` and reads every dynamic value from ``data-`` attributes on ``<body>``, so a
CSP pins it by hash. The page loads nothing external. A URL becomes a link only when it starts with
``https://github.com/``. A copy block holds ids of a fixed shape and fixed words only — never a title,
a question or a name — and a terminal command quotes every variable part (``shlex.quote``).
``scrub`` runs over the final page as defence in depth.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import re
import shlex
from typing import Any

SCRIPT = """
(function () {
  "use strict";
  var body = document.body;
  var generated = Number(body.dataset.generatedAt) * 1000;
  var staleMs = Number(body.dataset.staleS) * 1000;
  var refreshMs = Number(body.dataset.refreshS) * 1000;
  var forceMs = Number(body.dataset.forceReloadS) * 1000;
  var banner = document.getElementById("stale-banner");
  var KEY = "qs-board-open-details";
  var lastCopy = 0;
  var skippedSince = null;
  try {
    JSON.parse(sessionStorage.getItem(KEY) || "[]").forEach(function (id) {
      var d = document.getElementById(id);
      if (d && d.tagName === "DETAILS") { d.open = true; }
    });
  } catch (e) {}
  function saveOpen() {
    try {
      var ids = [];
      document.querySelectorAll("details[id]").forEach(function (d) { if (d.open) { ids.push(d.id); } });
      sessionStorage.setItem(KEY, JSON.stringify(ids));
    } catch (e) {}
  }
  document.querySelectorAll("details[id]").forEach(function (d) { d.addEventListener("toggle", saveOpen); });
  document.querySelectorAll("button[data-copy-btn]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      lastCopy = Date.now();
      var pre = btn.previousElementSibling;
      function select() {
        var range = document.createRange();
        range.selectNodeContents(pre);
        var sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(pre.textContent).then(function () { btn.textContent = "Copied"; }, select);
      } else {
        select();
      }
    });
  });
  function tick() {
    var now = Date.now();
    if (now - generated > staleMs) { banner.hidden = false; }
    var sel = window.getSelection();
    var busy = (sel !== null && sel.toString().length > 0) || now - lastCopy < 5000;
    if (busy) {
      if (skippedSince === null) { skippedSince = now; }
      if (now - skippedSince < forceMs) { setTimeout(tick, refreshMs); return; }
    }
    saveOpen();
    location.reload();
  }
  setTimeout(tick, refreshMs);
})();
"""

SCRIPT_HASH = base64.b64encode(hashlib.sha256(SCRIPT.encode("utf-8")).digest()).decode("ascii")
CSP = f"default-src 'none'; style-src 'unsafe-inline'; script-src 'sha256-{SCRIPT_HASH}'"

STYLE = """
body { font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0 1.5em 3em; color: #1b1f24; }
h1 { font-size: 1.4em; } h2 { font-size: 1.2em; border-bottom: 1px solid #d0d7de; padding-bottom: .2em; }
h3 { font-size: 1.05em; }
table { border-collapse: collapse; margin: .4em 0 1em; }
th, td { border: 1px solid #d0d7de; padding: .25em .5em; text-align: left; vertical-align: top; }
th { background: #f6f8fa; }
.muted { color: #656d76; }
.copy { display: flex; gap: .5em; align-items: flex-start; margin: .3em 0; }
.copy pre { margin: 0; padding: .3em .5em; background: #f6f8fa; border: 1px solid #d0d7de; white-space: pre-wrap; }
.waiting li { margin-bottom: .8em; }
.where { font-weight: 600; }
#stale-banner { background: #cf222e; color: #fff; padding: .6em 1em; margin: 0 -1.5em; }
#stale-banner pre { color: #1b1f24; }
[hidden] { display: none !important; }
svg .box rect { stroke: #57606a; stroke-width: 1; }
svg .box.unresolved rect { stroke: #cf222e; stroke-width: 2; stroke-dasharray: 4 2; }
svg .box text { font-size: 12px; fill: #1b1f24; }
svg .edge { fill: none; stroke: #57606a; stroke-width: 1.2; }
"""

STATE_COLOURS = {
    "proposed": "#eaeef2",
    "ready": "#ddf4ff",
    "planning": "#fff8c5",
    "contracted": "#fff1b8",
    "building": "#ffe7a3",
    "ready_to_merge": "#d2f5d0",
    "merged": "#aceebb",
    "validated": "#6fdd8b",
    "blocked": "#ffd8d3",
    "dropped": "#d0d7de",
}
UNKNOWN_COLOUR = "#ffffff"
NONE = '<p class="muted">none</p>'

BOX_W, BOX_H, GAP_X, GAP_Y, PAD = 190, 34, 70, 12, 10

# --------------------------------------------------------------------------- copy blocks (D6)

TASK_ID = re.compile(r"T\d+")
QUESTION_ID = re.compile(r"Q-\d+")
RUN_ID = re.compile(r"R\d+")
SHORT_ID = re.compile(r"[0-9a-zA-Z-]{1,64}")
HALTS = frozenset({"failed", "stuck"})


def _shaped(pattern: re.Pattern[str], value: Any) -> str | None:
    return value if isinstance(value, str) and pattern.fullmatch(value) else None


def copy_text(kind: str, value: Any = None, *, main: str = "") -> str | None:
    """The exact text of a copy block; ``None`` when its id fails its shape (no block then)."""
    if kind == "answer":
        qid = _shaped(QUESTION_ID, value)
        return None if qid is None else f"Answer to {qid}: "
    if kind == "criteria":
        tid = _shaped(TASK_ID, value)
        return None if tid is None else f"Show me the acceptance criteria of {tid} so I can validate them."
    if kind == "where_is":
        tid = _shaped(TASK_ID, value)
        return None if tid is None else f"Where is {tid}? Summarise its state, its open findings and what it waits for."
    if kind == "resume":
        rid = _shaped(RUN_ID, value)
        return None if rid is None else f"Resume the run {rid}: claim it and drain its queue."
    if kind == "halt":
        if value not in HALTS:
            return None
        return (
            f"The merge gate is {value}. Show me the self-check record, then clear the halt with"
            " cp.py halt clear only if I agree."
        )
    if kind == "backup":
        return "The Control Plane backup is failing. Show me the last backup error from the snapshot."
    if kind == "attach":
        sid = _shaped(SHORT_ID, value)
        return None if sid is None else f"claude attach {shlex.quote(sid)}"
    if kind == "board":
        return f"{shlex.quote(f'{main}/venv/bin/python')} {shlex.quote(f'{main}/scripts/qs/cp.py')} board"
    return None


def _copy(text: str | None, label: str = "") -> str:
    if text is None:
        return ""
    lead = f'<span class="muted">{esc(label)}</span>' if label else ""
    return (
        f'<div class="copy">{lead}<pre data-copy>{esc(text)}</pre>'
        '<button type="button" data-copy-btn>Copy</button></div>'
    )


# --------------------------------------------------------------------------- helpers


def esc(value: Any) -> str:
    """Escaped element content or attribute value; ``None`` is a dash."""
    return "—" if value is None else html.escape(str(value), quote=True)


def _raw(value: Any) -> str:
    """A value shown raw (a JSON object, a list…), escaped."""
    if value is None or isinstance(value, str):
        return esc(value)
    return esc(json.dumps(value, sort_keys=True, default=str))


def link(url: Any, label: Any) -> str:
    """A link only for ``https://github.com/``; anything else is text."""
    if isinstance(url, str) and url.startswith("https://github.com/"):
        return f'<a href="{esc(url)}" rel="noreferrer">{esc(label)}</a>'
    return esc(url)


def _table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def _where(where: dict[str, Any]) -> str:
    to = where.get("to")
    if to == "chat":
        return f"in the chat session <b>{esc(where.get('chat'))}</b>"
    if to == "any_chat":
        chats = ", ".join(f"<b>{esc(c)}</b>" for c in where.get("chats", []))
        return f"in any orchestrator chat ({chats})"
    return "in a new session on the main checkout"


# --------------------------------------------------------------------------- sections


def _waiting(m: dict[str, Any]) -> str:
    items = m["waiting"]
    if not items:
        return '<section id="waiting"><h2>Waiting for you</h2><p>Nothing is waiting for you.</p></section>'
    lis = []
    for item in items:
        detail = "" if item.get("detail") is None else f'<div class="detail">{esc(item["detail"])}</div>'
        lis.append(
            f'<li><div><span class="where">{_where(item["where"])}</span> — {esc(item["text"])}</div>{detail}'
            f"{_copy(copy_text(*item['copy'], main=m['main']))}</li>"
        )
    return f'<section id="waiting"><h2>Waiting for you ({len(items)})</h2><ol class="waiting">{"".join(lis)}</ol></section>'


def _graph(g: dict[str, Any]) -> str:
    layers = g["layers"]
    if not layers:
        return '<p class="muted">no task yet</p>'
    pos: dict[str, tuple[int, int]] = {}
    for col, layer in enumerate(layers):
        for row, node in enumerate(layer):
            pos[node["id"]] = (PAD + col * (BOX_W + GAP_X), PAD + row * (BOX_H + GAP_Y))
    width = PAD * 2 + len(layers) * BOX_W + (len(layers) - 1) * GAP_X
    height = PAD * 2 + max(len(layer) for layer in layers) * (BOX_H + GAP_Y) - GAP_Y
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"'
        ' role="img" aria-label="task graph">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7"'
        ' orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#57606a"/></marker></defs>',
    ]
    for dep, dependent in g["edges"]:
        (x1, y1), (x2, y2) = pos[dep], pos[dependent]
        sx, sy, ex, ey = x1 + BOX_W, y1 + BOX_H // 2, x2, y2 + BOX_H // 2
        parts.append(
            f'<path class="edge" d="M {sx} {sy} C {sx + 35} {sy}, {ex - 35} {ey}, {ex} {ey}" marker-end="url(#arrow)"/>'
        )
    for layer in layers:
        for node in layer:
            x, y = pos[node["id"]]
            fill = STATE_COLOURS.get(node["state"], UNKNOWN_COLOUR)
            cls = "box unresolved" if node["unresolved"] else "box"
            parts.append(
                f'<a href="#task-{esc(node["id"])}" class="{cls}"><title>{esc(node["title"])}</title>'
                f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="4" fill="{fill}"/>'
                f'<text x="{x + 6}" y="{y + 14}">{esc(node["id"])} · {esc(node["state"])}</text>'
                f'<text x="{x + 6}" y="{y + 28}">{esc(node["short_title"])}</text></a>'
            )
    parts.append("</svg>")
    note = ""
    if any(n["unresolved"] for layer in layers for n in layer):
        note += '<p class="muted">The last column (dashed border) holds unresolved dependencies.</p>'
    if g["outside"]:
        note += f'<p class="muted">{esc(g["outside"])} dependencies outside this run</p>'
    return "".join(parts) + note


def _node_cell(node: dict[str, Any] | None) -> str:
    if node is None:
        return "—"
    state = "taken over (you)" if node["taken_over"] else node["state"]
    return f"{esc(state)}<br><span class='muted'>{esc(node['name'])} · {esc(node['short_id'])}</span>"


def _pr_cell(t: dict[str, Any]) -> str:
    if t["pr_url"] is None and t["pr_number"] is None:
        return "—"
    label = f"#{t['pr_number']}" if t["pr_number"] is not None else t["pr_url"]
    pr = link(t["pr_url"], label) if t["pr_url"] is not None else esc(label)
    ci = "" if t["ci_state"] is None else f"<br>CI {esc(t['ci_state'])} @ {esc(t['ci_sha7'])}"
    return pr + ci


def _task_details(t: dict[str, Any], main: str) -> str:
    crit = "".join(
        f"<li>{esc(c['state'])}{'' if c['validated'] else ' (not validated)'} — {esc(c['text'])}</li>"
        for c in t["criteria"]
    )
    qs = "".join(
        f"<li>{esc(q['id'])} · {esc(q['state'])}{' · blocking' if q['blocking'] else ''} — {esc(q['text'])}</li>"
        for q in t["questions"]
    )
    copies = "".join(
        _copy(copy_text(*c, main=main), "in a terminal:" if c[0] == "attach" else "in the run's chat:")
        for c in t["copies"]
    )
    return (
        f'<details id="task-{esc(t["id"])}"><summary>{esc(t["id"])} — {esc(t["title"])}</summary>'
        f"<h4>Criteria</h4>{f'<ul>{crit}</ul>' if crit else NONE}"
        f"<h4>Open questions</h4>{f'<ul>{qs}</ul>' if qs else NONE}"
        f"{copies}</details>"
    )


def _alerts_table(rows: list[dict[str, Any]], *, runs: bool = False) -> str:
    if not rows:
        return '<p class="muted">no alert</p>'
    head = ["kind", "subject", "severity", "state", "first seen", "last seen"] + (["runs"] if runs else [])
    body = [
        [
            esc(a["kind"]),
            esc(a["subject"]),
            esc(a["severity"]),
            "cleared" if a["cleared"] else "<b>open</b>",
            esc(a["first_seen"]),
            esc(a["last_seen"]),
        ]
        + ([esc(", ".join(str(r) for r in a["runs"]))] if runs else [])
        for a in rows
    ]
    return _table(head, body)


def _reviews_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="muted">no review</p>'
    body = []
    for r in rows:
        outcomes = "<br>".join(
            f"{esc(t['task_id'])}: {esc(t['outcome'])} — {esc(t['reason'])} ({esc(t['recorded_at'])})"
            for t in r["tasks"]
        )
        body.append([esc(r["id"]), esc(r["kind"]), esc(r["state"]), outcomes, esc(r["summary"])])
    return _table(["review", "kind", "state", "per-task outcome", "summary"], body)


def _run_section(run: dict[str, Any], main: str) -> str:
    p = run["progress"]
    orch = "" if run["orchestrator"] is None else f" — <b>{esc(run['orchestrator'])}</b>"
    by_state = ", ".join(f"{esc(s)} {esc(n)}" for s, n in p["by_state"].items()) or "no task"
    rows = []
    for t in run["tasks"]:
        state = esc(t["state"]) + ("" if t["blocked_from"] is None else f" (from {esc(t['blocked_from'])})")
        kind = esc(t["kind"]) + ("" if t["lane"] is None else f" / {esc(t['lane'])}")
        if t["item_of"] is not None:
            kind += f"<br><span class='muted'>item of {esc(t['item_of'])}</span>"
        crit = f"{esc(t['criteria_met'])}/{esc(t['criteria_total'])}"
        if t["criteria_total"]:
            crit += " validated" if t["criteria_validated"] else " <b>not validated</b>"
        rows.append(
            [
                f'<a href="#task-{esc(t["id"])}">{esc(t["id"])}</a>',
                "—" if t["issue"] is None else f"#{esc(t['issue'])}",
                esc(t["title"]),
                kind,
                state,
                _node_cell(t["node"]),
                _pr_cell(t),
                crit,
                esc(t["open_findings"]),
            ]
        )
    table = (
        _table(["task", "issue", "title", "kind / lane", "state", "node", "PR / CI", "criteria", "open findings"], rows)
        if rows
        else ""
    )
    details = "".join(_task_details(t, main) for t in run["tasks"])
    return (
        f'<section class="run" id="run-{esc(run["id"])}"><h2>{esc(run["id"])} · {esc(run["name"])} — {esc(run["title"])}</h2>'
        f"<p>orchestrator: chat session <b>{esc(run['chat'])}</b> · {esc(run['short_id'])}{orch}</p>"
        f"<p>progress: done {esc(p['done'])} · remaining {esc(p['remaining'])} · dropped {esc(p['dropped'])}"
        f' <span class="muted">({by_state})</span></p>'
        f"<h3>Graph</h3>{_graph(run['graph'])}<h3>Tasks</h3>{table}{details}"
        f"<h3>Alerts</h3>{_alerts_table(run['alerts'])}<h3>Reviews</h3>{_reviews_table(run['reviews'])}</section>"
    )


def _cross_run(c: dict[str, Any]) -> str:
    return (
        '<section id="cross-run"><h2>Cross-run</h2>'
        f"<h3>Alerts</h3>{_alerts_table(c['alerts'], runs=True)}<h3>Reviews</h3>{_reviews_table(c['reviews'])}</section>"
    )


def _system(s: dict[str, Any]) -> str:
    age = s["heartbeat_age_s"]
    daemon = "running" if s["daemon_running"] else "<b>not running</b>"
    hb = "—" if age is None else f"{age:.0f} s ago"
    gate = "".join(f"<tr><th>{esc(k)}</th><td><code>{_raw(v)}</code></td></tr>" for k, v in s["merge_gate"].items())
    hooks = [
        [esc(h.get("id")), esc(h.get("hook")), esc(h.get("session_id")), esc(h.get("at")), _raw(h.get("detail"))]
        for h in s["hook_alerts"]
    ]
    return (
        '<section id="system"><h2>System</h2>'
        f"<p>daemon: {daemon} · pid {esc(s['pid'])} · heartbeat {esc(hb)} · DB schema v{esc(s['db_schema'])}"
        f" (code v{esc(s['code_schema'])})</p>"
        f'<h3>Merge gate</h3><p class="muted">the raw records; the authority is <code>tool merge</code></p>'
        f"<table>{gate}</table>"
        f"<h3>Backup</h3><p>last backup {esc(s['backup']['last_at'])} · last error <code>{_raw(s['backup']['last_error'])}</code></p>"
        f"<h3>Hook alerts</h3>{_table(['id', 'hook', 'session', 'at', 'detail'], hooks) if hooks else NONE}"
        "</section>"
    )


def _closed(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    body = [
        [esc(r["name"]), esc(r["title"]), esc(r["closed_at"]), esc(r["done"]), esc(r["dropped"]), esc(r["remaining"])]
        for r in rows
    ]
    table = _table(["run", "title", "closed at", "done", "dropped", "remaining"], body)
    return f'<details id="closed-runs"><summary>Closed runs ({len(rows)})</summary>{table}</details>'


def render(m: dict[str, Any]) -> str:
    """The whole page for the model ``m``."""
    banner = (
        f'<div id="stale-banner" hidden><b>last known state</b> as of {esc(m["generated_hms"])} UTC —'
        " the daemon is stopped or the board write is failing. Refresh it from a terminal:"
        f"{_copy(copy_text('board', main=m['main']))}</div>"
    )
    header = (
        f"<header><h1>Quiet Solar board</h1><p>generated at {esc(m['generated_at'])}"
        f" · <b>{len(m['waiting'])}</b> waiting for you</p></header>"
    )
    if m["empty"]:
        main = f'<p id="empty">{esc(m["empty_line"])}</p>'
    else:
        main = (
            _waiting(m)
            + "".join(_run_section(r, m["main"]) for r in m["runs"])
            + _cross_run(m["cross_run"])
            + _system(m["system"])
            + _closed(m["closed_runs"])
        )
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        f'<meta http-equiv="Content-Security-Policy" content="{esc(CSP)}">'
        '<meta name="referrer" content="no-referrer">'
        f"<title>Quiet Solar board</title><style>{STYLE}</style></head>"
        f'<body data-generated-at="{esc(m["generated_at_s"])}" data-stale-s="{esc(m["stale_s"])}"'
        f' data-refresh-s="{esc(m["refresh_s"])}" data-force-reload-s="{esc(m["force_reload_s"])}">'
        f"{banner}{header}{main}<script>{SCRIPT}</script></body></html>\n"
    )


# --------------------------------------------------------------------------- scrub (D7)

_TOKEN = re.compile(r"(?:run|node):[A-Z]\d+\.\d+\.[0-9a-f]{32}|review:V[1-9]\d*\.[01]\.[0-9a-f]{32}")
_HEX32 = re.compile(r"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])")


def scrub(text: str) -> str:
    """Redact every token shape and every standalone 32-hex run (a nonce) as ``[redacted]``."""
    return _HEX32.sub("[redacted]", _TOKEN.sub("[redacted]", text))
