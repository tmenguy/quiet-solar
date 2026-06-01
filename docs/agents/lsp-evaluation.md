---
title: LSP evaluation
slug: lsp-evaluation
kind: principle
last_verified: 2026-06-01
---

# LSP evaluation — should agents pair with a Python language server?

## Decision (2026-06-01): adopt native pyright in the Claude harness

**Enabled for Claude.** The Claude Code agents now use the built-in
`LSP` tool backed by the official `pyright-lsp` plugin. This supersedes
the previous "defer + build a jedi-via-MCP server" recommendation
(see [the rebuttal](#why-native-pyright-now-beats-the-old-jedi--mcp-plan)).

- **Claude Code** — adopted. The `pyright-lsp` plugin is enabled at
  project scope in `.claude/settings.json`
  (`enabledPlugins["pyright-lsp@claude-plugins-official"] = true`), and
  the `LSP` tool is added to the closed `tools:` allowlist of the 8
  code-navigating agents.
- **OpenCode** — **not enabled.** OpenCode defaults to pyright
  (`"lsp": true` in `opencode.json`) but exposes LSP to the agent only
  as diagnostics — no navigation. Not worth wiring now.
- **Cursor** — **TBD / deferred.** Cursor (2.4+) has editor-native LSP
  that surfaces ambiently in-session; there is no separate agent tool to
  enable, and no decision is forced.

## Problem statement

When a plan, implement, or review agent needs symbol-level information
about the codebase — "find all callers of `Battery.charge`", "what's
the return type of `LoadConstraint.score`", "where is `SOLVER_STEP_S`
referenced?" — its baseline tools are `grep`, `glob`, and reading whole
files. The token cost is significant: a broad `grep` returns many lines
of context per match; reading even a moderate file like
`home_model/solver.py` (~1,700 LOC) burns tens of thousands of tokens.
A language server returns **structured** answers (definition location,
call sites, type) in hundreds of bytes — often two orders of magnitude
cheaper for the dynamic queries grep is worst at.

This document records what native LSP brings to the harness, why it now
beats the prior build-it-ourselves plan, and how the wiring works.

## Per-harness capability matrix

| Harness | LSP backend | Diagnostics (type errors, missing imports) | Navigation (defs, refs, hover, symbols, impls, call hierarchy) | Status here |
|---|---|---|---|---|
| **Claude Code** | pyright (`pyright-lsp` plugin) | ✅ surfaced in-turn, before the quality gate | ✅ via the `LSP` tool | **Adopted** |
| **OpenCode** | pyright (`opencode.json` `"lsp": true`) | ✅ diagnostics-only | ❌ not surfaced to the agent | Not enabled |
| **Cursor** | editor-native (2.4+) | ✅ ambient in editor | ✅ ambient in editor | TBD / deferred |

The two Claude wins are distinct: **post-edit diagnostics** (type errors
and missing imports surfaced in the same turn as the edit, catching
mistakes before the quality gate runs) and **code navigation**
(definitions, find-references, hover types, document/workspace symbols,
implementations, call hierarchies).

## The `tools:` allowlist gate

Every qs Claude agent declares a **closed** `tools:` allowlist (e.g.
`tools: Bash, Read, Edit, Write, Grep, Glob, Agent, TodoWrite, WebFetch`).
Per `code.claude.com/docs/en/sub-agents`, a `tools:` list is a strict
**allowlist**: an agent gets *only* the listed tools (it inherits all
tools *only* when the field is omitted). `LSP` is a gated tool, so
**enabling the plugin is necessary but not sufficient** — `LSP` must
appear on an agent's `tools:` line for that agent to use it. Because the
auto post-edit diagnostics are a behavior *of the same `LSP` tool*, they
also stay dormant until `LSP` is on the list.

### Which agents get `LSP`

Principle: grant `LSP` to agents whose toolset already includes
code-reading tools **and** whose role navigates the real codebase.
Deliberately **exclude** the context-starved reviewers (whose design
forbids reading the codebase / the repo) and the merge/release agents
(no code navigation).

| Agent | `LSP`? | Why |
|---|---|---|
| `qs-create-plan` | ✅ | Globs/reads code while planning |
| `qs-implement-task` | ✅ | Edits code — post-edit diagnostics are the headline win |
| `qs-implement-setup-task` | ✅ | Edits scripts/config — same diagnostics win |
| `qs-review-task` | ✅ | Navigates code during consolidation |
| `qs-plan-concrete-planner` | ✅ | Reads file tree + source snippets to verify diffs |
| `qs-review-edge-case-hunter` | ✅ | Walks branching paths in the PR source |
| `qs-review-acceptance-auditor` | ✅ | Builds a traceability matrix over source |
| `qs-setup-task` | ✅ | Globs/reads code to scope a new issue |
| `qs-plan-critic` | ❌ | **Blind** — reviews plan text only |
| `qs-plan-dev-proxy` | ❌ | **Blind** — never reads source |
| `qs-plan-scope-guardian` | ❌ | **Blind** — plan + issue only |
| `qs-review-blind-hunter` | ❌ | **Blind** — diff only |
| `qs-review-coderabbit` | ❌ | Pass-through wrapper; no code navigation |
| `qs-finish-task` | ❌ | Merge / cleanup only |
| `qs-release` | ❌ | Version bump / tag only |

A pin test (`tests/qs/agents/test_lsp_tool_enabled.py`) asserts **both**
the include set and the exclude set, so granting `LSP` to a blind
reviewer can't slip in silently.

## Prerequisite: install pyright (manual, per-machine)

The `pyright-lsp` plugin shells out to the `pyright-langserver` binary.
Install it machine-level with npm:

```bash
npm install -g pyright
```

This ships both `pyright` and `pyright-langserver` (confirmed in
`code.claude.com/docs/en/plugins-reference`). **Not** via pip and
**not** in the venv — the project's type-checker is **mypy**; pyright
here is purely agent ergonomics and must not leak into
`requirements*.txt`. The Claude Code process spawns the language server
from its own login-shell PATH, so an npm-global binary is reliably
visible; a venv `bin/` (only on PATH when activated) is not.

### Graceful degradation

The committed wiring degrades gracefully when `pyright-langserver` is
absent: the `LSP` tool simply stays inactive and agents fall back to
grep/glob. This is a documented Claude-side runtime expectation, not
something this repo's tests assert.

### How to disable

- **One agent:** remove `LSP` from that agent's `tools:` line.
- **Whole harness:** remove the
  `"pyright-lsp@claude-plugins-official"` entry from `enabledPlugins`
  in `.claude/settings.json` (or set it to `false`).
- **Per machine:** uninstall the binary (`npm uninstall -g pyright`) —
  the tool then degrades silently as above.

## Why native pyright now beats the old jedi + MCP plan

The previous version of this doc recommended **defer**, and — if ever
adopted — building a **jedi-language-server** wrapped in a custom **MCP
server** for multi-harness reach. Two objections drove that plan:
pyright's **Node-dependency** and the desire for a **single
multi-harness** integration. Both are now outweighed:

- **Zero-build native integration.** Claude Code ships the `LSP` tool
  and an official `pyright-lsp` plugin. Adopting it is a two-line config
  change (enable the plugin, add `LSP` to allowlists) versus ~1 day of
  MCP-server scaffolding plus ongoing maintenance. The build cost the
  old plan tried to justify no longer exists.
- **The Node dependency is a one-line npm install**, isolated to the
  developer's machine and explicitly kept out of the venv and
  `requirements*.txt`. It does not touch the product runtime or CI.
- **The "multi-harness MCP" argument is moot for the win we want.**
  Cursor already has editor-native LSP; OpenCode already bundles
  pyright. We are not blocked on a portable MCP layer to get value —
  each harness provides its own. A custom MCP server would *duplicate*
  capabilities the harnesses already ship.
- **pyright's type inference is stronger than jedi's**, which matters
  most for the diagnostics win (catching real type errors before the
  gate) — the capability jedi was weakest at.

The structural insight from the old doc still holds: the `docs/agents/`
hierarchy answers *static* "what is a LoadCommand?" questions more
cheaply than any LSP because the answer is pre-curated. The LSP pulls
ahead for **dynamic** queries (call sites, type inference on arbitrary
expressions) and for **post-edit diagnostics**, which docs can't
provide.

## The issue's three questions, answered

The originating issue (#248) asked three things:

1. **What does it bring?** Two things grep cannot: (a) **post-edit
   diagnostics** — type errors and missing imports surfaced in the same
   turn as an edit, before the quality gate; and (b) **structured
   navigation** — exact definition/reference/hover/symbol answers
   instead of regex matches the agent must still open files to confirm.
2. **Is it better than grepping?** For *dynamic symbol* queries, yes —
   it returns precise, semantically-resolved locations and types where
   grep returns textual matches (including false positives in comments,
   strings, and unrelated names). For *conceptual* queries, the curated
   `docs/agents/` hierarchy is still cheaper. The two are complementary;
   the LSP does not replace grep for plain text search.
3. **Does it reduce or multiply tokens?** Qualitatively, it **reduces**
   tokens for the dynamic queries it targets — a `references` result is
   a short file:line list rather than many grepped context windows, and
   a `hover` is one signature rather than a whole-file read. There is a
   small fixed per-session cost (the language server handshake), but it
   is dominated by the savings on even a handful of navigation queries.

### Token benchmark — consciously deferred (not dropped)

A *precise* token benchmark (controlled before/after on real sessions)
is **deliberately deferred**, not abandoned. Rationale: the adoption
cost is now near-zero and reversible (see [How to disable](#how-to-disable)),
so a measurement gate would cost more than the change it guards. The
prior doc's order-of-magnitude estimates (≈4–360× per-query savings,
20–40% of an implement session) remain the working hypothesis. Revisit
with a real benchmark if/when token cost becomes the dominant constraint
on agent throughput, or if the diagnostics win proves marginal in
practice.

## See also

- [index.md](index.md) — the doc hierarchy LSP augments, not replaces.
- [../workflow/harness.md](../workflow/harness.md) — multi-harness
  abstraction (Claude / Cursor / OpenCode / Codex) and the "Code
  intelligence (LSP)" subsection.
