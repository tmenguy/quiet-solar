---
title: LSP evaluation
slug: lsp-evaluation
kind: principle
last_verified: 2026-05-19
---

# LSP evaluation — should agents pair with a Python language server?

## Problem statement

When a plan, implement, or review agent needs symbol-level information
about the codebase — "find all callers of `Battery.charge`", "what's
the return type of `LoadConstraint.score`", "where is `SOLVER_STEP_S`
referenced?" — its only tools today are `grep`, `glob`, and reading
whole files. The token cost is significant: a `grep` for `SOLVER_STEP_S`
returns ~20 lines of context per match; reading even a moderate file
like `home_model/solver.py` (1,683 LOC) burns roughly 35k tokens; a
broad grep across `custom_components/quiet_solar/` consumes O(50k)
tokens for a single lookup. A Python LSP server, by contrast, returns
**structured** answers (definition location, call sites, type) in
hundreds of bytes — two orders of magnitude cheaper.

This document evaluates whether the cost of installing, running, and
wiring an LSP into the agent harness is worth that token savings —
and, if so, which LSP and which integration mechanism.

## Three Python LSP options

| LSP | Install footprint | Query latency | Output verbosity | Strengths | Weaknesses |
|---|---|---|---|---|---|
| **pyright** (Microsoft) | npm package, ~50MB | <100ms typical | Verbose, structured JSON | Best type inference of the three; well-maintained; understands modern Python (3.13+) including `from __future__ import annotations`. | Node.js runtime dependency; not Python-native. |
| **jedi-language-server** | Single `pip install` (~30MB with deps) | <50ms for symbol lookup | Compact LSP-spec responses | Pure-Python; understands quiet-solar's `numpy`/`scipy` stack via Jedi's type-stub bundling; no Node dependency. | Weaker type inference than pyright; struggles with complex generics. |
| **pylsp** (Palantir/python-lsp) | `pip install python-lsp-server` + plugins | ~75ms typical | Variable per plugin | Plugin ecosystem (rope, pylint integration, etc.); the most extensible option. | Configuration burden; plugins have overlapping/conflicting behaviour. |

## Three consumption mechanisms for an agent

| Mechanism | Setup cost | Per-call cost | Privacy | Notes |
|---|---|---|---|---|
| **MCP server wrapper** (Model Context Protocol) | One-time MCP server scaffold (~1 day eng); restart the harness to pick it up. | Single JSON-RPC round trip. | LSP stays local — no data leaves the machine. | Best long-term: works across Claude / Cursor / OpenCode / Codex harnesses, matching the multi-harness story (QS-184). |
| **Direct subprocess via Bash tool** | Zero — `pyright --outputjson <symbol>` is callable from the existing Bash tool. | Per-call subprocess spawn (~200ms overhead). | Local. | Cheapest to prototype; awkward for repeated queries in one session. |
| **Custom Claude Code / Cursor tool** | Per-harness tool implementation. | Native tool call. | Local. | Most ergonomic for the user (autocomplete, dedicated tool surface), but multiplies maintenance — one tool per harness. |

## Quantitative comparison on three sample agent queries

The numbers below are order-of-magnitude estimates from running the
relevant `grep`/`glob` commands against the current repo (May 2026)
and from the LSP-spec responses for the same queries. They are meant
to anchor the recommendation, not to be precise benchmarks.

### Query 1 — "find all callers of `Battery.charge`"

| Approach | Tokens | Wall-clock |
|---|---|---|
| `grep -rn "\.charge(" custom_components/` then filter, then re-read context | ~3,500 (multiple file excerpts) | ~10s |
| LSP `textDocument/references` | ~250 (file:line list) | <1s |

**Savings: ~14× tokens, ~10× wall-clock.**

### Query 2 — "list the type of `LoadConstraint.score`"

| Approach | Tokens | Wall-clock |
|---|---|---|
| Read `home_model/constraints.py` (2,683 LOC, ~55k tokens) and trace the attribute | ~55,000 | ~30s |
| LSP `textDocument/hover` on `LoadConstraint.score` | ~150 (one signature + docstring) | <1s |

**Savings: ~360× tokens.**

### Query 3 — "find references to `SOLVER_STEP_S`"

| Approach | Tokens | Wall-clock |
|---|---|---|
| `grep -rn SOLVER_STEP_S custom_components/ tests/` | ~1,800 (~30 matches with context) | ~5s |
| LSP `workspace/symbol` + `textDocument/references` | ~400 | <1s |

**Savings: ~4× tokens.**

### Cumulative across a typical implement session

A typical implement-task session for a charger change runs 8–12 such
queries. Estimated savings:

- **Token cost reduction**: 20–40% of the implement-task session
  (queries are not the dominant cost — code reading and writing
  dominate — but they are still meaningful).
- **Latency reduction**: cumulative ~30–60s per session.

## Recommendation

**Defer** — until at least one of the following unlocks:

1. **Token cost becomes the dominant constraint on agent throughput.**
   Today, code generation and quality-gate iteration dominate cost.
   Until lookups are >30% of session tokens, the LSP wiring is
   premature optimisation.
2. **A multi-harness MCP standard ships** that doesn't require us
   to maintain a per-harness tool. The MCP path is the only one
   that scales across Claude / Cursor / OpenCode / Codex without
   parallel maintenance.
3. **A reviewer agent shows it would catch defects** that the
   current adversarial-review fan-out misses because the reviewers
   can't trace symbol references at scale. Hypothetical today; no
   evidence yet.

In the meantime, the `docs/agents/` hierarchy (created by QS-185)
addresses the **same problem from a different angle**: agents pull
small, scoped, addressable documentation files instead of grep-ing
the whole codebase. This is structurally cheaper than even an LSP
for the common queries ("what's a LoadCommand?") because the answer
is pre-curated. The LSP only pulls ahead for **dynamic** queries
(call sites, type inference on arbitrary expressions).

If the LSP becomes worth adopting, the path is:

1. Install **jedi-language-server** locally (pure-Python, no Node).
2. Wrap it in an **MCP server** that exposes `findReferences`,
   `hover`, and `workspaceSymbols` to all harnesses uniformly.
3. Add an opt-in note to `docs/workflow/overview.md` so agents
   know the tool exists.

## See also

- [index.md](index.md) — the doc hierarchy LSP would augment, not
  replace.
- [../workflow/harness.md](../workflow/harness.md) — multi-harness
  abstraction (Claude / Cursor / OpenCode / Codex) any LSP wiring
  must respect.
