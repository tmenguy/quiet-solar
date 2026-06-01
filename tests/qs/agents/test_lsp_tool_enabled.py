"""Pin the native-pyright ``LSP`` wiring for the Claude harness (QS-248).

Four assertions, encoded as pin tests so the deliberate include/exclude
split and the plugin-enablement schema cannot silently regress:

- ``test_lsp_enabled_on_code_navigating_agents`` — each of the 8
  code-navigating Claude agents has ``LSP`` on its closed ``tools:``
  allowlist. Because a ``tools:`` list is a strict allowlist (per
  ``code.claude.com/docs/en/sub-agents``), the ``LSP`` tool — and its
  auto post-edit diagnostics, which are a behavior of the same tool —
  stays inactive for any agent that omits it.
- ``test_lsp_excluded_on_blind_and_merge_agents`` — the 7
  context-starved reviewers / merge / release agents do **not** list
  ``LSP``. The blind reviewers' design forbids reading the codebase, so
  granting them code navigation would be a direct hard-rule violation.
- ``test_pyright_plugin_enabled_in_settings`` — ``.claude/settings.json``
  enables the official plugin via the object form
  ``enabledPlugins["pyright-lsp@claude-plugins-official"] is True``
  (schema verified against ``json.schemastore.org/claude-code-settings.json``).
- ``test_include_exclude_partition_is_complete`` — the include and
  exclude lists are disjoint and jointly cover every ``.claude/agents/*.md``
  file on disk, so a newly-added agent can't escape both parametrized
  tests (review-fix #01).

Plus one helper-robustness check, ``test_tools_helper_edge_cases``,
covering the omitted-``tools:`` (inherits-all), trailing-comma, and
exact-token-membership branches (review-fix #01).

Follows the ``test_opencode_agents.py`` pattern: ``REPO_ROOT`` via
``parents[3]`` and a local ``_parse_frontmatter``. ``yaml`` (PyYAML) is
transitively available via ``homeassistant``; no extra test requirement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CLAUDE_AGENTS_DIR = REPO_ROOT / ".claude" / "agents"
CLAUDE_SETTINGS = REPO_ROOT / ".claude" / "settings.json"

# Agents whose toolset already reads code AND whose role navigates the
# real codebase. They get ``LSP``.
LSP_INCLUDE_AGENTS: tuple[str, ...] = (
    "qs-create-plan",
    "qs-implement-task",
    "qs-implement-setup-task",
    "qs-review-task",
    "qs-plan-concrete-planner",
    "qs-review-edge-case-hunter",
    "qs-review-acceptance-auditor",
    "qs-setup-task",
)

# Context-starved (blind) reviewers + merge/release agents. They must
# NOT get ``LSP`` — code navigation contradicts their design.
LSP_EXCLUDE_AGENTS: tuple[str, ...] = (
    "qs-plan-critic",
    "qs-plan-dev-proxy",
    "qs-plan-scope-guardian",
    "qs-review-blind-hunter",
    "qs-review-coderabbit",
    "qs-finish-task",
    "qs-release",
)

PYRIGHT_PLUGIN_ID = "pyright-lsp@claude-plugins-official"


def _parse_frontmatter(path: Path) -> tuple[dict, str]:
    """Split an agent file into ``(frontmatter_dict, body_text)``.

    Normalizes CRLF before scanning so a Windows-saved file parses
    identically; raises ``AssertionError`` with a path-anchored message
    on a malformed file (mirrors ``test_opencode_agents._parse_frontmatter``).
    """
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert text.startswith("---\n"), f"{path}: missing opening ``---`` delimiter"
    rest = text[len("---\n") :]
    end = rest.find("\n---\n")
    assert end != -1, f"{path}: missing closing ``---`` delimiter"
    return yaml.safe_load(rest[:end]), rest[end + len("\n---\n") :]


# Sentinel for an agent that OMITS the ``tools:`` field. Per
# ``code.claude.com/docs/en/sub-agents`` an omitted ``tools`` means the
# agent inherits ALL tools — which includes ``LSP``. We model that as
# "LSP present" so the exclude assertion fails meaningfully (catching a
# regression where an exclude agent drops its allowlist and silently
# regains LSP) instead of erroring on a type check (review-fix #01).
_INHERITS_ALL_TOOLS = object()


def _tools(name: str) -> object:
    """Return the agent's tool tokens, or ``_INHERITS_ALL_TOOLS``.

    A string ``tools`` field is split on commas with empty/whitespace
    tokens filtered out (so a trailing comma like ``"Bash, LSP,"`` does
    not yield a phantom ``""`` token — review-fix #01). When ``tools`` is
    absent or not a string, the agent inherits all tools; we return the
    sentinel rather than asserting, so callers can treat it as
    "LSP present".
    """
    fm, _ = _parse_frontmatter(CLAUDE_AGENTS_DIR / f"{name}.md")
    tools = fm.get("tools")
    if not isinstance(tools, str):
        return _INHERITS_ALL_TOOLS
    return [t.strip() for t in tools.split(",") if t.strip()]


def _has_lsp(name: str) -> bool:
    """Whether ``name`` can use the ``LSP`` tool.

    ``LSP`` membership is intentionally an **exact-token** check (not a
    substring ``in`` on the raw frontmatter string) so a hypothetical
    future tool named ``LSPFoo`` can never be mistaken for ``LSP``. An
    agent that inherits all tools counts as having ``LSP``.
    """
    tools = _tools(name)
    if tools is _INHERITS_ALL_TOOLS:
        return True
    assert isinstance(tools, list)  # narrow for the type checker
    return "LSP" in tools


@pytest.mark.parametrize("name", LSP_INCLUDE_AGENTS)
def test_lsp_enabled_on_code_navigating_agents(name: str) -> None:
    assert _has_lsp(name), (
        f"{name}: expected 'LSP' on the tools allowlist — code-navigating "
        "agents must carry the LSP tool for diagnostics + navigation."
    )


@pytest.mark.parametrize("name", LSP_EXCLUDE_AGENTS)
def test_lsp_excluded_on_blind_and_merge_agents(name: str) -> None:
    assert not _has_lsp(name), (
        f"{name}: 'LSP' must NOT be on the tools allowlist — this agent is "
        "context-starved or merge/release-only and must not navigate code "
        "(an omitted 'tools:' line inherits all tools, including LSP — also a "
        "regression)."
    )


def test_include_exclude_partition_is_complete() -> None:
    """The include/exclude lists must partition every on-disk agent.

    Globs ``.claude/agents/*.md`` (the dynamic pattern used by sibling
    ``test_harness_flag_explicit.py``) and asserts the two hardcoded
    lists are (a) disjoint and (b) jointly exhaustive. Without this, a
    newly-added agent file would fall into neither parametrized test and
    the deliberate include/exclude split (AC1) could silently regress.
    """
    on_disk = {p.stem for p in CLAUDE_AGENTS_DIR.glob("*.md")}
    include = set(LSP_INCLUDE_AGENTS)
    exclude = set(LSP_EXCLUDE_AGENTS)

    overlap = include & exclude
    assert not overlap, f"agents appear in BOTH include and exclude lists: {overlap}"

    partition = include | exclude
    missing = on_disk - partition
    extra = partition - on_disk
    assert not missing, (
        f"on-disk agents missing from the include/exclude partition: {missing} "
        "— add each to exactly one list so the LSP split is asserted for it."
    )
    assert not extra, (
        f"include/exclude lists reference non-existent agent files: {extra}"
    )


def test_tools_helper_edge_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover the ``_tools`` / ``_has_lsp`` edge cases (review-fix #01).

    Exercises the three branches the real agent files don't trigger:
    an omitted ``tools:`` field (inherits all tools → LSP present), a
    trailing comma (no phantom empty token), and the exact-token
    membership guard (``LSPFoo`` is not ``LSP``).
    """
    monkeypatch.setattr(f"{__name__}.CLAUDE_AGENTS_DIR", tmp_path)

    # Omitted ``tools:`` field → inherits all tools → counts as LSP-present.
    (tmp_path / "no-tools.md").write_text("---\nname: a\n---\nbody\n", encoding="utf-8")
    assert _tools("no-tools") is _INHERITS_ALL_TOOLS
    assert _has_lsp("no-tools") is True

    # Trailing comma must not yield a phantom empty token.
    (tmp_path / "trailing.md").write_text(
        "---\nname: b\ntools: Bash, LSP,\n---\nbody\n", encoding="utf-8"
    )
    assert _tools("trailing") == ["Bash", "LSP"]
    assert _has_lsp("trailing") is True

    # Exact-token membership: a ``LSPFoo`` tool is NOT ``LSP``.
    (tmp_path / "substr.md").write_text(
        "---\nname: c\ntools: Bash, LSPFoo\n---\nbody\n", encoding="utf-8"
    )
    assert _has_lsp("substr") is False


def test_pyright_plugin_enabled_in_settings() -> None:
    settings = json.loads(CLAUDE_SETTINGS.read_text(encoding="utf-8"))
    enabled = settings.get("enabledPlugins")
    assert isinstance(enabled, dict), (
        ".claude/settings.json must declare an 'enabledPlugins' object"
    )
    assert enabled.get(PYRIGHT_PLUGIN_ID) is True, (
        f"enabledPlugins['{PYRIGHT_PLUGIN_ID}'] must be the JSON boolean true"
    )
