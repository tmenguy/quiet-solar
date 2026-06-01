"""Pin the native-pyright ``LSP`` wiring for the Claude harness (QS-248).

Three assertions, encoded as pin tests so the deliberate include/exclude
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


def _tools(name: str) -> list[str]:
    fm, _ = _parse_frontmatter(CLAUDE_AGENTS_DIR / f"{name}.md")
    tools = fm.get("tools")
    assert isinstance(tools, str), f"{name}: 'tools' frontmatter must be a string"
    return [t.strip() for t in tools.split(",")]


@pytest.mark.parametrize("name", LSP_INCLUDE_AGENTS)
def test_lsp_enabled_on_code_navigating_agents(name: str) -> None:
    assert "LSP" in _tools(name), (
        f"{name}: expected 'LSP' on the tools allowlist — code-navigating "
        "agents must carry the LSP tool for diagnostics + navigation."
    )


@pytest.mark.parametrize("name", LSP_EXCLUDE_AGENTS)
def test_lsp_excluded_on_blind_and_merge_agents(name: str) -> None:
    assert "LSP" not in _tools(name), (
        f"{name}: 'LSP' must NOT be on the tools allowlist — this agent is "
        "context-starved or merge/release-only and must not navigate code."
    )


def test_pyright_plugin_enabled_in_settings() -> None:
    settings = json.loads(CLAUDE_SETTINGS.read_text(encoding="utf-8"))
    enabled = settings.get("enabledPlugins")
    assert isinstance(enabled, dict), (
        ".claude/settings.json must declare an 'enabledPlugins' object"
    )
    assert enabled.get(PYRIGHT_PLUGIN_ID) is True, (
        f"enabledPlugins['{PYRIGHT_PLUGIN_ID}'] must be the JSON boolean true"
    )
