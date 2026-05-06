"""Tests for QS-147: implement-setup-task phase registration and template.

Covers:
- render_agent.py accepts --phase implement-setup-task (AC 5)
- next_step.py handles implement-setup-task transitions (AC 6)
- create-plan template supports {{IMPLEMENT_PHASE}} placeholder (AC 7)
- qs-implement-setup-task.md.tmpl has correct permissions and quality gate (AC 3, 4)
- .gitignore pattern excludes per-task agents but not static ones (AC 1)

Run with: source venv/bin/activate && pytest _qsprocess_opencode/tests/ -v
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from unittest.mock import patch

import next_step
import pytest
import render_agent

# ---------------------------------------------------------------------------
# render_agent.py — PHASE_TEMPLATES registration (AC 5)
# ---------------------------------------------------------------------------


class TestRenderAgentPhaseRegistration:
    def test_implement_setup_task_in_phase_templates(self) -> None:
        """implement-setup-task must be a known phase."""
        assert "implement-setup-task" in render_agent.PHASE_TEMPLATES

    def test_implement_setup_task_template_file_exists(self) -> None:
        """The template file referenced by the mapping must exist."""
        tmpl_name = render_agent.PHASE_TEMPLATES["implement-setup-task"]
        repo_root = render_agent._repo_root()
        tmpl_path = repo_root / "_qsprocess_opencode" / "agent_templates" / tmpl_name
        assert tmpl_path.is_file(), f"Template not found: {tmpl_path}"

    def test_implement_setup_task_renders(self, tmp_path: Path) -> None:
        """render_agent.main() should succeed with --phase implement-setup-task."""
        agents_dir = tmp_path / ".opencode" / "agents"
        agents_dir.mkdir(parents=True)

        args = [
            "render_agent.py",
            "--phase",
            "implement-setup-task",
            "--work-dir",
            str(tmp_path),
            "--issue",
            "99",
            "--title",
            "Test story",
            "--story-file",
            "_qsprocess_opencode/stories/QS-99.story.md",
        ]
        with patch("sys.argv", args), patch("render_agent.output_json"):
            render_agent.main()

        out = agents_dir / "qs-implement-setup-task-QS-99.md"
        assert out.is_file()


# ---------------------------------------------------------------------------
# next_step.py — PHASE_TRANSITIONS registration (AC 6)
# ---------------------------------------------------------------------------


class TestNextStepTransition:
    def test_implement_setup_task_in_transitions(self) -> None:
        """implement-setup-task must be a known finishing phase."""
        assert "implement-setup-task" in next_step.PHASE_TRANSITIONS

    def test_implement_setup_task_transitions_to_review_task(self) -> None:
        """implement-setup-task should transition to review-task, same as implement-task."""
        transition = next_step.PHASE_TRANSITIONS["implement-setup-task"]
        assert transition is not None
        assert transition["next_agent_phase"] == "review-task"

    def test_implement_setup_task_renders_same_review_agents(self) -> None:
        """implement-setup-task should render the same 5 review agents."""
        t_setup = next_step.PHASE_TRANSITIONS["implement-setup-task"]
        t_impl = next_step.PHASE_TRANSITIONS["implement-task"]
        assert t_setup["render_phases"] == t_impl["render_phases"]


# ---------------------------------------------------------------------------
# qs-implement-setup-task.md.tmpl — permissions and quality gate (AC 3, 4)
# ---------------------------------------------------------------------------


class TestImplementSetupTaskTemplate:
    @pytest.fixture()
    def template_content(self) -> str:
        repo_root = render_agent._repo_root()
        tmpl = repo_root / "_qsprocess_opencode" / "agent_templates" / "qs-implement-setup-task.md.tmpl"
        return tmpl.read_text(encoding="utf-8")

    def test_allows_qsprocess_opencode(self, template_content: str) -> None:
        """Edit permissions must allow _qsprocess_opencode/**."""
        assert '"_qsprocess_opencode/**": allow' in template_content

    def test_allows_scripts_qs_opencode(self, template_content: str) -> None:
        """Edit permissions must allow scripts/qs_opencode/**."""
        assert '"scripts/qs_opencode/**": allow' in template_content

    def test_denies_custom_components(self, template_content: str) -> None:
        """Must NOT allow custom_components/quiet_solar/**."""
        assert '"custom_components/quiet_solar/**": allow' not in template_content

    def test_denies_tests(self, template_content: str) -> None:
        """Must NOT allow tests/**."""
        assert '"tests/**": allow' not in template_content

    def test_quality_gate_uses_pytest(self, template_content: str) -> None:
        """Quality gate must run pytest on _qsprocess_opencode/tests/."""
        assert "pytest _qsprocess_opencode/tests/" in template_content

    def test_quality_gate_includes_ruff(self, template_content: str) -> None:
        """Quality gate must include ruff check."""
        assert "ruff check _qsprocess_opencode/" in template_content

    def test_quality_gate_not_full(self, template_content: str) -> None:
        """Must NOT reference the full quality gate script."""
        assert "python scripts/qs/quality_gate.py" not in template_content


# ---------------------------------------------------------------------------
# qs-create-plan.md.tmpl — {{IMPLEMENT_PHASE}} placeholder (AC 7)
# ---------------------------------------------------------------------------


class TestCreatePlanTemplate:
    @pytest.fixture()
    def template_content(self) -> str:
        repo_root = render_agent._repo_root()
        tmpl = repo_root / "_qsprocess_opencode" / "agent_templates" / "qs-create-plan.md.tmpl"
        return tmpl.read_text(encoding="utf-8")

    def test_uses_implement_phase_placeholder(self, template_content: str) -> None:
        """The template must reference {{IMPLEMENT_PHASE}} instead of hardcoded implement-task."""
        assert "{{IMPLEMENT_PHASE}}" in template_content

    def test_no_hardcoded_implement_task_in_phase7(self, template_content: str) -> None:
        """Phase 7 render/handoff commands must not hardcode implement-task."""
        phase7_start = template_content.index("### Phase 7")
        phase7_text = template_content[phase7_start:]
        assert "--phase implement-task" not in phase7_text
        assert "{{IMPLEMENT_PHASE}}" in phase7_text

    def test_renders_fails_without_implement_phase(self, tmp_path: Path) -> None:
        """When IMPLEMENT_PHASE is not provided, rendering should fail."""
        agents_dir = tmp_path / ".opencode" / "agents"
        agents_dir.mkdir(parents=True)

        args = [
            "render_agent.py",
            "--phase",
            "create-plan",
            "--work-dir",
            str(tmp_path),
            "--issue",
            "99",
            "--title",
            "Test story",
            "--story-file",
            "_qsprocess_opencode/stories/QS-99.story.md",
        ]
        with patch("sys.argv", args), patch("render_agent.output_json"):
            with pytest.raises(SystemExit) as exc_info:
                render_agent.main()
            assert exc_info.value.code is not None and exc_info.value.code != 0

    def test_renders_with_implement_phase_extra(self, tmp_path: Path) -> None:
        """When IMPLEMENT_PHASE=implement-task is passed, rendering succeeds."""
        agents_dir = tmp_path / ".opencode" / "agents"
        agents_dir.mkdir(parents=True)

        args = [
            "render_agent.py",
            "--phase",
            "create-plan",
            "--work-dir",
            str(tmp_path),
            "--issue",
            "99",
            "--title",
            "Test story",
            "--story-file",
            "_qsprocess_opencode/stories/QS-99.story.md",
            "--extra",
            "IMPLEMENT_PHASE=implement-task",
        ]
        with patch("sys.argv", args), patch("render_agent.output_json"):
            render_agent.main()

        out = agents_dir / "qs-create-plan-QS-99.md"
        assert out.is_file()
        content = out.read_text()
        assert "implement-task" in content


# ---------------------------------------------------------------------------
# .gitignore — per-task agent pattern (AC 1)
# ---------------------------------------------------------------------------


class TestGitignorePattern:
    def test_gitignore_has_pattern(self) -> None:
        """Root .gitignore must contain the per-task agent exclusion pattern."""
        repo_root = render_agent._repo_root()
        gitignore = repo_root / ".gitignore"
        assert gitignore.is_file(), ".gitignore must exist"
        content = gitignore.read_text()
        assert "qs-*-QS-*.md" in content

    def test_static_agents_not_matched(self) -> None:
        """The pattern qs-*-QS-*.md must NOT match static agents."""
        pattern = "qs-*-QS-*.md"
        assert not fnmatch.fnmatch("qs-setup-task.md", pattern)
        assert not fnmatch.fnmatch("qs-release.md", pattern)
        assert fnmatch.fnmatch("qs-create-plan-QS-42.md", pattern)
        assert fnmatch.fnmatch("qs-implement-task-QS-147.md", pattern)


# ---------------------------------------------------------------------------
# next_step.py — --next-phase override (Finding 1)
# ---------------------------------------------------------------------------


class TestNextStepNextPhaseOverride:
    def _run_next_step(self, extra_args: list[str] | None = None) -> dict:
        """Helper to run next_step.main() and capture output."""
        captured: dict = {}

        def capture(data: dict) -> None:
            captured.update(data)

        args = [
            "next_step.py",
            "--phase",
            "create-plan",
            "--issue",
            "99",
            "--work-dir",
            "/tmp/test",
            "--title",
            "Test",
            "--story-file",
            "story.md",
            *(extra_args or []),
        ]
        with (
            patch("sys.argv", args),
            patch("next_step.output_json", side_effect=capture),
            patch("next_step.build_launcher_payload", return_value={"new_context": "x"}),
        ):
            next_step.main()
        return captured

    def test_next_phase_overrides_default(self) -> None:
        """--next-phase should override the default transition target."""
        captured = self._run_next_step(["--next-phase", "implement-setup-task"])

        assert captured["next_phase"] == "implement-setup-task"
        assert captured["next_agent"] == "qs-implement-setup-task-QS-99"
        assert any("implement-setup-task" in cmd for cmd in captured["render_commands"])

    def test_no_next_phase_uses_default(self) -> None:
        """Without --next-phase, default transition is used."""
        captured = self._run_next_step()

        assert captured["next_phase"] == "implement-task"
        assert captured["next_agent"] == "qs-implement-task-QS-99"

    def test_next_phase_rejects_invalid_value(self) -> None:
        """--next-phase with an invalid value should be rejected by argparse."""
        with pytest.raises(SystemExit) as exc_info:
            self._run_next_step(["--next-phase", "nonexistent-phase"])
        assert exc_info.value.code != 0

    def test_next_phase_on_multi_render_raises(self) -> None:
        """--next-phase on a multi-render transition should raise an error."""
        captured: dict = {}

        def capture(data: dict) -> None:
            captured.update(data)

        args = [
            "next_step.py",
            "--phase",
            "implement-task",
            "--issue",
            "99",
            "--work-dir",
            "/tmp/test",
            "--title",
            "Test",
            "--story-file",
            "story.md",
            "--pr",
            "5",
            "--next-phase",
            "finish-task",
        ]
        with (
            patch("sys.argv", args),
            patch("next_step.output_json", side_effect=capture),
            patch("next_step.build_launcher_payload", return_value={"new_context": "x"}),
            pytest.raises(SystemExit),
        ):
            next_step.main()

    def test_next_phase_self_loop_raises(self) -> None:
        """--next-phase equal to --phase should raise a self-loop error."""
        args = [
            "next_step.py",
            "--phase",
            "create-plan",
            "--issue",
            "99",
            "--work-dir",
            "/tmp/test",
            "--title",
            "Test",
            "--story-file",
            "story.md",
            "--next-phase",
            "create-plan",
        ]
        with (
            patch("sys.argv", args),
            patch("next_step.output_json"),
            patch("next_step.build_launcher_payload", return_value={"new_context": "x"}),
            pytest.raises(SystemExit, match="self-loop"),
        ):
            next_step.main()

    def test_next_phase_overridden_flag(self) -> None:
        """Output JSON should contain next_phase_overridden boolean."""
        captured_default = self._run_next_step()
        assert captured_default["next_phase_overridden"] is False

        captured_override = self._run_next_step(["--next-phase", "implement-setup-task"])
        assert captured_override["next_phase_overridden"] is True
