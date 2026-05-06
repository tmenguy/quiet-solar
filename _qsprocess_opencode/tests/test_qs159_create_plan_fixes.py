"""Tests for QS-159: Fix create-plan phase handoff and subagent type errors.

Covers:
- AC #1: IMPLEMENT_PHASE defaults to implement-task in render_agent.py context
- AC #2/#3: create-plan template has routing guidance for implement-phase selection
- AC #4: setup-task agent has mandatory verification step after rendering
- Bonus: cleanup_worktree.remove_worktree CWD fix and error surfacing

Run with: source venv/bin/activate && pytest _qsprocess_opencode/tests/ -v
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import cleanup_worktree
import render_agent

# ---------------------------------------------------------------------------
# AC #1: IMPLEMENT_PHASE default in render_agent.py
# ---------------------------------------------------------------------------


class TestImplementPhaseDefault:
    def test_implement_phase_in_default_context(self, tmp_path: Path) -> None:
        """render_agent should include IMPLEMENT_PHASE=implement-task by default."""
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
            # Should NOT raise — IMPLEMENT_PHASE should be provided by default
            render_agent.main()

        out = agents_dir / "qs-create-plan-QS-99.md"
        assert out.is_file()
        content = out.read_text()
        # The rendered output should contain "implement-task" (the default)
        assert "implement-task" in content

    def test_implement_phase_extra_overrides_default(self, tmp_path: Path) -> None:
        """--extra IMPLEMENT_PHASE=implement-setup-task should override the default."""
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
            "IMPLEMENT_PHASE=implement-setup-task",
        ]
        with patch("sys.argv", args), patch("render_agent.output_json"):
            render_agent.main()

        out = agents_dir / "qs-create-plan-QS-99.md"
        content = out.read_text()
        assert "implement-setup-task" in content


# ---------------------------------------------------------------------------
# AC #2/#3: create-plan template has routing decision block
# ---------------------------------------------------------------------------


class TestCreatePlanRoutingGuidance:
    @staticmethod
    def _template_content() -> str:
        repo_root = render_agent._repo_root()
        tmpl = repo_root / "_qsprocess_opencode" / "agent_templates" / "qs-create-plan.md.tmpl"
        return tmpl.read_text(encoding="utf-8")

    def test_has_routing_decision_block(self) -> None:
        """Phase 7 should contain a routing decision block for implement phase."""
        content = self._template_content()
        phase7_start = content.index("### Phase 7")
        phase7_text = content[phase7_start:]
        assert "Determine implement phase" in phase7_text

    def test_lists_dev_environment_patterns(self) -> None:
        """The routing block should list dev-environment file patterns."""
        content = self._template_content()
        phase7_start = content.index("### Phase 7")
        phase7_text = content[phase7_start:]
        assert "_qsprocess_opencode/**" in phase7_text
        assert "scripts/**" in phase7_text
        assert ".opencode/**" in phase7_text

    def test_mentions_next_phase_flag(self) -> None:
        """The routing block should mention --next-phase for next_step.py."""
        content = self._template_content()
        phase7_start = content.index("### Phase 7")
        phase7_text = content[phase7_start:]
        assert "--next-phase" in phase7_text

    def test_mentions_implement_setup_task(self) -> None:
        """The routing block should mention implement-setup-task as an option."""
        content = self._template_content()
        phase7_start = content.index("### Phase 7")
        phase7_text = content[phase7_start:]
        assert "implement-setup-task" in phase7_text


# ---------------------------------------------------------------------------
# AC #4: setup-task has mandatory verification step
# ---------------------------------------------------------------------------


class TestSetupTaskVerification:
    @staticmethod
    def _agent_content() -> str:
        repo_root = render_agent._repo_root()
        agent = repo_root / ".opencode" / "agents" / "qs-setup-task.md"
        return agent.read_text(encoding="utf-8")

    def test_has_mandatory_verification(self) -> None:
        """setup-task must have a MANDATORY VERIFICATION block."""
        content = self._agent_content()
        assert "MANDATORY VERIFICATION" in content

    def test_verification_lists_all_five_files(self) -> None:
        """Verification must list all 5 expected agent files."""
        content = self._agent_content()
        assert "qs-create-plan-QS-" in content
        assert "qs-plan-critic-QS-" in content
        assert "qs-plan-concrete-planner-QS-" in content
        assert "qs-plan-dev-proxy-QS-" in content
        assert "qs-plan-scope-guardian-QS-" in content

    def test_verification_before_launcher(self) -> None:
        """Verification must appear BEFORE the launcher step."""
        content = self._agent_content()
        verif_pos = content.index("MANDATORY VERIFICATION")
        launcher_pos = content.index("Tell the user what to do next")
        assert verif_pos < launcher_pos, "Verification must come before launcher"


# ---------------------------------------------------------------------------
# Bonus: cleanup_worktree.remove_worktree CWD fix
# ---------------------------------------------------------------------------


class TestRemoveWorktreeCwdFix:
    def test_changes_cwd_before_removal(self, tmp_path: Path) -> None:
        """remove_worktree should chdir out of the worktree before deleting it."""
        import os

        work_dir = tmp_path / "worktree"
        work_dir.mkdir()
        main_wt = tmp_path / "main"
        main_wt.mkdir()

        cwd_during_rmtree = []

        original_rmtree = cleanup_worktree.shutil.rmtree

        def tracking_rmtree(path: str | Path, **kwargs: object) -> None:
            cwd_during_rmtree.append(os.getcwd())
            # Don't actually remove in test

        with (
            patch("cleanup_worktree.get_main_worktree", return_value=main_wt),
            patch("cleanup_worktree.subprocess.run") as mock_run,
            patch("cleanup_worktree.shutil.rmtree", side_effect=tracking_rmtree),
        ):
            mock_run.return_value = subprocess.CompletedProcess([], 0, stdout="", stderr="")
            # Start inside the worktree
            old_cwd = os.getcwd()
            os.chdir(str(work_dir))
            try:
                cleanup_worktree.remove_worktree(work_dir)
            finally:
                os.chdir(old_cwd)

        assert len(cwd_during_rmtree) == 1
        assert cwd_during_rmtree[0] == str(main_wt)

    def test_rmtree_error_surfaced(self, tmp_path: Path) -> None:
        """shutil.rmtree errors should be returned, not silently swallowed."""
        work_dir = tmp_path / "worktree"
        work_dir.mkdir()
        main_wt = tmp_path / "main"
        main_wt.mkdir()

        with (
            patch("cleanup_worktree.get_main_worktree", return_value=main_wt),
            patch("cleanup_worktree.subprocess.run") as mock_run,
            patch("cleanup_worktree.shutil.rmtree", side_effect=OSError("busy")),
        ):
            mock_run.return_value = subprocess.CompletedProcess([], 0, stdout="", stderr="")
            error = cleanup_worktree.remove_worktree(work_dir)

        assert error is not None
        assert "shutil.rmtree failed" in error
