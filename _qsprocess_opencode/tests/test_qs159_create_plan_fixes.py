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

# Compute repo root independently of render_agent internals
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

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

    def test_implement_phase_not_injected_for_other_phases(self, tmp_path: Path) -> None:
        """IMPLEMENT_PHASE should not be in context for non-create-plan phases."""
        agents_dir = tmp_path / ".opencode" / "agents"
        agents_dir.mkdir(parents=True)

        # implement-setup-task template does not use {{IMPLEMENT_PHASE}},
        # so it should render fine without the default
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
# AC #2/#3: create-plan template has routing decision block
# ---------------------------------------------------------------------------


class TestCreatePlanRoutingGuidance:
    @staticmethod
    def _template_content() -> str:
        tmpl = _REPO_ROOT / "_qsprocess_opencode" / "agent_templates" / "qs-create-plan.md.tmpl"
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
        agent = _REPO_ROOT / ".opencode" / "agents" / "qs-setup-task.md"
        return agent.read_text(encoding="utf-8")

    def test_has_mandatory_verification(self) -> None:
        """setup-task must have a MANDATORY VERIFICATION block."""
        content = self._agent_content()
        assert "MANDATORY VERIFICATION" in content

    def test_verification_lists_all_five_files(self) -> None:
        """Verification must list all 5 expected agent files individually."""
        content = self._agent_content()
        assert "qs-create-plan-QS-" in content
        assert "qs-plan-critic-QS-" in content
        assert "qs-plan-concrete-planner-QS-" in content
        assert "qs-plan-dev-proxy-QS-" in content
        assert "qs-plan-scope-guardian-QS-" in content

    def test_verification_uses_exact_filenames_not_glob(self) -> None:
        """Verification should check exact filenames, not rely on a glob pattern."""
        content = self._agent_content()
        verif_start = content.index("MANDATORY VERIFICATION")
        verif_text = content[verif_start:]
        # Should list individual ls commands, not a glob
        assert "ls {{worktree_path}}/.opencode/agents/qs-create-plan-QS-" in verif_text

    def test_verification_before_launcher(self) -> None:
        """Verification must appear BEFORE the launcher step."""
        content = self._agent_content()
        verif_pos = content.index("MANDATORY VERIFICATION")
        launcher_pos = content.index("Tell the user what to do next")
        assert verif_pos < launcher_pos, "Verification must come before launcher"


# ---------------------------------------------------------------------------
# cleanup_worktree.remove_worktree fixes
# ---------------------------------------------------------------------------


class TestRemoveWorktreeFixes:
    def test_uses_cwd_kwarg_not_os_chdir(self, tmp_path: Path) -> None:
        """remove_worktree should pass cwd= to subprocess.run, not call os.chdir."""
        work_dir = tmp_path / "worktree"
        work_dir.mkdir()
        main_wt = tmp_path / "main"
        main_wt.mkdir()

        with (
            patch("cleanup_worktree.get_main_worktree", return_value=main_wt),
            patch("cleanup_worktree.subprocess.run") as mock_run,
            patch("cleanup_worktree.shutil.rmtree"),
        ):
            mock_run.return_value = subprocess.CompletedProcess([], 0, stdout="", stderr="")
            cleanup_worktree.remove_worktree(work_dir)

        # Verify cwd kwarg was passed
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs.get("cwd") == str(main_wt)

    def test_does_not_mutate_cwd(self, tmp_path: Path) -> None:
        """remove_worktree must not change the process CWD."""
        import os

        work_dir = tmp_path / "worktree"
        work_dir.mkdir()
        main_wt = tmp_path / "main"
        main_wt.mkdir()

        with (
            patch("cleanup_worktree.get_main_worktree", return_value=main_wt),
            patch("cleanup_worktree.subprocess.run") as mock_run,
            patch("cleanup_worktree.shutil.rmtree"),
        ):
            mock_run.return_value = subprocess.CompletedProcess([], 0, stdout="", stderr="")
            original_cwd = os.getcwd()
            cleanup_worktree.remove_worktree(work_dir)
            assert os.getcwd() == original_cwd

    def test_handles_get_main_worktree_failure(self, tmp_path: Path) -> None:
        """When get_main_worktree raises, should still attempt rmtree."""
        work_dir = tmp_path / "worktree"
        work_dir.mkdir()

        with (
            patch("cleanup_worktree.get_main_worktree", side_effect=RuntimeError("no worktrees")),
            patch("cleanup_worktree.shutil.rmtree") as mock_rmtree,
        ):
            error = cleanup_worktree.remove_worktree(work_dir)

        assert error is not None
        assert "Could not determine main worktree" in error
        mock_rmtree.assert_called_once()

    def test_git_failure_with_empty_stderr(self, tmp_path: Path) -> None:
        """Git failure with empty stderr should still report an error."""
        work_dir = tmp_path / "worktree"
        work_dir.mkdir()
        main_wt = tmp_path / "main"
        main_wt.mkdir()

        with (
            patch("cleanup_worktree.get_main_worktree", return_value=main_wt),
            patch("cleanup_worktree.subprocess.run") as mock_run,
            patch("cleanup_worktree.shutil.rmtree"),
        ):
            mock_run.return_value = subprocess.CompletedProcess([], 128, stdout="", stderr="")
            error = cleanup_worktree.remove_worktree(work_dir)

        assert error is not None
        assert "exited 128" in error

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


class TestRemoveWorktreePartialFailureStatus:
    """Finding 3: main() should report error status when remove_worktree fails."""

    def _run_main(self, args: list[str]) -> dict:
        captured: dict = {}

        def capture_json(data: dict) -> None:
            captured.update(data)

        with (
            patch("sys.argv", ["cleanup_worktree.py", *args]),
            patch("cleanup_worktree.output_json", side_effect=capture_json),
        ):
            cleanup_worktree.main()
        return captured

    def test_error_status_on_worktree_removal_failure(self, tmp_path: Path) -> None:
        """When remove_worktree returns an error, status should be 'error'."""
        with (
            patch("cleanup_worktree.remove_agent_files", return_value=([], [])),
            patch("cleanup_worktree.remove_worktree", return_value="git worktree remove failed"),
        ):
            result = self._run_main(["--work-dir", str(tmp_path), "--issue", "42", "--force"])

        assert result["status"] == "error"
        assert "failed" in result["message"]

    def test_removed_status_on_success(self, tmp_path: Path) -> None:
        """When remove_worktree returns None, status should be 'removed'."""
        with (
            patch("cleanup_worktree.remove_agent_files", return_value=([], [])),
            patch("cleanup_worktree.remove_worktree", return_value=None),
        ):
            result = self._run_main(["--work-dir", str(tmp_path), "--issue", "42", "--force"])

        assert result["status"] == "removed"
