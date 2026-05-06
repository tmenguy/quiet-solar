"""Tests for cleanup_worktree.py functions.

Run with: source venv/bin/activate && pytest _qsprocess_opencode/tests/ -v
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import cleanup_worktree

# ---------------------------------------------------------------------------
# check_worktree_status
# ---------------------------------------------------------------------------


class TestCheckWorktreeStatus:
    def _mock_run(self, results: list[subprocess.CompletedProcess]) -> MagicMock:
        return patch("cleanup_worktree.subprocess.run", side_effect=results)

    def test_clean_worktree(self, tmp_path: Path) -> None:
        results = [
            subprocess.CompletedProcess([], 0, stdout="QS_42\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        ]
        with self._mock_run(results):
            status = cleanup_worktree.check_worktree_status(tmp_path)

        assert status["safe_to_remove"] is True
        assert status["uncommitted_files"] == []
        assert status["unpushed_commits"] == 0
        assert status["branch"] == "QS_42"

    def test_dirty_worktree_uncommitted(self, tmp_path: Path) -> None:
        results = [
            subprocess.CompletedProcess([], 0, stdout="QS_42\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout=" M file1.py\n?? file2.py\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        ]
        with self._mock_run(results):
            status = cleanup_worktree.check_worktree_status(tmp_path)

        assert status["safe_to_remove"] is False
        assert len(status["uncommitted_files"]) == 2

    def test_dirty_worktree_unpushed(self, tmp_path: Path) -> None:
        results = [
            subprocess.CompletedProcess([], 0, stdout="QS_42\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="abc123 some commit\ndef456 another\n", stderr=""),
        ]
        with self._mock_run(results):
            status = cleanup_worktree.check_worktree_status(tmp_path)

        assert status["safe_to_remove"] is False
        assert status["unpushed_commits"] == 2

    def test_no_upstream(self, tmp_path: Path) -> None:
        results = [
            subprocess.CompletedProcess([], 0, stdout="QS_42\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess([], 128, stdout="", stderr="fatal: no upstream"),
        ]
        with self._mock_run(results):
            status = cleanup_worktree.check_worktree_status(tmp_path)

        assert status["safe_to_remove"] is False
        assert status["unpushed_commits"] == -1

    def test_branch_detection_failure(self, tmp_path: Path) -> None:
        results = [
            subprocess.CompletedProcess([], 128, stdout="", stderr="fatal"),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        ]
        with self._mock_run(results):
            status = cleanup_worktree.check_worktree_status(tmp_path)

        assert status["branch"] == "unknown"


# ---------------------------------------------------------------------------
# remove_agent_files
# ---------------------------------------------------------------------------


class TestRemoveAgentFiles:
    def test_removes_matching_files(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / ".opencode" / "agents"
        agent_dir.mkdir(parents=True)
        (agent_dir / "qs-create-plan-QS-42.md").write_text("x")
        (agent_dir / "qs-implement-task-QS-42.md").write_text("x")
        (agent_dir / "qs-create-plan-QS-99.md").write_text("x")  # different issue

        removed = cleanup_worktree.remove_agent_files(tmp_path, 42)

        assert len(removed) == 2
        assert not (agent_dir / "qs-create-plan-QS-42.md").exists()
        assert not (agent_dir / "qs-implement-task-QS-42.md").exists()
        assert (agent_dir / "qs-create-plan-QS-99.md").exists()

    def test_no_agent_dir(self, tmp_path: Path) -> None:
        removed = cleanup_worktree.remove_agent_files(tmp_path, 42)
        assert removed == []


# ---------------------------------------------------------------------------
# list_agent_files
# ---------------------------------------------------------------------------


class TestListAgentFiles:
    def test_lists_matching_files(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / ".opencode" / "agents"
        agent_dir.mkdir(parents=True)
        (agent_dir / "qs-create-plan-QS-42.md").write_text("x")
        (agent_dir / "qs-implement-task-QS-42.md").write_text("x")

        result = cleanup_worktree.list_agent_files(tmp_path, 42)
        assert len(result) == 2

    def test_no_agent_dir(self, tmp_path: Path) -> None:
        assert cleanup_worktree.list_agent_files(tmp_path, 42) == []


# ---------------------------------------------------------------------------
# push_branch
# ---------------------------------------------------------------------------


class TestPushBranch:
    def test_success(self, tmp_path: Path) -> None:
        with patch("cleanup_worktree.subprocess.run") as mock:
            mock.return_value = subprocess.CompletedProcess([], 0, stdout="Everything up-to-date", stderr="")
            ok, out = cleanup_worktree.push_branch(tmp_path)
        assert ok is True

    def test_failure(self, tmp_path: Path) -> None:
        with patch("cleanup_worktree.subprocess.run") as mock:
            mock.return_value = subprocess.CompletedProcess([], 1, stdout="", stderr="rejected")
            ok, out = cleanup_worktree.push_branch(tmp_path)
        assert ok is False
        assert "rejected" in out


# ---------------------------------------------------------------------------
# main (CLI integration)
# ---------------------------------------------------------------------------


class TestMain:
    def _run_main(self, args: list[str]) -> dict:
        """Run main() with args and capture JSON output."""
        with patch("sys.argv", ["cleanup_worktree.py", *args]):
            captured: dict = {}

            def capture_json(data: dict) -> None:
                captured.update(data)

            with patch("cleanup_worktree.output_json", side_effect=capture_json):
                cleanup_worktree.main()
            return captured

    def test_nonexistent_dir(self) -> None:
        result = self._run_main(["--work-dir", "/nonexistent/path", "--issue", "42"])
        assert result["status"] == "error"

    def test_dirty_returns_action_required(self, tmp_path: Path) -> None:
        with patch("cleanup_worktree.check_worktree_status") as mock:
            mock.return_value = {
                "safe_to_remove": False,
                "uncommitted_files": ["dirty.py"],
                "unpushed_commits": 1,
                "branch": "QS_42",
            }
            result = self._run_main(["--work-dir", str(tmp_path), "--issue", "42"])

        assert result["status"] == "action_required"
        assert "--force" in result["options"]
        assert "--push-first" in result["options"]

    def test_clean_removes(self, tmp_path: Path) -> None:
        with (
            patch("cleanup_worktree.check_worktree_status") as mock_status,
            patch("cleanup_worktree.remove_agent_files", return_value=[]) as mock_agents,
            patch("cleanup_worktree.remove_worktree", return_value=None) as mock_wt,
        ):
            mock_status.return_value = {
                "safe_to_remove": True,
                "uncommitted_files": [],
                "unpushed_commits": 0,
                "branch": "QS_42",
            }
            result = self._run_main(["--work-dir", str(tmp_path), "--issue", "42"])

        assert result["status"] == "removed"
        mock_agents.assert_called_once()
        mock_wt.assert_called_once()

    def test_force_skips_status_check(self, tmp_path: Path) -> None:
        with (
            patch("cleanup_worktree.check_worktree_status") as mock_status,
            patch("cleanup_worktree.remove_agent_files", return_value=[]),
            patch("cleanup_worktree.remove_worktree", return_value=None),
        ):
            result = self._run_main(["--work-dir", str(tmp_path), "--issue", "42", "--force"])

        assert result["status"] == "removed"
        mock_status.assert_not_called()

    def test_push_first_success(self, tmp_path: Path) -> None:
        with (
            patch("cleanup_worktree.check_worktree_status") as mock_status,
            patch("cleanup_worktree.push_branch", return_value=(True, "ok")),
            patch("cleanup_worktree.remove_agent_files", return_value=[]),
            patch("cleanup_worktree.remove_worktree", return_value=None),
        ):
            mock_status.return_value = {
                "safe_to_remove": False,
                "uncommitted_files": [],
                "unpushed_commits": 2,
                "branch": "QS_42",
            }
            result = self._run_main(["--work-dir", str(tmp_path), "--issue", "42", "--push-first"])

        assert result["status"] == "removed"

    def test_push_first_failure(self, tmp_path: Path) -> None:
        with (
            patch("cleanup_worktree.check_worktree_status") as mock_status,
            patch("cleanup_worktree.push_branch", return_value=(False, "rejected")),
        ):
            mock_status.return_value = {
                "safe_to_remove": False,
                "uncommitted_files": [],
                "unpushed_commits": 2,
                "branch": "QS_42",
            }
            result = self._run_main(["--work-dir", str(tmp_path), "--issue", "42", "--push-first"])

        assert result["status"] == "error"
        assert "rejected" in result["message"]

    def test_dry_run(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / ".opencode" / "agents"
        agent_dir.mkdir(parents=True)
        (agent_dir / "qs-create-plan-QS-42.md").write_text("x")

        with patch("cleanup_worktree.check_worktree_status") as mock_status:
            mock_status.return_value = {
                "safe_to_remove": True,
                "uncommitted_files": [],
                "unpushed_commits": 0,
                "branch": "QS_42",
            }
            result = self._run_main(["--work-dir", str(tmp_path), "--issue", "42", "--dry-run"])

        assert result["status"] == "dry_run"
        assert len(result["would_remove_agents"]) == 1
        # File should still exist
        assert (agent_dir / "qs-create-plan-QS-42.md").exists()

    def test_dry_run_push_first(self, tmp_path: Path) -> None:
        with patch("cleanup_worktree.check_worktree_status") as mock_status:
            mock_status.return_value = {
                "safe_to_remove": False,
                "uncommitted_files": [],
                "unpushed_commits": 2,
                "branch": "QS_42",
            }
            result = self._run_main(["--work-dir", str(tmp_path), "--issue", "42", "--push-first", "--dry-run"])

        assert result["status"] == "dry_run"
        assert result["would_push"] is True

    def test_push_first_with_uncommitted_blocks(self, tmp_path: Path) -> None:
        """--push-first should refuse if there are uncommitted files."""
        with patch("cleanup_worktree.check_worktree_status") as mock_status:
            mock_status.return_value = {
                "safe_to_remove": False,
                "uncommitted_files": ["dirty.py"],
                "unpushed_commits": 2,
                "branch": "QS_42",
            }
            result = self._run_main(["--work-dir", str(tmp_path), "--issue", "42", "--push-first"])

        assert result["status"] == "action_required"
        assert "--force" in result["options"]
        assert "--push-first" not in result.get("options", {})
        assert "uncommitted" in result["message"].lower()


# ---------------------------------------------------------------------------
# Reorder: agent files removed BEFORE status check (QS-155)
# ---------------------------------------------------------------------------


class TestAgentRemovalBeforeStatusCheck:
    """Verify remove_agent_files runs before check_worktree_status in main()."""

    def _run_main(self, args: list[str]) -> dict:
        """Run main() with args and capture JSON output."""
        with patch("sys.argv", ["cleanup_worktree.py", *args]):
            captured: dict = {}

            def capture_json(data: dict) -> None:
                captured.update(data)

            with patch("cleanup_worktree.output_json", side_effect=capture_json):
                cleanup_worktree.main()
            return captured

    def test_agent_only_worktree_cleans_successfully(self, tmp_path: Path) -> None:
        """Worktree with ONLY agent files should clean up (status=removed).

        Before the fix, check_worktree_status saw untracked agent .md files
        and returned action_required, aborting before removal.
        """
        # Create agent files
        agent_dir = tmp_path / ".opencode" / "agents"
        agent_dir.mkdir(parents=True)
        (agent_dir / "qs-create-plan-QS-42.md").write_text("x")
        (agent_dir / "qs-review-task-QS-42.md").write_text("x")

        call_order: list[str] = []

        original_remove = cleanup_worktree.remove_agent_files
        original_check = cleanup_worktree.check_worktree_status

        def tracking_remove(work_dir: Path, issue: int) -> list[str]:
            call_order.append("remove_agent_files")
            return original_remove(work_dir, issue)

        def tracking_check(work_dir: Path) -> dict:
            call_order.append("check_worktree_status")
            # After agent removal, tree should be clean
            return {
                "safe_to_remove": True,
                "uncommitted_files": [],
                "unpushed_commits": 0,
                "branch": "QS_42",
            }

        with (
            patch("cleanup_worktree.remove_agent_files", side_effect=tracking_remove),
            patch("cleanup_worktree.check_worktree_status", side_effect=tracking_check),
            patch("cleanup_worktree.remove_worktree", return_value=None),
        ):
            result = self._run_main(["--work-dir", str(tmp_path), "--issue", "42"])

        assert result["status"] == "removed"
        assert call_order.index("remove_agent_files") < call_order.index("check_worktree_status")

    def test_agent_files_plus_dirty_still_blocks(self, tmp_path: Path) -> None:
        """Agent files removed first, but real dirty files still trigger action_required."""
        agent_dir = tmp_path / ".opencode" / "agents"
        agent_dir.mkdir(parents=True)
        (agent_dir / "qs-create-plan-QS-42.md").write_text("x")

        call_order: list[str] = []

        def tracking_remove(work_dir: Path, issue: int) -> list[str]:
            call_order.append("remove_agent_files")
            return [".opencode/agents/qs-create-plan-QS-42.md"]

        def tracking_check(work_dir: Path) -> dict:
            call_order.append("check_worktree_status")
            return {
                "safe_to_remove": False,
                "uncommitted_files": ["dirty.py"],
                "unpushed_commits": 0,
                "branch": "QS_42",
            }

        with (
            patch("cleanup_worktree.remove_agent_files", side_effect=tracking_remove),
            patch("cleanup_worktree.check_worktree_status", side_effect=tracking_check),
        ):
            result = self._run_main(["--work-dir", str(tmp_path), "--issue", "42"])

        assert result["status"] == "action_required"
        assert call_order.index("remove_agent_files") < call_order.index("check_worktree_status")

    def test_force_path_also_removes_agents_first(self, tmp_path: Path) -> None:
        """--force should also remove agent files before worktree deletion."""
        call_order: list[str] = []

        def tracking_remove(work_dir: Path, issue: int) -> list[str]:
            call_order.append("remove_agent_files")
            return []

        def tracking_wt_remove(work_dir: Path) -> str | None:
            call_order.append("remove_worktree")
            return None

        with (
            patch("cleanup_worktree.remove_agent_files", side_effect=tracking_remove),
            patch("cleanup_worktree.remove_worktree", side_effect=tracking_wt_remove),
        ):
            result = self._run_main(["--work-dir", str(tmp_path), "--issue", "42", "--force"])

        assert result["status"] == "removed"
        assert call_order.index("remove_agent_files") < call_order.index("remove_worktree")

    def test_dry_run_lists_agents_without_removing(self, tmp_path: Path) -> None:
        """--dry-run should list agent files but not remove them."""
        agent_dir = tmp_path / ".opencode" / "agents"
        agent_dir.mkdir(parents=True)
        (agent_dir / "qs-create-plan-QS-42.md").write_text("x")

        result = self._run_main(["--work-dir", str(tmp_path), "--issue", "42", "--dry-run"])

        assert result["status"] == "dry_run"
        assert len(result["would_remove_agents"]) == 1
        # File should still exist
        assert (agent_dir / "qs-create-plan-QS-42.md").exists()
