"""Tests for the migrate_artifacts script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts" / "qs_opencode"
SCRIPT = SCRIPTS_DIR / "migrate_artifacts.py"


def run_migrate(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run migrate_artifacts.py with given args."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=check,
    )


@pytest.fixture()
def src_dir(tmp_path: Path) -> Path:
    """Create a source directory with sample artifacts."""
    src = tmp_path / "source"
    src.mkdir()
    return src


@pytest.fixture()
def dst_dir(tmp_path: Path) -> Path:
    """Create a destination directory."""
    dst = tmp_path / "dest"
    dst.mkdir()
    return dst


class TestNamingConvention:
    """Test filename mapping logic."""

    def test_github_issue_number_extracted(self, src_dir: Path, dst_dir: Path) -> None:
        """Files with Github-#N become QS-N.story.md."""
        (src_dir / "bug-Github-#101-pool-target.md").write_text("content101")
        result = run_migrate(str(src_dir), str(dst_dir))
        assert result.returncode == 0
        assert (dst_dir / "QS-101.story.md").exists()
        assert (dst_dir / "QS-101.story.md").read_text() == "content101"

    def test_no_github_number_gets_legacy_slug(self, src_dir: Path, dst_dir: Path) -> None:
        """Files without Github-#N become QS-legacy-{slug}.story.md."""
        (src_dir / "1-1-agentic-development-workflow.md").write_text("content")
        result = run_migrate(str(src_dir), str(dst_dir))
        assert result.returncode == 0
        assert (dst_dir / "QS-legacy-1-1-agentic-development-workflow.story.md").exists()

    def test_github_number_in_middle_of_name(self, src_dir: Path, dst_dir: Path) -> None:
        """Github-#N anywhere in filename is extracted."""
        (src_dir / "1-14-Github-#60-robust-story.md").write_text("c")
        result = run_migrate(str(src_dir), str(dst_dir))
        assert result.returncode == 0
        assert (dst_dir / "QS-60.story.md").exists()

    def test_slug_lowercased_and_hyphenated(self, src_dir: Path, dst_dir: Path) -> None:
        """Non-alphanumeric chars become hyphens, collapsed."""
        (src_dir / "Some--Weird___Name.md").write_text("x")
        result = run_migrate(str(src_dir), str(dst_dir))
        assert result.returncode == 0
        assert (dst_dir / "QS-legacy-some-weird-name.story.md").exists()


class TestSkipping:
    """Test skip behavior when target exists."""

    def test_existing_target_skipped(self, src_dir: Path, dst_dir: Path) -> None:
        """If target already exists, source is skipped."""
        (src_dir / "bug-Github-#101-something.md").write_text("new")
        (dst_dir / "QS-101.story.md").write_text("existing")
        result = run_migrate(str(src_dir), str(dst_dir))
        assert result.returncode == 0
        assert (dst_dir / "QS-101.story.md").read_text() == "existing"
        assert "skipped" in result.stdout.lower()


class TestCollisions:
    """Test naming collision handling."""

    def test_two_sources_same_target_gets_duplicate_suffix(self, src_dir: Path, dst_dir: Path) -> None:
        """When two sources map to same target, second gets -duplicate suffix."""
        (src_dir / "bug-Github-#101-first.md").write_text("first")
        (src_dir / "bug-Github-#101-second.md").write_text("second")
        result = run_migrate(str(src_dir), str(dst_dir))
        assert result.returncode == 0
        assert (dst_dir / "QS-101.story.md").exists()
        assert (dst_dir / "QS-101-duplicate-1.story.md").exists()
        assert "collision" in result.stdout.lower() or "warning" in result.stdout.lower()


class TestSummary:
    """Test summary output and count assertion."""

    def test_summary_counts_correct(self, src_dir: Path, dst_dir: Path) -> None:
        """Summary shows N = migrated + skipped."""
        (src_dir / "bug-Github-#101-a.md").write_text("a")
        (src_dir / "1-1-workflow.md").write_text("b")
        (dst_dir / "QS-101.story.md").write_text("existing")
        result = run_migrate(str(src_dir), str(dst_dir))
        assert result.returncode == 0
        # Should report 2 source, 1 migrated, 1 skipped
        assert "2" in result.stdout  # source count
        assert (
            "1 migrated" in result.stdout.lower()
            or "migrated: 1" in result.stdout.lower()
            or "1 migrated" in result.stdout
        )


class TestDryRun:
    """Test --dry-run mode."""

    def test_dry_run_no_files_written(self, src_dir: Path, dst_dir: Path) -> None:
        """Dry run prints plan but writes nothing."""
        (src_dir / "bug-Github-#101-thing.md").write_text("content")
        result = run_migrate(str(src_dir), str(dst_dir), "--dry-run")
        assert result.returncode == 0
        assert not (dst_dir / "QS-101.story.md").exists()
        assert "dry" in result.stdout.lower() or "DRY" in result.stdout


class TestContentVerbatim:
    """Test that content is copied as-is."""

    def test_content_not_modified(self, src_dir: Path, dst_dir: Path) -> None:
        """Content is copied verbatim, no ref scrubbing."""
        content = "References _bmad-output/ and _qsprocess/ paths"
        (src_dir / "deferred-work.md").write_text(content)
        result = run_migrate(str(src_dir), str(dst_dir))
        assert result.returncode == 0
        target = dst_dir / "QS-legacy-deferred-work.story.md"
        assert target.read_text() == content
