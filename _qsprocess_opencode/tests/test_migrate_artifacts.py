"""Tests for the migrate_artifacts script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts" / "qs_opencode"
SCRIPT = SCRIPTS_DIR / "migrate_artifacts.py"

# Import module directly for unit tests (F11: coverage-visible tests)
sys.path.insert(0, str(SCRIPTS_DIR))
from migrate_artifacts import _compute_target_name, _slugify, migrate  # noqa: E402


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


# ---------------------------------------------------------------------------
# Direct-import unit tests (F11: visible to coverage)
# ---------------------------------------------------------------------------


class TestSlugifyDirect:
    """Direct tests for _slugify."""

    def test_basic_slug(self) -> None:
        """Basic filename becomes lowercase hyphenated slug."""
        assert _slugify("Some--Weird___Name.md") == "some-weird-name"

    def test_trailing_hyphen_after_truncation(self) -> None:
        """Truncation at a hyphen boundary does not leave trailing hyphen."""
        # Build a slug that would end with a hyphen at max_len=10
        result = _slugify("abcdefghi-jkl.md", max_len=10)
        assert not result.endswith("-")
        assert result == "abcdefghi"

    def test_no_extension(self) -> None:
        """Strip .md extension."""
        assert _slugify("hello-world.md") == "hello-world"


class TestComputeTargetNameDirect:
    """Direct tests for _compute_target_name."""

    def test_with_github_number(self) -> None:
        """Github-#N extracts issue number."""
        assert _compute_target_name("bug-Github-#101-pool.md") == "QS-101.story.md"

    def test_without_github_number(self) -> None:
        """No Github-#N produces legacy slug."""
        assert _compute_target_name("deferred-work.md") == "QS-legacy-deferred-work.story.md"


class TestMigrateDirect:
    """Direct tests for migrate() function."""

    def test_basic_migration(self, src_dir: Path, dst_dir: Path) -> None:
        """Migrate copies files with correct naming."""
        (src_dir / "bug-Github-#42-thing.md").write_text("content42", encoding="utf-8")
        result = migrate(src_dir, dst_dir)
        assert result == 0
        assert (dst_dir / "QS-42.story.md").read_text(encoding="utf-8") == "content42"

    def test_dry_run_writes_nothing(self, src_dir: Path, dst_dir: Path) -> None:
        """Dry run returns 0 but writes no files."""
        (src_dir / "bug-Github-#42-thing.md").write_text("x", encoding="utf-8")
        result = migrate(src_dir, dst_dir, dry_run=True)
        assert result == 0
        assert not (dst_dir / "QS-42.story.md").exists()

    def test_skip_existing(self, src_dir: Path, dst_dir: Path) -> None:
        """Existing target is skipped, content preserved."""
        (src_dir / "bug-Github-#42-thing.md").write_text("new", encoding="utf-8")
        (dst_dir / "QS-42.story.md").write_text("old", encoding="utf-8")
        result = migrate(src_dir, dst_dir)
        assert result == 0
        assert (dst_dir / "QS-42.story.md").read_text(encoding="utf-8") == "old"

    def test_empty_source(self, src_dir: Path, dst_dir: Path) -> None:
        """Empty source directory returns 0."""
        result = migrate(src_dir, dst_dir)
        assert result == 0

    def test_collision_handling(self, src_dir: Path, dst_dir: Path) -> None:
        """Two sources mapping to same target: second gets -duplicate suffix."""
        (src_dir / "bug-Github-#42-first.md").write_text("a", encoding="utf-8")
        (src_dir / "bug-Github-#42-second.md").write_text("b", encoding="utf-8")
        result = migrate(src_dir, dst_dir)
        assert result == 0
        assert (dst_dir / "QS-42.story.md").exists()
        assert (dst_dir / "QS-42-duplicate-1.story.md").exists()


# ---------------------------------------------------------------------------
# Subprocess CLI integration tests (original)
# ---------------------------------------------------------------------------


class TestNamingConvention:
    """Test filename mapping logic via CLI."""

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
        assert "2 source" in result.stdout.lower()
        assert "1 migrated" in result.stdout.lower()


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
