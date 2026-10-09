"""The Control Plane's code version (QS-406 D8): what the daemon has loaded vs. what is on disk.

A leaf module (D17): it imports only ``errors`` and ``paths``.

- ``code_version(main)`` hashes the sorted relative paths and contents of
  ``scripts/qs/control_plane/**/*.py``, ``cp.py``, ``models.py`` and
  ``targets.py`` (``hashed_files``). It is cached on a (path, ``mtime_ns``,
  size) signature; a missing file is hashed as absent and never raises.
- ``loaded_version()`` is the version the daemon process has loaded,
  computed once (as the first statement of ``cli._daemon``). A hashed file
  modified since the package was imported makes it ``"unverified:<hash>"``,
  which no disk hash equals, so the ``code_version`` hook restarts the daemon.
- ``git_busy(main)`` names a file showing a git operation in progress.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

from . import errors, paths

ENTRY_FILES = ("cp.py", "models.py", "targets.py")  # outside control_plane/: the entry and what the messenger imports
GIT_BUSY_FILES = ("index.lock", "MERGE_HEAD", "rebase-merge", "rebase-apply")
UNVERIFIED = "unverified:"

_Signature = tuple[tuple[str, int | None, int | None], ...]
_cache: dict[Path, tuple[_Signature, str]] = {}
_loaded: str | None = None
_loaded_done = False
_future_logged: set[Path] = set()


def _qs(root: Path) -> Path:
    return Path(root) / "scripts" / "qs"


def hashed_files(root: Path) -> list[Path]:
    """Every file the code version covers, under the repo ``root`` (missing entry files included)."""
    qs = _qs(root)
    return sorted({*(qs / "control_plane").rglob("*.py"), *(qs / name for name in ENTRY_FILES)})


def _stat(path: Path) -> os.stat_result | None:
    try:
        return os.stat(path)
    except OSError:
        return None


def _signature(files: list[Path]) -> _Signature:
    out = []
    for f in files:
        st = _stat(f)
        out.append((str(f), None, None) if st is None else (str(f), st.st_mtime_ns, st.st_size))
    return tuple(out)


def code_version(main: Path) -> str:
    """The sha256 of the code on disk under the checkout ``main`` (cached on the files' signature)."""
    root = Path(main)
    files = hashed_files(root)
    signature = _signature(files)
    cached = _cache.get(root)
    if cached is not None and cached[0] == signature:
        return cached[1]
    h = hashlib.sha256()
    qs = _qs(root)
    for f in files:
        rel = f.relative_to(qs).as_posix()
        try:
            data = f.read_bytes()
        except OSError:
            h.update(f"{rel}\0-\0".encode())  # absent
            continue
        h.update(f"{rel}\0{len(data)}\0".encode())
        h.update(data)
    version = h.hexdigest()
    _cache[root] = (signature, version)
    return version


def _log(message: str) -> None:
    sys.stderr.write(f"[cp-codever] {message}\n")
    sys.stderr.flush()


def loaded_version() -> str | None:
    """The version this process has loaded (computed once; ``None`` when there is no main checkout)."""
    global _loaded, _loaded_done
    if _loaded_done:
        return _loaded
    try:
        main = paths.main()
    except errors.CpError:
        _loaded, _loaded_done = None, True
        return None
    version = code_version(main)
    imported_at = sys.modules[__package__].IMPORTED_AT_NS  # read at call time: tests patch it
    now = time.time_ns()
    for f in hashed_files(main):
        st = _stat(f)
        if st is None:
            continue
        if st.st_mtime_ns > now:
            if f not in _future_logged:
                _future_logged.add(f)
                _log(f"{f} has an mtime in the future; ignored")
            continue
        if st.st_mtime_ns >= imported_at:
            version = UNVERIFIED + version  # changed since import: what is loaded is unknown
            break
    _loaded, _loaded_done = version, True
    return version


def git_busy(main: Path) -> str | None:
    """The file showing a git operation in progress in ``main`` (``index.lock``, ``MERGE_HEAD``, a rebase), or ``None``."""
    for name in GIT_BUSY_FILES:
        candidate = Path(main) / ".git" / name
        if candidate.exists():
            return str(candidate)
    return None


def _reset_for_tests() -> None:
    global _loaded, _loaded_done
    _cache.clear()
    _future_logged.clear()
    _loaded, _loaded_done = None, False
