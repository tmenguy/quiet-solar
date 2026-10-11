"""Where things live: the code root, the main checkout, the live DB, the path guard (§2).

Only the main checkout's code may open the live DB (``<MAIN>/harness_state.db``).
A worktree's copy of the code, or any code under pytest, is refused with
``PATH_GUARD`` — use ``QS_CP_DB`` to point at a temporary DB instead.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import errors

DB_NAME = "harness_state.db"
DEFAULT_BACKUP_DIR = Path("~/.local/state/quiet-solar/cp-backups")
DEFAULT_MESSENGER_DIR = Path("~/.local/state/quiet-solar/cp-messenger")


def code_root() -> Path:
    """The repo root that contains this package."""
    return Path(__file__).resolve().parents[3]


def main_checkout(root: Path) -> Path:
    """The main checkout of the repository ``root`` belongs to (no subprocess).

    A ``.git`` directory means ``root`` is the main checkout. A ``.git`` file
    (a linked worktree) names its gitdir, whose ``commondir`` leads to the
    main checkout's ``.git``.
    """
    git = root / ".git"
    if git.is_dir():
        return root
    if git.is_file():
        gitdir_line = next((ln for ln in git.read_text().splitlines() if ln.startswith("gitdir:")), None)
        if gitdir_line is not None:
            gitdir = Path(gitdir_line.split(":", 1)[1].strip())
            if not gitdir.is_absolute():
                gitdir = root / gitdir
            commondir_file = gitdir / "commondir"
            common = gitdir / commondir_file.read_text().strip() if commondir_file.is_file() else gitdir
            return common.resolve().parent
    raise errors.CpError("PATH_GUARD", f"{root} is not a git checkout")


def main() -> Path:
    """The main checkout of this code."""
    return main_checkout(code_root())


def is_main_checkout(root: Path) -> bool:
    return main_checkout(root) == root


def main_head_branch(main_dir: Path) -> str | None:
    """``"main"`` when the main checkout has branch ``main`` checked out, else ``None``."""
    head = main_dir / ".git" / "HEAD"
    try:
        text = head.read_text().strip()
    except OSError:
        return None
    return "main" if text == "ref: refs/heads/main" else None


def live_db(main_dir: Path) -> Path:
    return main_dir / DB_NAME


def is_live_db_path(p: Path) -> bool:
    """A ``harness_state.db`` sitting at the root of a git checkout."""
    return p.name == DB_NAME and (p.parent / ".git").exists()


def sidecar(db: Path, suffix: str) -> Path:
    """``<db><suffix>``, e.g. ``harness_state.db.migrate.lock``."""
    return db.with_name(db.name + suffix)


def _under_pytest() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


def select_db() -> Path:
    """The DB this process may open, or ``PATH_GUARD``."""
    root = code_root()
    env = os.environ.get("QS_CP_DB")
    if env:
        resolved = Path(env).expanduser().resolve()
        if is_live_db_path(resolved):
            if _under_pytest():
                raise errors.CpError("PATH_GUARD", f"{resolved} is a live DB; refused under pytest")
            if not (is_main_checkout(root) and resolved == live_db(root).resolve()):
                raise errors.CpError(
                    "PATH_GUARD",
                    f"{resolved} is a live DB that this copy of the code ({root}) may not open",
                    hint=_HINT,
                )
        return resolved
    if not is_main_checkout(root):
        raise errors.CpError("PATH_GUARD", f"this code ({root}) is not the main checkout's", hint=_HINT)
    if _under_pytest():
        raise errors.CpError("PATH_GUARD", "the live DB is refused under pytest; set QS_CP_DB")
    return live_db(root)


_HINT = "call <MAIN>/scripts/qs/cp.py, or set QS_CP_DB to a temporary DB"


def _linked_worktrees(main_dir: Path) -> list[Path]:
    """The worktrees registered under ``<main>/.git/worktrees/*/gitdir`` (read without a subprocess)."""
    found: list[Path] = []
    registry = main_dir / ".git" / "worktrees"
    for entry in sorted(registry.glob("*/gitdir")):
        try:
            found.append(Path(entry.read_text().strip()).parent)
        except OSError:
            continue
    return found


def state_dir(env: str, default: Path) -> Path:
    """``$env``, else ``default``: a state directory outside every checkout of this repository (QS-406 D13).

    The code root, the main checkout and every linked worktree are refused (``POLICY_REFUSED``, naming
    ``env``). Other ``.git`` ancestors, such as a dotfiles home, are not.
    """
    raw = os.environ.get(env)
    target = (Path(raw) if raw else default).expanduser().resolve()
    root = code_root()
    main_dir = main_checkout(root)
    for forbidden in (root, main_dir, *_linked_worktrees(main_dir)):
        checkout = forbidden.resolve()
        if target == checkout or target.is_relative_to(checkout):
            raise errors.CpError(
                "POLICY_REFUSED",
                f"state dir {target} is inside the checkout {checkout}",
                hint=f"set {env} to a directory outside every checkout",
            )
    return target


def backup_dir() -> Path:
    """Where backups go: ``QS_CP_BACKUP_DIR``, never inside a checkout."""
    return state_dir("QS_CP_BACKUP_DIR", DEFAULT_BACKUP_DIR)


def messenger_dir() -> Path:
    """The watchdog messenger's working directory: ``QS_CP_MESSENGER_DIR``, never inside a checkout."""
    return state_dir("QS_CP_MESSENGER_DIR", DEFAULT_MESSENGER_DIR)


def ensure_private_dir(d: Path) -> Path:
    """``d`` as a 0700 directory owned by this user (QS-406 D13).

    A missing directory is created 0700 (its parents with the default mode); an existing one owned by
    this user is tightened to 0700 (#399 created ``cp-backups`` with the umask); anything else is
    ``POLICY_REFUSED``.
    """
    d.parent.mkdir(parents=True, exist_ok=True)
    try:
        d.mkdir(mode=0o700)
    except FileExistsError:
        pass
    st = os.stat(d)
    if not d.is_dir():
        raise errors.CpError("POLICY_REFUSED", f"{d} is not a directory")
    if st.st_uid != os.getuid():
        raise errors.CpError("POLICY_REFUSED", f"{d} is owned by another user (uid {st.st_uid})")
    if st.st_mode & 0o777 != 0o700:
        os.chmod(d, 0o700)
    return d
