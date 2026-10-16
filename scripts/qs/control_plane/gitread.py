"""``cp.py review read`` (#407 D6a): a review session's only way to read code — git, read-only, fixed argv.

Four ops, each one fixed ``git`` invocation in the main checkout, on hex shas only (the session's prompt
gives every ``base_sha`` / ``head_sha``, so no moving ref is needed):

==========  ==========================================  =====================================================
op          arguments                                   runs
==========  ==========================================  =====================================================
``diff``    ``--base SHA --head SHA [--path P]``        ``git diff --no-ext-diff --no-textconv --no-color``
``show``    ``--rev SHA --path P``                      ``git cat-file blob <rev>:<P>``
``grep``    ``--rev SHA --pattern X [--path P]``        ``git grep -n --no-color -e <X> <rev> [-- P]``
``ls-tree`` ``--rev SHA [--path P]``                    ``git ls-tree -r --name-only <rev> [-- P]``
==========  ==========================================  =====================================================

No other git option can be passed. Exempt: no DB, no token. The output is decoded with
``surrogateescape`` (``lossy`` when a byte was not UTF-8), cut at ``READ_MAX_BYTES``, then printed with
``errors="replace"``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import errors, ledger, runner

OPS = ("diff", "show", "grep", "ls-tree")
READ_MAX_BYTES = 262144  # 256 KiB
READ_TIMEOUT_S = 60.0
_SHA = re.compile(r"[0-9a-f]{7,40}")
_NEEDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {  # op → (required, optional) arguments
    "diff": (frozenset({"base", "head"}), frozenset({"path"})),
    "show": (frozenset({"rev", "path"}), frozenset()),
    "grep": (frozenset({"rev", "pattern"}), frozenset({"path"})),
    "ls-tree": (frozenset({"rev"}), frozenset({"path"})),
}
ENV = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_PAGER": "cat"}
ENV_REMOVED = ("GIT_EXTERNAL_DIFF",)


def _sha(value: str, flag: str) -> str:
    if not _SHA.fullmatch(value):
        raise errors.CpError("USAGE", f"--{flag} must be a 7 to 40 character hex sha: {value!r}")
    return value


def argv(main: Path, op: str, given: dict[str, str]) -> list[str]:
    """The fixed argv of ``op`` for the given arguments (every refusal is ``USAGE``)."""
    if op not in _NEEDS:
        raise errors.CpError("USAGE", f"op must be one of {', '.join(OPS)}: {op!r}")
    required, optional = _NEEDS[op]
    missing = sorted(required - set(given))
    extra = sorted(set(given) - required - optional)
    if missing or extra:
        names = ", ".join(f"--{n}" for n in missing or extra)
        raise errors.CpError("USAGE", f"review read {op}: {'missing' if missing else 'not taken'} {names}")
    path = ledger.norm_path(given["path"]) if "path" in given else None
    if "path" in required and path is None:
        raise errors.CpError("USAGE", f"review read {op}: --path must name a file")
    tail = [] if path is None else ["--", path]
    git = ["git", "-C", str(main), "-c", "core.pager=cat", "-c", "diff.external="]
    if op == "diff":
        base, head = _sha(given["base"], "base"), _sha(given["head"], "head")
        return [*git, "diff", "--no-ext-diff", "--no-textconv", "--no-color", base, head, *tail]
    rev = _sha(given["rev"], "rev")
    if op == "show":
        return [*git, "cat-file", "blob", f"{rev}:{path}"]
    if op == "grep":
        return [*git, "grep", "-n", "--no-color", "-e", given["pattern"], rev, *tail]
    return [*git, "ls-tree", "-r", "--name-only", rev, *tail]


def _lossy(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return True  # a lone surrogate: surrogateescape kept a byte that was not UTF-8
    return False


def read(run: runner.Runner, main: Path, op: str, given: dict[str, str]) -> dict[str, Any]:
    """Run ``op`` → ``{"output", "truncated", "lossy"}``; a failed git is ``NOT_FOUND`` (128) or ``INTERNAL``."""
    command = argv(main, op, given)
    res = run.run(
        command,
        cwd=main,
        env_extra=ENV,
        env_remove=ENV_REMOVED,
        timeout=READ_TIMEOUT_S,
        decode_errors="surrogateescape",
    )
    if res.returncode != 0 and not (op == "grep" and res.returncode == 1):  # grep: 1 is "no match"
        detail = res.stderr.encode("utf-8", "surrogateescape").decode("utf-8", "replace").strip()[-300:]
        code = "NOT_FOUND" if res.returncode == 128 else "INTERNAL"
        raise errors.CpError(code, f"git {command[7]} exited {res.returncode}: {detail}")
    raw = res.stdout.encode("utf-8", "surrogateescape")
    return {
        "output": raw[:READ_MAX_BYTES].decode("utf-8", "replace"),
        "truncated": len(raw) > READ_MAX_BYTES,
        "lossy": _lossy(res.stdout),
    }
