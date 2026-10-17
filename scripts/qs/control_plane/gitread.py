"""``cp.py review read`` (#407 D6a): a review session's only way to read code — git, read-only, fixed argv.

Five ops, each one fixed ``git`` invocation in the main checkout, on hex shas only (the session's prompt
gives every ``base_sha`` / ``head_sha``, so no moving ref is needed):

==========  ==========================================  =====================================================
op          arguments                                   runs
==========  ==========================================  =====================================================
``diff``    ``--base SHA --head SHA [--path P]``        ``git diff --no-ext-diff --no-textconv --no-color``
``show``    ``--rev SHA --path P``                      ``git cat-file blob <rev>:<P>``
``grep``    ``--rev SHA --pattern X [--path P]``        ``git -c grep.patternType=basic grep -n --no-color -e <X> <rev> [-- P]`` (``<X>`` is a BRE whatever the git config)
``ls-tree`` ``--rev SHA [--path P]``                    ``git ls-tree -r --name-only <rev> [-- P]``
``files``   ``--base SHA --head SHA [--path P]``        ``git diff … --name-status <base> <head> [-- P]``
==========  ==========================================  =====================================================

``core.quotePath=false`` stops git from quoting non-ASCII bytes only: a name holding ``"``, ``\\``, a tab or a
newline is still C-quoted, so only an ordinary name printed by ``files`` / ``ls-tree`` is a valid ``--path``
as printed (``grep`` prints ``<rev>:<path>:<line>:``, not a bare path). No other git option can be passed.
Exempt: no DB, no token. The output is decoded with ``surrogateescape``, cut at ``READ_MAX_BYTES`` (backing off
to a UTF-8 boundary), then printed with ``errors="replace"``; ``lossy`` when a kept byte was not UTF-8.
"""

from __future__ import annotations

import codecs
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import errors, ledger, runner

OPS = ("diff", "show", "grep", "ls-tree", "files")
_SUBCOMMAND = {"diff": "diff", "show": "cat-file", "grep": "grep", "ls-tree": "ls-tree", "files": "diff"}
READ_MAX_BYTES = 262144  # 256 KiB
READ_TIMEOUT_S = 60.0
SHA_RE = re.compile(r"[0-9a-f]{7,64}")  # the one sha check (``reviews`` uses it too): SHA-1 or SHA-256
_NEEDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {  # op → (required, optional) arguments
    "diff": (frozenset({"base", "head"}), frozenset({"path"})),
    "show": (frozenset({"rev", "path"}), frozenset()),
    "grep": (frozenset({"rev", "pattern"}), frozenset({"path"})),
    "ls-tree": (frozenset({"rev"}), frozenset({"path"})),
    "files": (frozenset({"base", "head"}), frozenset({"path"})),
}
ENV = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_PAGER": "cat"}
ENV_REMOVED = (  # an inherited repository location would override `-C <main>`
    "GIT_EXTERNAL_DIFF",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_NAMESPACE",
    "GIT_COMMON_DIR",
)


def env_removed(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """``ENV_REMOVED`` plus every other inherited ``GIT_*`` variable, except those ``ENV`` sets (both git paths)."""
    names = os.environ if environ is None else environ
    return (*ENV_REMOVED, *sorted(n for n in names if n.startswith("GIT_") and n not in ENV and n not in ENV_REMOVED))


def _sha(value: str, flag: str) -> str:
    if not SHA_RE.fullmatch(value):
        raise errors.CpError("USAGE", f"--{flag} must be a 7 to 64 character hex sha: {value!r}")
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
    git = ["git", "-C", str(main), "-c", "core.pager=cat", "-c", "diff.external=", "-c", "core.quotePath=false"]
    if op in ("diff", "files"):
        base, head = _sha(given["base"], "base"), _sha(given["head"], "head")
        names = ["--name-status"] if op == "files" else []
        return [*git, "diff", "--no-ext-diff", "--no-textconv", "--no-color", *names, base, head, *tail]
    rev = _sha(given["rev"], "rev")
    if op == "show":
        return [*git, "cat-file", "blob", f"{rev}:{path}"]
    if op == "grep":
        return [*git, "-c", "grep.patternType=basic", "grep", "-n", "--no-color", "-e", given["pattern"], rev, *tail]
    return [*git, "ls-tree", "-r", "--name-only", rev, *tail]


def _cut(raw: bytes) -> bytes:
    """The first ``READ_MAX_BYTES``, never ending inside a multibyte character (the split one is dropped)."""
    kept = raw[:READ_MAX_BYTES]
    if len(raw) > READ_MAX_BYTES:
        decoder = codecs.getincrementaldecoder("utf-8")("surrogateescape")
        decoder.decode(kept, final=False)  # an incomplete last character stays pending
        kept = kept[: len(kept) - len(decoder.getstate()[0])]
    return kept


def _lossy(kept: bytes) -> bool:
    try:
        kept.decode("utf-8")
    except UnicodeDecodeError:
        return True  # a kept byte that was not UTF-8
    return False


def read(run: runner.Runner, main: Path, op: str, given: dict[str, str]) -> dict[str, Any]:
    """Run ``op`` → ``{"output", "truncated", "lossy"}``; a failed git is ``NOT_FOUND`` (128) or ``INTERNAL``."""
    command = argv(main, op, given)
    res = run.run(
        command,
        cwd=main,
        env_extra=ENV,
        env_remove=env_removed(),
        timeout=READ_TIMEOUT_S,
        decode_errors="surrogateescape",
    )
    if res.returncode != 0 and not (op == "grep" and res.returncode == 1):  # grep: 1 is "no match"
        detail = res.stderr.encode("utf-8", "surrogateescape").decode("utf-8", "replace").strip()[-300:]
        code = "NOT_FOUND" if res.returncode == 128 else "INTERNAL"
        raise errors.CpError(code, f"git {_SUBCOMMAND[op]} exited {res.returncode}: {detail}")
    raw = res.stdout.encode("utf-8", "surrogateescape")
    kept = _cut(raw)
    return {
        "output": kept.decode("utf-8", "replace"),
        "truncated": len(raw) > READ_MAX_BYTES,
        "lossy": _lossy(kept),
    }
