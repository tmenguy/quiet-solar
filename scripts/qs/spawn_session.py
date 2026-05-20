#!/usr/bin/env python3
"""Spawn a new interactive OpenCode session via the HTTP API.

The new pipeline activates the next phase's agent and sends a kickoff
prompt in a single synchronous flow by calling::

    POST /session                         # creates a new session
    POST /session/<id>/prompt_async       # binds agent + sends prompt

No ``/instance/reload`` is needed: static ``.opencode/agents/`` files
are discovered at server startup, so we never have to make the server
re-scan agent files mid-flight. Skipping reload eliminates the
``SessionRunState`` cancellation race that blocked the legacy pipeline
from activating the agent in the same request.

CLI fallback semantics (AC #5 / #6 / #7 of QS-177)
--------------------------------------------------

When the OpenCode HTTP API is unreachable — connection refused, DNS
failure, timeout, ``HTTPError``, ``OSError``, ``JSONDecodeError``, or
missing ``id`` in the response — the script probes
``shutil.which('opencode')`` and emits a CLI-form fallback payload so
the user still lands in the next phase. If ``opencode`` is also
missing from PATH, the script exits 2 with ``fallback_unavailable``.

Orphan-session handling: if ``POST /session`` succeeds but the
subsequent ``POST /session/<id>/prompt_async`` fails, the script
attempts a best-effort ``DELETE /session/<id>``. On DELETE success
the user gets the standard ``fallback_cli`` payload (orphan cleaned
up). On DELETE failure the orphan session id is surfaced via
``status: "session_orphaned"`` (exit 2) so the user can clean it up
manually in the OpenCode UI.

- **Agent-file pre-flight** (QS-190): before any HTTP call, ``main()``
  verifies that ``<directory>/.opencode/agents/<agent>.md`` exists. If
  not, the script emits ``{"status": "agent_file_missing", ...}`` on
  stdout, a clear error on stderr, and exits 2 (parallel to the
  ``fallback_unavailable`` branch shape). This closes the QS-177 AC #12
  known limitation where a missing agent file silently lands the
  session on the default OpenCode agent.

Closed limitation (per QS-177 AC #12, closed by QS-190):
spawn_session.py performs a pre-flight check that aborts with
``status: "agent_file_missing"`` before the HTTP API can silently
land a session on the default agent. The pre-flight is documented
in the CLI fallback semantics section above.

Usage::

    python scripts/qs/spawn_session.py \\
        --agent qs-create-plan \\
        --directory /path/to/worktree \\
        [--title "QS_42: Foo"] \\
        [--prompt "Begin your phase protocol."]

The ``--directory`` flag is **required** — it lands in the
``x-opencode-directory`` header so the session is created under the
worktree's OpenCode workspace (otherwise the session is invisible in
the sidebar). The ``--title`` defaults to the agent name; the
``--prompt`` defaults to ``"Begin your phase protocol."``.

There is **no** ``--delay`` flag and **no** ``subprocess`` use —
agent activation happens in-band on ``POST /session/<id>/prompt_async``.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import shlex
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from utils import output_json  # type: ignore[import-not-found]

# Default OpenCode server port. The CLI listens on localhost; the port
# can be overridden via the ``OPENCODE_SERVER_PORT`` env var.
DEFAULT_PORT = 4096

# Default kickoff prompt — matches the convention in the Claude /
# Cursor launchers (``next_prompt or "Begin your phase protocol."``).
DEFAULT_KICKOFF = "Begin your phase protocol."

# Default HTTP timeout for every OpenCode API call (seconds). Matches
# the legacy spawn-session helper (under ``legacy/``).
DEFAULT_TIMEOUT = 10


class SpawnSessionError(RuntimeError):
    """Raised by ``_api`` / ``spawn_session`` on any HTTP/JSON failure.

    Carries the attempted URL and the underlying cause so callers
    (notably ``main()``) can produce a structured fallback payload
    without re-parsing the exception message.

    Attributes:
        url: The URL that failed (or the synthetic URL of the
            response-shape check that failed).
        cause: The underlying exception, if any.
        orphan_session_id: When set, signals that ``POST /session``
            succeeded but a subsequent call failed AND best-effort
            ``DELETE /session/<id>`` also failed — the user needs to
            clean the orphan up manually.
    """

    def __init__(
        self,
        message: str,
        *,
        url: str = "",
        cause: BaseException | None = None,
        orphan_session_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.cause = cause
        self.orphan_session_id = orphan_session_id


def _base_url() -> str:
    """Return ``http://127.0.0.1:<port>`` for the OpenCode server."""
    port = os.environ.get("OPENCODE_SERVER_PORT", str(DEFAULT_PORT))
    return f"http://127.0.0.1:{port}"


def _api(
    method: str,
    path: str,
    body: dict | None = None,
    *,
    directory: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> object:
    """Call the OpenCode HTTP API and return parsed JSON (or ``None``).

    Raises ``SpawnSessionError`` on any failure class (``URLError``,
    ``HTTPError``, ``OSError``, ``TimeoutError``,
    ``HTTPException`` (incl. ``InvalidURL``), ``JSONDecodeError``,
    and ``ValueError`` from ``http.client``'s header validation).
    The original exception is attached via ``__cause__`` so callers
    can inspect it if needed.

    Return type is annotated ``object`` (any JSON value): ``json.loads``
    can return ``dict``, ``list``, ``str``, ``int``, ``float``,
    ``bool``, or ``None``. The OpenCode ``DELETE /session/<id>``
    endpoint returns ``true`` (parsed to Python ``bool``); callers
    discard the DELETE return, so the wider type stays internal but
    is honest about what ``json.loads`` can produce (review fix #03
    should-fix #3).

    Args:
        method: HTTP method (``"POST"``, ``"DELETE"``).
        path: URL path under ``_base_url()`` (e.g. ``"/session"``).
        body: Optional JSON body. ``None`` sends no body.
        directory: Worktree directory; sent as the
            ``x-opencode-directory`` header so the session is bound to
            the correct OpenCode workspace.
        timeout: HTTP timeout in seconds. Defaults to
            ``DEFAULT_TIMEOUT`` (matching the legacy pipeline).

    Returns:
        Parsed JSON of any shape, or ``None`` when the response body
        is empty (e.g. a 204 from ``prompt_async``).
    """
    url = f"{_base_url()}{path}"
    headers: dict[str, str] = {}
    if directory:
        headers["x-opencode-directory"] = directory

    data: bytes | None = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if not raw:
                return None
            return json.loads(raw)
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        OSError,
        http.client.HTTPException,
        ValueError,
    ) as exc:
        # ``OSError`` covers ``socket.timeout`` / ``TimeoutError`` in
        # modern Python; ``URLError`` covers DNS failure, connection
        # refused, and other transport errors; ``HTTPError`` covers
        # non-2xx HTTP status codes; ``http.client.HTTPException``
        # covers ``InvalidURL`` (malformed port, control char in
        # session_id) and other protocol-layer failures that don't
        # subclass any of the above (review fix #01 must-fix #1).
        # ``ValueError`` covers ``http.client``'s header-value
        # validation — raised when CR/LF or other illegal chars
        # appear in the ``x-opencode-directory`` header. The
        # upstream argparse guards reject these too, but the layered
        # defense protects programmer-direct callers (review fix
        # #03 must-fix #1).
        # ``JSONDecodeError`` is a subclass of ``ValueError`` so it's
        # absorbed by this clause too — the message is reformatted
        # the same way; no separate branch needed.
        raise SpawnSessionError(
            f"{method} {path} failed: {exc}", url=url, cause=exc,
        ) from exc


def spawn_session(
    *,
    agent: str,
    directory: str,
    title: str | None = None,
    prompt: str = DEFAULT_KICKOFF,
) -> dict:
    """Create a new OpenCode session and activate ``agent`` with ``prompt``.

    Args:
        agent: Agent name to bind (e.g. ``"qs-create-plan"``).
        directory: Worktree directory; required by the OpenCode server
            via the ``x-opencode-directory`` header.
        title: Optional session title; defaults to ``agent`` when
            ``None``.
        prompt: Kickoff prompt sent inside ``parts[0].text``; defaults
            to ``DEFAULT_KICKOFF``.

    Returns:
        Dict with keys ``session_id``, ``agent``, ``title``,
        ``directory``, ``prompt``, ``status: "session_created"`` on
        success.

    Raises:
        SpawnSessionError: on any HTTP/JSON failure. When
            ``POST /session`` succeeded but a subsequent call failed
            AND ``DELETE /session/<id>`` also failed, the exception's
            ``.orphan_session_id`` carries the leaked session id so
            ``main()`` can surface it to the user.
    """
    # Normalize empty / whitespace ``title`` from programmer-direct
    # callers — the CLI ``main()`` already rejects this via
    # ``parser.error`` (fix plan #01 nice-to-have #23), but the
    # public ``spawn_session`` function accepts an explicit
    # ``title=""`` and would otherwise send ``{"title": ""}`` to the
    # server. Falling back to the agent name (the same default as
    # ``title=None``) keeps both call paths consistent (review fix
    # #03 nice-to-have #21).
    if title is not None and not title.strip():
        title = None
    actual_title = title if title is not None else agent

    # 1. Create session
    session = _api(
        "POST", "/session", {"title": actual_title}, directory=directory,
    )
    if not isinstance(session, dict):
        raise SpawnSessionError(
            f"Unexpected /session response (not a dict): {session!r}",
            url=f"{_base_url()}/session",
        )
    # The verified OpenCode API contract returns the session id under
    # the lowercase ``id`` key only — the legacy ``or session.get('ID')``
    # fallback was YAGNI defensive code with no actual server support
    # (review fix #01 nice-to-have #18).
    session_id = session.get("id")
    # Reject not just ``None`` but any non-string or whitespace-only
    # value. A server returning ``{"id": ""}``, ``{"id": 0}``,
    # ``{"id": False}``, or ``{"id": ["x"]}`` would otherwise survive
    # ``if session_id is None`` and produce a nonsense URL like
    # ``/session//prompt_async`` or ``/session/False/prompt_async``
    # (review fix #02 must-fix #1).
    if not isinstance(session_id, str) or not session_id.strip():
        raise SpawnSessionError(
            f"server response missing or invalid 'id' field: {session!r}",
            url=f"{_base_url()}/session",
        )
    session_id_str = session_id

    # Reject ``..`` substrings in the id even though
    # ``urllib.parse.quote(safe="")`` escapes ``/`` to ``%2F``.
    # ``..`` is in the RFC 3986 unreserved set ``[A-Za-z0-9_.-~]``
    # and survives quoting untouched — a URL-normalising middleware
    # (corporate proxy, reverse proxy, container ingress) could
    # still rewrite ``/session/abc/../delete-me/prompt_async`` to
    # ``/session/delete-me/prompt_async`` before reaching the
    # OpenCode server. We reject ``..`` substrings uniformly (incl.
    # interior patterns like ``abc..def``) for a uniform contract
    # rather than parsing path segments — review fix #03
    # should-fix #4.
    if ".." in session_id_str:
        raise SpawnSessionError(
            f"server returned an id with traversal segments: "
            f"{session_id_str!r}",
            url=f"{_base_url()}/session",
        )

    # URL-escape the session id before path interpolation so a hostile
    # / buggy server response (e.g. ``{"id": "abc/../delete-me"}``)
    # cannot cause path traversal or other URL-shape attacks against
    # the OpenCode API. Review fix #01 should-fix #7.
    safe_id = urllib.parse.quote(session_id_str, safe="")

    # 2. Send prompt + activate agent (in-band, single call)
    try:
        _api(
            "POST",
            f"/session/{safe_id}/prompt_async",
            {"agent": agent, "parts": [{"type": "text", "text": prompt}]},
            directory=directory,
        )
    except SpawnSessionError as prompt_exc:
        # Best-effort cleanup so we don't leak an orphan session in
        # the OpenCode UI. If DELETE also fails, surface the orphan
        # id via ``orphan_session_id`` so main() can emit the
        # ``session_orphaned`` payload (AC #6).
        delete_path = f"/session/{safe_id}"
        try:
            _api("DELETE", delete_path, directory=directory)
        except SpawnSessionError as delete_exc:
            combined = (
                f"prompt_async failed: {prompt_exc}; "
                f"DELETE {delete_path} also failed: {delete_exc}"
            )
            raise SpawnSessionError(
                combined,
                url=f"{_base_url()}{delete_path}",
                cause=prompt_exc,
                orphan_session_id=session_id_str,
            ) from prompt_exc
        # DELETE succeeded — re-raise the original prompt_async error
        # so main() falls back to the CLI form (the user still needs
        # to land in the next phase).
        raise

    return {
        "session_id": session_id_str,
        "agent": agent,
        "title": actual_title,
        "directory": directory,
        "prompt": prompt,
        "status": "session_created",
    }


def _build_cli_command(agent: str, directory: str, prompt: str) -> str:
    """Return the bare ``opencode <dir> --agent <a> --prompt <p>`` fallback command.

    Every interpolated value is shell-escaped via ``shlex.quote``
    (parallel to the legacy launch-command helper under ``legacy/``).
    """
    return (
        f"opencode {shlex.quote(directory)} "
        f"--agent {shlex.quote(agent)} "
        f"--prompt {shlex.quote(prompt)}"
    )


def _emit(payload: dict, *, stderr_msg: str, exit_code: int) -> int:
    """Helper for ``_emit_fallback``: write stderr + JSON, return exit code.

    Eliminates the triplicate ``print(...) + output_json(...) +
    sys.exit(N)`` pattern in ``_emit_fallback``'s three branches —
    a single typo in one branch's stderr message can't drift away
    from the others (review fix #02 should-fix #5).
    """
    print(stderr_msg, file=sys.stderr)
    output_json(payload)
    return exit_code


def _emit_fallback(
    *,
    agent: str,
    directory: str,
    prompt: str,
    exc: SpawnSessionError,
) -> int:
    """Print the appropriate fallback payload + return the desired exit code.

    Decision tree:

    - Orphan (``exc.orphan_session_id`` set) AND ``opencode`` on PATH
      → ``session_orphaned`` with ``new_context_cli``, exit 2.
    - Orphan AND ``opencode`` missing → ``session_orphaned`` WITHOUT
      ``new_context_cli`` (we have no working recovery path; the
      ``detail`` field calls out the missing CLI), exit 2.
    - ``opencode`` on PATH → ``fallback_cli`` with ``new_context_cli``,
      exit 0.
    - ``opencode`` missing → ``fallback_unavailable`` (no
      ``new_context_cli``, recovery hint appended to ``detail``),
      exit 2.

    Detail-shape asymmetry (review fix #01 nice-to-have #19): the
    ``fallback_cli`` and ``session_orphaned`` (with CLI) payloads
    carry the bare ``str(exc)`` in ``detail`` — the recovery path
    lives in ``new_context_cli``. The ``fallback_unavailable`` and
    ``session_orphaned`` (without CLI) payloads append the
    "opencode CLI not on PATH" recovery hint directly to ``detail``
    because they have no ``new_context_cli`` key. This is deliberate
    — payloads that surface a CLI fallback keep the bare error
    visible; payloads without a CLI fallback fold the recovery hint
    into ``detail`` so the user sees it in one place.
    """
    detail = str(exc)
    has_cli = bool(shutil.which("opencode"))

    # Truthy check (review fix #03 nice-to-have #22) — treats an
    # explicit empty-string ``orphan_session_id=""`` as "no orphan"
    # so a future fixture / refactor that synthesizes that value
    # can't accidentally surface a JSON payload with
    # ``session_id: ""``. Upstream code already guarantees non-empty.
    if exc.orphan_session_id:
        orphan_payload: dict = {
            "status": "session_orphaned",
            "session_id": exc.orphan_session_id,
            "agent": agent,
            "directory": directory,
            "detail": detail,
        }
        orphan_stderr = (
            f"ERROR: OpenCode session {exc.orphan_session_id} was created "
            f"but agent activation failed and cleanup also failed; "
            f"clean up the orphan session manually in the OpenCode UI."
        )
        if has_cli:
            # Lazy build — only when actually emitted (review fix
            # #03 nice-to-have #15).
            orphan_payload["new_context_cli"] = _build_cli_command(
                agent, directory, prompt,
            )
        else:
            # No CLI to recover with — fold the recovery hint into
            # ``detail`` (review fix #01 nice-to-have #22) and
            # extend the stderr line so the user-visible first line
            # also surfaces the missing-CLI context (review fix #03
            # should-fix #5).
            orphan_payload["detail"] = (
                f"{detail}; opencode CLI also not on PATH — clean up "
                f"session {exc.orphan_session_id} manually in the OpenCode UI"
            )
            orphan_stderr += " (opencode CLI also not on PATH)"
        return _emit(orphan_payload, stderr_msg=orphan_stderr, exit_code=2)

    if has_cli:
        return _emit(
            {
                "status": "fallback_cli",
                "agent": agent,
                "directory": directory,
                "new_context_cli": _build_cli_command(agent, directory, prompt),
                "detail": detail,
            },
            stderr_msg=(
                f"WARNING: OpenCode HTTP API unreachable ({detail}); "
                f"falling back to opencode CLI."
            ),
            exit_code=0,
        )

    return _emit(
        {
            "status": "fallback_unavailable",
            "agent": agent,
            "directory": directory,
            "detail": (
                f"{detail}; opencode CLI not on PATH — install or activate "
                f"the server manually"
            ),
        },
        stderr_msg=(
            f"ERROR: OpenCode HTTP API unreachable AND opencode CLI not on "
            f"PATH ({detail}); cannot launch the next phase."
        ),
        exit_code=2,
    )


def _agent_file_path(directory: str, agent: str) -> Path:
    """Return the expected ``.opencode/agents/<agent>.md`` path under ``directory``.

    Centralises the convention so the pre-flight guard in ``main()``
    and any future caller (e.g. the OpenCode launcher's CLI-form
    branch in ``launchers/opencode.py``) compute the same path.
    """
    return Path(directory) / ".opencode" / "agents" / f"{agent}.md"


def _reject_control_chars(
    parser: argparse.ArgumentParser, name: str, value: str,
) -> None:
    """Reject ASCII control chars (< 0x20) except ``\\t`` in ``value``.

    The CLI value reaches the OpenCode server via either a header
    (``x-opencode-directory``) or a JSON body field
    (``prompt``/``title``/``agent``). ``http.client`` rejects CR/LF
    in header values with ``ValueError`` — the widened ``_api``
    except routes those through the fallback, but an upstream
    ``parser.error`` gives a clearer message + exit code 2 to the
    user. Layered defense — review fix #03 must-fix #1.

    Tab (``\\t``) is the documented exception: it's whitespace that
    survives the empty/strip guards by design (an intentional value
    is the user's call).
    """
    for ch in value:
        if ord(ch) < 0x20 and ch != "\t":
            parser.error(
                f"{name} must not contain control characters "
                f"(found 0x{ord(ch):02x} at position {value.index(ch)})",
            )


def main() -> None:
    """CLI entry point — see module docstring for usage."""
    parser = argparse.ArgumentParser(
        description=(
            "Spawn a new interactive OpenCode session via the HTTP API "
            "and activate `--agent` with `--prompt` in a single call."
        ),
    )
    parser.add_argument(
        "--agent",
        required=True,
        help="Agent name to bind to the session (e.g. `qs-create-plan`).",
    )
    parser.add_argument(
        "--directory",
        required=True,
        help=(
            "Worktree directory; lands in the `x-opencode-directory` "
            "header so the session is created under the correct "
            "OpenCode workspace."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Session title (defaults to the `--agent` value).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_KICKOFF,
        help=(
            "Kickoff prompt sent inside `parts[0].text` "
            f"(defaults to {DEFAULT_KICKOFF!r})."
        ),
    )
    args = parser.parse_args()

    # Validate-then-strip: the empty-check uses ``.strip()`` (rejects
    # ``"   "``), then we strip-back to canonicalize. **Do not
    # reorder** — flipping the strip-back before the empty-check
    # would silently mask the whitespace-only case (review fix #03
    # nice-to-have #16).
    #
    # Reject empty / whitespace-only values upfront so we don't silently
    # drop the `x-opencode-directory` header (review fix #01 should-fix
    # #8) or send an empty prompt body that the server would 422 on
    # with an opaque error (review fix #01 nice-to-have #23). ``--agent``
    # added for parity in review fix #02 must-fix #2.
    if not args.agent.strip():
        parser.error("--agent must be a non-empty string")
    if not args.directory.strip():
        parser.error("--directory must be a non-empty path")
    if not args.prompt.strip():
        parser.error("--prompt must be a non-empty string")
    # ``--title`` is optional — omitting it defaults to the agent name.
    # An explicitly-empty value is rejected for the same reason as
    # ``--prompt`` (an empty session title surfaces as a blank entry
    # in the OpenCode sidebar).
    if args.title is not None and not args.title.strip():
        parser.error("--title must be a non-empty string when provided")

    # Reject control characters (CR/LF/NUL/other ASCII < 0x20 except
    # tab) in any value that reaches the server — these would cause
    # ``http.client`` to raise ``ValueError`` from inside ``urlopen``
    # for the header path, or land verbatim in the JSON body for
    # other fields (review fix #03 must-fix #1). The widened ``_api``
    # except catches the header case as a safety net; this guard
    # gives the user a clearer message + exit code 2.
    #
    # **Check the pre-strip value** so a trailing ``\n`` / ``\r``
    # isn't silently swallowed by ``strip()`` below (``\n``.strip())
    # = ``""`` so the empty-check already caught BARE ``\n``, but
    # ``"qs\n".strip()`` = ``"qs"`` — trailing-newline values pass
    # the empty-check and the strip-back, and would otherwise reach
    # the server. Order matters here.
    _reject_control_chars(parser, "--agent", args.agent)
    _reject_control_chars(parser, "--directory", args.directory)
    _reject_control_chars(parser, "--prompt", args.prompt)
    if args.title is not None:
        _reject_control_chars(parser, "--title", args.title)

    # Strip surrounding whitespace from the validated values so a
    # shell-history copy-paste artifact (e.g. ``--directory '  /tmp/wt
    # '``) doesn't break the worktree-to-server binding silently
    # (review fix #02 should-fix #8).
    args.agent = args.agent.strip()
    args.directory = args.directory.strip()
    args.prompt = args.prompt.strip()
    if args.title is not None:
        args.title = args.title.strip()

    # Pre-flight: verify the target agent file exists in the worktree.
    # Without this, the OpenCode HTTP API silently lands the session on the
    # default agent if .opencode/agents/<agent>.md is missing — the
    # QS-177 AC #12 known limitation. Closes that gap (QS-190 AC-2).
    expected = _agent_file_path(args.directory, args.agent)
    if not expected.is_file():
        detail = (
            f"ERROR: OpenCode agent file not found at {expected}; "
            f"the HTTP API would silently fall back to the default agent. "
            f"Mirror .claude/agents/{args.agent}.md into .opencode/agents/ "
            f"or create the file before retrying."
        )
        sys.exit(_emit(
            {
                "status": "agent_file_missing",
                "agent": args.agent,
                "directory": args.directory,
                "expected_path": str(expected),
                "detail": detail,
            },
            stderr_msg=detail,
            exit_code=2,
        ))

    try:
        result = spawn_session(
            agent=args.agent,
            directory=args.directory,
            title=args.title,
            prompt=args.prompt,
        )
    except SpawnSessionError as exc:
        sys.exit(_emit_fallback(
            agent=args.agent,
            directory=args.directory,
            prompt=args.prompt,
            exc=exc,
        ))

    output_json(result)
    sys.exit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
