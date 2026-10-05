"""The subprocess seam: every external effect goes through a ``Runner``."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class Runner:
    """Runs a command with ``env_extra`` added and ``env_remove`` removed.

    Its children stay in the caller's process group, except with
    ``detach=True`` (``start_new_session``), used for every ``claude --bg``
    launch so no long-lived descendant keeps a tool's claim alive.
    """

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | str | None = None,
        env_extra: Mapping[str, str] | None = None,
        env_remove: Iterable[str] = (),
        timeout: float | None = None,
        detach: bool = False,
    ) -> RunResult:
        env = dict(os.environ)
        env.update(env_extra or {})
        for name in env_remove:
            env.pop(name, None)
        try:
            proc = subprocess.run(
                list(argv),
                cwd=cwd,
                env=env,
                text=True,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                timeout=timeout,
                start_new_session=detach,
                check=False,
            )
        except OSError as exc:
            return RunResult(127, "", str(exc))
        except subprocess.TimeoutExpired as exc:
            return RunResult(124, _text(exc.stdout), f"timeout after {timeout}s")
        return RunResult(proc.returncode, proc.stdout, proc.stderr)


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value
