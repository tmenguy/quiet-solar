"""Control Plane errors: one ``CpError(code)`` → an exit code and a JSON error code."""

from __future__ import annotations

from typing import Any

EXIT_CODES: dict[str, int] = {
    "INTERNAL": 1,
    "TOOL_FAILED": 1,
    "USAGE": 2,
    "STALE_TOKEN": 3,
    "STOPPED": 4,
    "SCHEMA_TOO_NEW": 5,
    "SCHEMA_PENDING": 5,
    "BUSY": 6,
    "PATH_GUARD": 7,
    "NOT_FOUND": 8,
    "CONFLICT": 8,
    "INVALID_STATE": 8,
    "POLICY_REFUSED": 9,
}


class CpError(Exception):
    """An expected failure, rendered as ``{"ok": false, "error": code, ...}``."""

    def __init__(self, code: str, detail: str = "", **extra: Any) -> None:
        if code not in EXIT_CODES:
            raise ValueError(f"unknown error code {code!r}")
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail
        self.extra = extra

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.code]

    def payload(self) -> dict[str, Any]:
        return {"ok": False, "error": self.code, "detail": self.detail, **self.extra}
