"""Stable public failures for the optional tokenizer runtime."""

from __future__ import annotations


class BoundaryEngineError(RuntimeError):
    """A stable, content-free runtime rejection."""

    __slots__ = ("code",)

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"token boundary analysis failed: {code}")
