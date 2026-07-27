"""Stdin-only command line boundary for one synthetic differential case."""

from __future__ import annotations

import json
import os
import select
import stat
import sys
import time
from collections.abc import Sequence
from typing import Final

from . import analyze_boundary
from .contract import (
    MAX_DOCUMENT_BYTES,
    BoundaryContractError,
    parse_boundary_case,
)
from .errors import BoundaryEngineError
from .report import BoundaryReportError, canonical_boundary_report_bytes

ERROR_SCHEMA_VERSION: Final = "token-boundary-cli-error-v1"
ERROR_KIND: Final = "token-boundary.cli-error"
READ_CHUNK_BYTES: Final = 16 * 1024
READ_TIMEOUT_SECONDS: Final = 5.0
STDIN_DESCRIPTOR: Final = 0
STDOUT_DESCRIPTOR: Final = 1
STDERR_DESCRIPTOR: Final = 2
CLI_ERROR_CODES: Final = frozenset(
    {
        "cli.arguments",
        "cli.interrupted",
        "io.stderr_unavailable",
        "io.stdin_shape",
        "io.stdin_timeout",
        "io.stdin_tty",
        "io.stdin_unavailable",
        "io.stdout_unavailable",
    }
)
CONTRACT_ERROR_CODES: Final = frozenset(
    {
        "case.type",
        "case_id",
        "document.depth",
        "document.fields",
        "document.json",
        "document.nodes",
        "document.root",
        "document.size",
        "document.type",
        "document.utf8",
        "max_length.range",
        "max_length.type",
        "schema_version",
        "text.size",
        "text.unicode",
        "text.value",
    }
)
ENGINE_ERROR_CODES: Final = frozenset(
    {
        "artifact.document",
        "artifact.fields",
        "artifact.identity",
        "artifact.model",
        "artifact.normalizer",
        "artifact.pipeline",
        "artifact.processor",
        "artifact.verification",
        "case.type",
        "directory.close",
        "directory.shape",
        "directory.value",
        "file.close",
        "file.io",
        "file.read",
        "file.shape",
        "file.size",
        "manifest.document",
        "manifest.identity",
        "manifest.semantics",
        "provenance.boundary.range",
        "provenance.boundary.type",
        "provenance.text.size",
        "provenance.text.type",
        "provenance.text.value",
        "provenance.trace",
        "report.invariant",
        "runtime.artifact_load",
        "runtime.encode",
        "runtime.encoding",
        "runtime.normalizer",
        "runtime.target_count",
        "runtime.truncation",
        "runtime.unavailable",
        "runtime.version",
    }
)


class BoundaryCLIError(RuntimeError):
    """A stable, content-free process-boundary rejection."""

    __slots__ = ("code",)

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"token boundary CLI failed: {code}")


def _error_payload(code: str) -> bytes:
    return (
        json.dumps(
            {
                "error": {"code": code},
                "kind": ERROR_KIND,
                "schema_version": ERROR_SCHEMA_VERSION,
                "status": "error",
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _stdin_is_fifo(descriptor: int) -> bool:
    probe_failed = False
    try:
        interactive = os.isatty(descriptor)
    except OSError:
        probe_failed = True
        interactive = False
    if probe_failed:
        raise BoundaryCLIError("io.stdin_unavailable")
    if interactive:
        raise BoundaryCLIError("io.stdin_tty")

    stat_failed = False
    try:
        metadata = os.fstat(descriptor)
    except OSError:
        stat_failed = True
        metadata = None
    if stat_failed or metadata is None:
        raise BoundaryCLIError("io.stdin_unavailable")
    if stat.S_ISREG(metadata.st_mode):
        return False
    if stat.S_ISFIFO(metadata.st_mode):
        return True
    raise BoundaryCLIError("io.stdin_shape")


def _wait_for_stdin(descriptor: int, deadline: float) -> None:
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BoundaryCLIError("io.stdin_timeout")
        wait_failed = False
        try:
            readable, _, _ = select.select(
                (descriptor,),
                (),
                (),
                remaining,
            )
        except InterruptedError:
            continue
        except (OSError, ValueError):
            wait_failed = True
            readable = []
        if wait_failed:
            raise BoundaryCLIError("io.stdin_unavailable")
        if readable:
            return
        raise BoundaryCLIError("io.stdin_timeout")


def _read_stdin(descriptor: int = STDIN_DESCRIPTOR) -> bytes:
    is_fifo = _stdin_is_fifo(descriptor)
    deadline = time.monotonic() + READ_TIMEOUT_SECONDS if is_fifo else None
    chunks: list[bytes] = []
    remaining = MAX_DOCUMENT_BYTES + 1
    while remaining:
        if deadline is not None:
            _wait_for_stdin(descriptor, deadline)
        read_failed = False
        interrupted = False
        try:
            chunk = os.read(
                descriptor,
                min(READ_CHUNK_BYTES, remaining),
            )
        except InterruptedError:
            interrupted = True
            chunk = b""
        except OSError:
            read_failed = True
            chunk = b""
        if interrupted:
            continue
        if read_failed or not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    if read_failed:
        raise BoundaryCLIError("io.stdin_unavailable")
    return b"".join(chunks)


def _write_all(descriptor: int, payload: bytes, code: str) -> None:
    view = memoryview(payload)
    write_failed = False
    while view:
        try:
            written = os.write(descriptor, view)
        except OSError:
            write_failed = True
            written = 0
        if write_failed or written <= 0:
            break
        view = view[written:]
    if write_failed or view:
        raise BoundaryCLIError(code)


def _emit_error(code: str) -> None:
    try:
        _write_all(
            STDERR_DESCRIPTOR,
            _error_payload(code),
            "io.stderr_unavailable",
        )
    except (BoundaryCLIError, KeyboardInterrupt):
        pass
    # There is no safer fallback channel after stderr itself fails.
    except Exception:  # noqa: BLE001, S110
        pass


def _allowlisted(code: object, allowed: frozenset[str]) -> str:
    if type(code) is str and code in allowed:
        assert isinstance(code, str)
        return code
    return "internal.error"


def main(argv: Sequence[str] | None = None) -> int:
    """Read one bounded case from stdin and write one canonical report."""

    error_code: str | None = None
    output: bytes | None = None
    status: str | None = None
    try:
        arguments = tuple(sys.argv[1:] if argv is None else argv)
        if arguments:
            raise BoundaryCLIError("cli.arguments")
        payload = _read_stdin()
        case = parse_boundary_case(payload)
        report = analyze_boundary(case)
        output = canonical_boundary_report_bytes(report)
        status = report.status
    except BoundaryCLIError as error:
        error_code = _allowlisted(error.code, CLI_ERROR_CODES)
    except BoundaryContractError as error:
        error_code = _allowlisted(error.code, CONTRACT_ERROR_CODES)
    except BoundaryEngineError as error:
        error_code = _allowlisted(error.code, ENGINE_ERROR_CODES)
    except BoundaryReportError:
        error_code = "report.invariant"
    except KeyboardInterrupt:
        error_code = "cli.interrupted"
    # The process boundary must never serialize an unexpected exception.
    except Exception:  # noqa: BLE001
        error_code = "internal.error"

    if error_code is not None:
        _emit_error(error_code)
        return 2
    if output is None or status is None:
        _emit_error("internal.error")
        return 2

    output_failed = False
    try:
        _write_all(
            STDOUT_DESCRIPTOR,
            output,
            "io.stdout_unavailable",
        )
    except BoundaryCLIError:
        output_failed = True
    except KeyboardInterrupt:
        error_code = "cli.interrupted"
    # Preserve the same redacted boundary during transport delivery.
    except Exception:  # noqa: BLE001
        error_code = "internal.error"
    if output_failed:
        _emit_error("io.stdout_unavailable")
        return 2
    if error_code is not None:
        _emit_error(error_code)
        return 2
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
