"""Fail-closed CLI for one supplied synthetic pretokenized JSON trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import PurePosixPath
import secrets
import stat
import sys
from typing import Sequence

from .contract import ContractError, MAX_INPUT_BYTES, parse_synthetic_trace
from .topology import TopologyError, audit_trace, canonical_audit_bytes


class IOBoundaryError(RuntimeError):
    """A stable, redacted filesystem-boundary failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _safe_relative_path(value: str, code: str) -> PurePosixPath:
    if (
        not value
        or "\x00" in value
        or "\\" in value
        or value.startswith("/")
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise IOBoundaryError(code)
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts:
        raise IOBoundaryError(code)
    return path


def _open_parent(path: PurePosixPath, code: str) -> tuple[int, str]:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(".", directory_flags)
    except OSError:
        pass
    if descriptor is None:
        raise IOBoundaryError(code)

    traversal_failed = False
    try:
        for part in path.parts[:-1]:
            child = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except OSError:
        traversal_failed = True
    if traversal_failed:
        os.close(descriptor)
        raise IOBoundaryError(code)
    return descriptor, path.parts[-1]


def _read_input(path: PurePosixPath) -> bytes:
    parent_fd, name = _open_parent(path, "io.input_unavailable")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor: int | None = None
        try:
            descriptor = os.open(name, flags, dir_fd=parent_fd)
        except OSError:
            pass
        if descriptor is None:
            raise IOBoundaryError("io.input_unavailable")

        failure = False
        payload: bytes | None = None
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise IOBoundaryError("io.input_not_regular")
            if metadata.st_size > MAX_INPUT_BYTES:
                raise ContractError("input.byte_limit")
            chunks: list[bytes] = []
            remaining = MAX_INPUT_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            if len(payload) > MAX_INPUT_BYTES:
                raise ContractError("input.byte_limit")
            final_metadata = os.fstat(descriptor)
            identity_before = (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
            identity_after = (
                final_metadata.st_dev,
                final_metadata.st_ino,
                final_metadata.st_size,
                final_metadata.st_mtime_ns,
                final_metadata.st_ctime_ns,
            )
            if (
                len(payload) != metadata.st_size
                or identity_after != identity_before
            ):
                raise IOBoundaryError("io.input_changed")
        except OSError:
            failure = True
        finally:
            os.close(descriptor)
        if failure:
            raise IOBoundaryError("io.input_unavailable")
        if payload is None:
            raise IOBoundaryError("io.input_unavailable")
        return payload
    finally:
        os.close(parent_fd)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def _atomic_write(path: PurePosixPath, payload: bytes) -> None:
    parent_fd, name = _open_parent(path, "io.output_parent_unavailable")
    temporary_name: str | None = None
    descriptor: int | None = None
    try:
        stat_failed = False
        try:
            metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            metadata = None
        except OSError:
            metadata = None
            stat_failed = True
        if stat_failed:
            raise IOBoundaryError("io.output_unavailable")
        if metadata is not None and not stat.S_ISREG(metadata.st_mode):
            raise IOBoundaryError("io.output_not_regular")

        for _ in range(16):
            candidate = f".loss-topology-{secrets.token_hex(12)}.tmp"
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                descriptor = os.open(
                    candidate,
                    flags,
                    0o600,
                    dir_fd=parent_fd,
                )
            except FileExistsError:
                continue
            except OSError:
                descriptor = None
                break
            temporary_name = candidate
            break
        if descriptor is None or temporary_name is None:
            raise IOBoundaryError("io.output_unavailable")

        write_failed = False
        try:
            _write_all(descriptor, payload)
            os.fchmod(descriptor, 0o644)
            os.fsync(descriptor)
        except OSError:
            write_failed = True
        finally:
            os.close(descriptor)
            descriptor = None
        if write_failed:
            raise IOBoundaryError("io.output_unavailable")

        replace_failed = False
        try:
            os.replace(
                temporary_name,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            temporary_name = None
            os.fsync(parent_fd)
        except OSError:
            replace_failed = True
        if replace_failed:
            raise IOBoundaryError("io.output_unavailable")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except OSError:
                pass
        os.close(parent_fd)


def _error_payload(code: str) -> bytes:
    return (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "loss-topology.cli-error",
                "status": "error",
                "error": {"code": code},
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _summary_payload(status: str, payload: bytes) -> bytes:
    return (
        json.dumps(
            {
                "output_sha256": hashlib.sha256(payload).hexdigest(),
                "status": status,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


class _RedactedArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise IOBoundaryError("cli.arguments")


def _parser() -> argparse.ArgumentParser:
    parser = _RedactedArgumentParser(
        prog="loss-topology-audit",
        description=(
            "Audit one supplied synthetic pretokenized JSON trace without "
            "loading a tokenizer, model, trainer, or dataset."
        ),
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        input_path = _safe_relative_path(
            arguments.input,
            "io.input_path",
        )
        output_path = _safe_relative_path(
            arguments.output,
            "io.output_path",
        )
        if input_path == output_path:
            raise IOBoundaryError("io.paths_alias")
        payload = _read_input(input_path)
        trace = parse_synthetic_trace(payload)
        report = audit_trace(trace)
        output = canonical_audit_bytes(report)
        _atomic_write(output_path, output)
    except (ContractError, TopologyError, IOBoundaryError) as exc:
        sys.stderr.buffer.write(_error_payload(exc.code))
        return 2
    except Exception:
        sys.stderr.buffer.write(_error_payload("internal.error"))
        return 2

    sys.stdout.buffer.write(_summary_payload(report.status, output))
    return 0 if report.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
