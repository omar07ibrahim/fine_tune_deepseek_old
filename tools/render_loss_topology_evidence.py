"""Generate deterministic, source-bound visual evidence for LossTopology Lab.

The generator is intentionally closed over two committed synthetic fixtures
and the standard-library-only ``loss_topology`` package. It never imports or
executes the inherited trainer and has no model, tokenizer, dataset, network,
or GPU path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import html
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from loss_topology import (
    ASSISTANT_ONLY,
    IGNORE_INDEX,
    PolicyAudit,
    SyntheticTrace,
    audit_label_topology,
    audit_trace,
    build_labels,
    canonical_audit_bytes,
    parse_synthetic_trace,
)


OUTPUT_PARTS = ("docs", "evidence", "generated")
MANIFEST_NAME = "loss-topology-evidence.v1.json"
HEALTHY_NAME = "healthy.audit.json"
EMPTY_NAME = "empty-assistant.audit.json"
TRANSCRIPT_NAME = "cli-session.txt"
TRANSCRIPT_SVG_NAME = "cli-session.svg"
POLICY_SVG_NAME = "policy-topology.svg"
FAULT_SVG_NAME = "assistant-only-fault-diagnostics.svg"
ARTIFACT_NAMES = (
    FAULT_SVG_NAME,
    TRANSCRIPT_SVG_NAME,
    TRANSCRIPT_NAME,
    EMPTY_NAME,
    HEALTHY_NAME,
    POLICY_SVG_NAME,
)
PUBLICATION_ORDER = (*ARTIFACT_NAMES, MANIFEST_NAME)
FIXTURE_PATHS = (
    "fixtures/synthetic/healthy.v1.json",
    "fixtures/synthetic/empty-assistant.v1.json",
)
EXPECTED_FIXTURE_SHA256 = {
    "fixtures/synthetic/empty-assistant.v1.json": (
        "9a8bcfc90dae4ff832f18f139dd2b2cdba86ad71133964e5a0c202ba8fda0013"
    ),
    "fixtures/synthetic/healthy.v1.json": (
        "f4108dc2f0db02b1e5cfd13179b598b41242855d779f3276310cb953be664791"
    ),
}
SOURCE_PATHS = (
    "fixtures/synthetic/empty-assistant.v1.json",
    "fixtures/synthetic/healthy.v1.json",
    "loss_topology/__init__.py",
    "loss_topology/cli.py",
    "loss_topology/contract.py",
    "loss_topology/topology.py",
    "pyproject.toml",
    "tools/render_loss_topology_evidence.py",
)
MAX_SOURCE_BYTES = 2 * 1024 * 1024
STAGE_PREFIX = ".loss-topology-evidence-stage-"
BACKUP_PREFIX = ".loss-topology-evidence-backup-"
BOUNDARY_KINDS = frozenset(
    {
        "prefix_special",
        "message_start",
        "message_end",
        "suffix_special",
    }
)


class EvidenceError(RuntimeError):
    """Stable, path-redacted evidence generation failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class CLIRun:
    """One actual subprocess invocation and its canonical output."""

    identifier: str
    argv: tuple[str, ...]
    input_path: str
    input_bytes: bytes
    output_bytes: bytes
    stdout: bytes
    stderr: bytes
    returncode: int


@dataclass(frozen=True, slots=True)
class FaultRun:
    """One controlled in-memory mutation audited through the public API."""

    identifier: str
    title: str
    mutations: tuple[tuple[int, str], ...]
    audit: PolicyAudit


@dataclass(frozen=True, slots=True)
class DirectoryHandle:
    """Pinned generated-directory descriptor and its ancestry identities."""

    descriptor: int
    identities: tuple[tuple[int, int], ...]
    created_parts: tuple[str, ...]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _regular_read_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _safe_parts(relative_path: str) -> tuple[str, ...]:
    path = PurePosixPath(relative_path)
    if (
        not relative_path
        or "\x00" in relative_path
        or "\\" in relative_path
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise EvidenceError("path.invalid")
    return path.parts


def _read_all(descriptor: int, expected_size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = expected_size
    while remaining:
        chunk = os.read(descriptor, min(remaining, 64 * 1024))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    payload = b"".join(chunks)
    if len(payload) != expected_size:
        raise EvidenceError("io.source_changed")
    return payload


def _read_repo_file(root_fd: int, relative_path: str) -> bytes:
    parts = _safe_parts(relative_path)
    parent_fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            child_fd: int | None = None
            try:
                child_fd = os.open(part, _directory_flags(), dir_fd=parent_fd)
            except OSError:
                pass
            if child_fd is None:
                raise EvidenceError("io.source_unavailable")
            os.close(parent_fd)
            parent_fd = child_fd

        descriptor: int | None = None
        try:
            descriptor = os.open(
                parts[-1],
                _regular_read_flags(),
                dir_fd=parent_fd,
            )
        except OSError:
            pass
        if descriptor is None:
            raise EvidenceError("io.source_unavailable")
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_size > MAX_SOURCE_BYTES
            ):
                raise EvidenceError("io.source_not_regular")
            payload = _read_all(descriptor, before.st_size)
            after = os.fstat(descriptor)
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise EvidenceError("io.source_changed")
            return payload
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)


def _open_root(repository_root: Path) -> int:
    descriptor: int | None = None
    try:
        descriptor = os.open(repository_root, _directory_flags())
    except OSError:
        pass
    if descriptor is None:
        raise EvidenceError("io.root_unavailable")
    return descriptor


def _open_generated_directory(
    root_fd: int,
    *,
    create: bool,
) -> DirectoryHandle:
    current_fd = os.dup(root_fd)
    identities: list[tuple[int, int]] = []
    created_parts: list[str] = []
    try:
        root_stat = os.fstat(current_fd)
        identities.append((root_stat.st_dev, root_stat.st_ino))
        for part in OUTPUT_PARTS:
            child_fd: int | None = None
            try:
                child_fd = os.open(
                    part,
                    _directory_flags(),
                    dir_fd=current_fd,
                )
            except FileNotFoundError:
                if not create:
                    raise EvidenceError("io.output_directory_missing")
                try:
                    os.mkdir(part, 0o755, dir_fd=current_fd)
                    os.fsync(current_fd)
                    child_fd = os.open(
                        part,
                        _directory_flags(),
                        dir_fd=current_fd,
                    )
                    created_parts.append(part)
                except OSError:
                    child_fd = None
            except OSError:
                pass
            if child_fd is None:
                raise EvidenceError("io.output_directory_unavailable")
            child_stat = os.fstat(child_fd)
            identities.append((child_stat.st_dev, child_stat.st_ino))
            os.close(current_fd)
            current_fd = child_fd
        return DirectoryHandle(
            descriptor=current_fd,
            identities=tuple(identities),
            created_parts=tuple(created_parts),
        )
    except Exception:
        os.close(current_fd)
        raise


def _verify_generated_identity(
    root_fd: int,
    expected: tuple[tuple[int, int], ...],
) -> bool:
    current_fd = os.dup(root_fd)
    try:
        metadata = os.fstat(current_fd)
        observed = [(metadata.st_dev, metadata.st_ino)]
        for part in OUTPUT_PARTS:
            try:
                child_fd = os.open(
                    part,
                    _directory_flags(),
                    dir_fd=current_fd,
                )
            except OSError:
                return False
            os.close(current_fd)
            current_fd = child_fd
            metadata = os.fstat(current_fd)
            observed.append((metadata.st_dev, metadata.st_ino))
        return tuple(observed) == expected
    finally:
        os.close(current_fd)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def _new_temporary_file(
    directory_fd: int,
    prefix: str,
    payload: bytes,
) -> str:
    descriptor: int | None = None
    name: str | None = None
    for _ in range(32):
        candidate = f"{prefix}{secrets.token_hex(12)}"
        try:
            descriptor = os.open(
                candidate,
                (
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                ),
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError:
            continue
        except OSError:
            descriptor = None
            break
        name = candidate
        break
    if descriptor is None or name is None:
        raise EvidenceError("io.stage_unavailable")

    failed = False
    try:
        _write_all(descriptor, payload)
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
    except OSError:
        failed = True
    finally:
        os.close(descriptor)
    if failed:
        try:
            os.unlink(name, dir_fd=directory_fd)
        except OSError:
            pass
        raise EvidenceError("io.stage_write")
    return name


def _replace_entry(source: str, destination: str, directory_fd: int) -> None:
    os.replace(
        source,
        destination,
        src_dir_fd=directory_fd,
        dst_dir_fd=directory_fd,
    )


def _remove_if_present(name: str, directory_fd: int) -> bool:
    try:
        os.unlink(name, dir_fd=directory_fd)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def _read_generated_entry(directory_fd: int, name: str) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            _regular_read_flags(),
            dir_fd=directory_fd,
        )
    except OSError:
        pass
    if descriptor is None:
        raise EvidenceError("io.artifact_unavailable")
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size > MAX_SOURCE_BYTES
        ):
            raise EvidenceError("io.artifact_not_regular")
        payload = _read_all(descriptor, before.st_size)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise EvidenceError("io.artifact_changed")
        return payload
    finally:
        os.close(descriptor)


def _publish_bundle(repository_root: Path, bundle: dict[str, bytes]) -> None:
    if tuple(bundle) != PUBLICATION_ORDER:
        raise EvidenceError("bundle.order")
    root_fd = _open_root(repository_root)
    generated: DirectoryHandle | None = None
    staged: dict[str, str] = {}
    backups: dict[str, str | None] = {}
    destination_identities: dict[
        str,
        tuple[int, int, int, int, int] | None,
    ] = {}
    published: list[str] = []
    preserve_backups = False
    try:
        generated = _open_generated_directory(root_fd, create=True)
        directory_fd = generated.descriptor

        for name in PUBLICATION_ORDER:
            try:
                metadata = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                metadata = None
            except OSError:
                raise EvidenceError("io.destination_unavailable")
            if metadata is not None and not stat.S_ISREG(metadata.st_mode):
                raise EvidenceError("io.destination_not_regular")
            destination_identities[name] = (
                None
                if metadata is None
                else (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                )
            )

            staged[name] = _new_temporary_file(
                directory_fd,
                STAGE_PREFIX,
                bundle[name],
            )
            backups[name] = None
            if metadata is not None:
                existing_payload = _read_generated_entry(
                    directory_fd,
                    name,
                )
                backup_name = _new_temporary_file(
                    directory_fd,
                    BACKUP_PREFIX,
                    existing_payload,
                )
                backups[name] = backup_name
                current_stat = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                current_identity = (
                    current_stat.st_dev,
                    current_stat.st_ino,
                    current_stat.st_size,
                    current_stat.st_mtime_ns,
                    current_stat.st_ctime_ns,
                )
                if current_identity != destination_identities[name]:
                    raise EvidenceError("io.destination_changed")

        os.fsync(directory_fd)
        if not _verify_generated_identity(root_fd, generated.identities):
            raise EvidenceError("io.output_directory_changed")
        for name, expected_identity in destination_identities.items():
            try:
                current_stat = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                current_identity = None
            except OSError:
                raise EvidenceError("io.destination_unavailable")
            else:
                current_identity = (
                    current_stat.st_dev,
                    current_stat.st_ino,
                    current_stat.st_size,
                    current_stat.st_mtime_ns,
                    current_stat.st_ctime_ns,
                )
            if current_identity != expected_identity:
                raise EvidenceError("io.destination_changed")

        try:
            for name in PUBLICATION_ORDER:
                _replace_entry(staged[name], name, directory_fd)
                del staged[name]
                published.append(name)
                os.fsync(directory_fd)
                if not _verify_generated_identity(root_fd, generated.identities):
                    raise OSError("directory identity changed")
            if not _verify_generated_identity(root_fd, generated.identities):
                raise EvidenceError("io.output_directory_changed")
            for name, expected in bundle.items():
                if _read_generated_entry(directory_fd, name) != expected:
                    raise EvidenceError("io.publish_verification")
        except (OSError, EvidenceError) as error:
            rollback_ok = True
            for name in reversed(published):
                backup = backups[name]
                if backup is None:
                    rollback_ok = (
                        _remove_if_present(name, directory_fd) and rollback_ok
                    )
                else:
                    try:
                        _replace_entry(backup, name, directory_fd)
                        backups[name] = None
                    except OSError:
                        rollback_ok = False
            try:
                os.fsync(directory_fd)
            except OSError:
                rollback_ok = False
            if not rollback_ok:
                preserve_backups = True
                raise EvidenceError("io.rollback_failed")
            if isinstance(error, EvidenceError):
                raise error
            raise EvidenceError("io.publish_failed")

        for name in PUBLICATION_ORDER:
            backup = backups[name]
            if backup is not None:
                if not _remove_if_present(backup, directory_fd):
                    raise EvidenceError("io.backup_cleanup")
                backups[name] = None
        os.fsync(directory_fd)
    finally:
        if generated is not None:
            directory_fd = generated.descriptor
            for temporary_name in tuple(staged.values()):
                _remove_if_present(temporary_name, directory_fd)
            if not preserve_backups:
                for backup_name in tuple(backups.values()):
                    if backup_name is not None:
                        _remove_if_present(backup_name, directory_fd)
            os.close(directory_fd)
        os.close(root_fd)


def _check_bundle(repository_root: Path, bundle: dict[str, bytes]) -> None:
    if tuple(bundle) != PUBLICATION_ORDER:
        raise EvidenceError("bundle.order")
    root_fd = _open_root(repository_root)
    generated: DirectoryHandle | None = None
    try:
        generated = _open_generated_directory(root_fd, create=False)
        for name, expected in bundle.items():
            if _read_generated_entry(generated.descriptor, name) != expected:
                raise EvidenceError("bundle.drift")
        entries = set(os.listdir(generated.descriptor))
        if entries != set(PUBLICATION_ORDER):
            raise EvidenceError("bundle.inventory")
        if not _verify_generated_identity(root_fd, generated.identities):
            raise EvidenceError("io.output_directory_changed")
    finally:
        if generated is not None:
            os.close(generated.descriptor)
        os.close(root_fd)


def _expected_report(input_bytes: bytes) -> bytes:
    trace = parse_synthetic_trace(input_bytes)
    return canonical_audit_bytes(audit_trace(trace))


def _run_cli_evidence(
    repository_root: Path,
    fixture_payloads: dict[str, bytes],
) -> tuple[CLIRun, ...]:
    interpreter = shutil.which("python3")
    if interpreter is None:
        raise EvidenceError("runtime.python3_unavailable")
    environment = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": f"{Path(interpreter).parent}:{os.defpath}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(repository_root),
    }
    specifications = (
        (
            "healthy",
            "input/healthy.v1.json",
            "output/healthy.audit.json",
            0,
        ),
        (
            "empty-assistant",
            "input/empty-assistant.v1.json",
            "output/empty-assistant.audit.json",
            1,
        ),
    )
    runs: list[CLIRun] = []
    try:
        temporary = tempfile.TemporaryDirectory(
            prefix=".loss-topology-evidence-run-",
            dir=repository_root,
        )
    except OSError:
        raise EvidenceError("io.sandbox_unavailable")
    with temporary as temporary_name:
        sandbox = Path(temporary_name)
        (sandbox / "input").mkdir(mode=0o755)
        (sandbox / "output").mkdir(mode=0o755)
        for path, payload in fixture_payloads.items():
            target = sandbox / "input" / Path(path).name
            target.write_bytes(payload)

        for identifier, input_path, output_path, expected_exit in specifications:
            argv = (
                "python3",
                "-m",
                "loss_topology.cli",
                "--input",
                input_path,
                "--output",
                output_path,
            )
            try:
                result = subprocess.run(
                    argv,
                    cwd=sandbox,
                    env=environment,
                    capture_output=True,
                    check=False,
                    timeout=10,
                )
            except (OSError, subprocess.SubprocessError):
                raise EvidenceError("runtime.cli_failed")
            if result.returncode != expected_exit or result.stderr != b"":
                raise EvidenceError("runtime.cli_contract")
            output = sandbox / output_path
            if output.is_symlink() or not output.is_file():
                raise EvidenceError("runtime.cli_output")
            output_bytes = output.read_bytes()
            input_bytes = (
                sandbox / input_path
            ).read_bytes()
            expected_output = _expected_report(input_bytes)
            if output_bytes != expected_output:
                raise EvidenceError("runtime.cli_output")
            expected_summary = (
                json.dumps(
                    {
                        "output_sha256": _sha256(output_bytes),
                        "status": json.loads(output_bytes)["status"],
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            if result.stdout != expected_summary:
                raise EvidenceError("runtime.cli_summary")
            runs.append(
                CLIRun(
                    identifier=identifier,
                    argv=argv,
                    input_path=(
                        "fixtures/synthetic/healthy.v1.json"
                        if identifier == "healthy"
                        else "fixtures/synthetic/empty-assistant.v1.json"
                    ),
                    input_bytes=input_bytes,
                    output_bytes=output_bytes,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    returncode=result.returncode,
                )
            )
    return tuple(runs)


def _fault_runs(trace: SyntheticTrace) -> tuple[FaultRun, ...]:
    baseline = list(build_labels(trace, ASSISTANT_ONLY))
    categories = _segment_categories(trace)
    if (
        categories[12] != "boundary"
        or categories[8] != "user"
        or categories[31] != "padding"
        or categories[14] != "assistant"
        or baseline[12] != IGNORE_INDEX
        or baseline[8] != IGNORE_INDEX
        or baseline[31] != IGNORE_INDEX
        or baseline[14] == IGNORE_INDEX
    ):
        raise EvidenceError("fault.fixture_topology")
    specifications = (
        (
            "boundary-leak",
            "Boundary leak @ 12",
            ((12, "supervise"),),
        ),
        (
            "user-padding-leak",
            "User + padding @ 8, 31",
            ((8, "supervise"), (31, "supervise")),
        ),
        (
            "missing-target",
            "Missing target @ 14",
            ((14, "ignore"),),
        ),
    )
    results: list[FaultRun] = []
    for identifier, title, mutations in specifications:
        labels = baseline.copy()
        for position, action in mutations:
            if action == "supervise":
                labels[position] = trace.trace.token_ids[position]
            elif action == "ignore":
                labels[position] = IGNORE_INDEX
            else:
                raise EvidenceError("fault.action")
        audit = audit_label_topology(
            trace,
            tuple(labels),
            ASSISTANT_ONLY,
        )
        results.append(
            FaultRun(
                identifier=identifier,
                title=title,
                mutations=mutations,
                audit=audit,
            )
        )
    return tuple(results)


def _segment_categories(trace: SyntheticTrace) -> tuple[str, ...]:
    categories = ["unknown"] * len(trace.trace.token_ids)
    for segment in trace.trace.segments:
        if segment.kind == "padding":
            category = "padding"
        elif segment.kind in BOUNDARY_KINDS:
            category = "boundary"
        elif segment.message_index is None:
            raise EvidenceError("render.segment")
        else:
            category = trace.messages[segment.message_index].role
        for position in range(segment.start, segment.end):
            categories[position] = category
    if "unknown" in categories:
        raise EvidenceError("render.segment")
    return tuple(categories)


def _svg_document(
    *,
    width: int,
    height: int,
    title: str,
    description: str,
    body: str,
) -> bytes:
    title_text = html.escape(title)
    description_text = html.escape(description)
    document = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">
  <title id="title">{title_text}</title>
  <desc id="description">{description_text}</desc>
  <rect width="{width}" height="{height}" fill="#08111f"/>
  <style>
    .sans {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
    .title {{ fill: #f8fafc; font-size: 34px; font-weight: 760; }}
    .subtitle {{ fill: #cbd5e1; font-size: 17px; }}
    .label {{ fill: #e2e8f0; font-size: 16px; font-weight: 680; }}
    .small {{ fill: #cbd5e1; font-size: 13px; }}
    .tiny {{ fill: #e2e8f0; font-size: 11px; }}
    .scope {{ fill: #dbeafe; font-size: 13px; font-weight: 650; letter-spacing: .35px; }}
  </style>
{body}
</svg>
"""
    return document.encode("utf-8")


def _render_policy_svg(trace: SyntheticTrace, report_bytes: bytes) -> bytes:
    report = json.loads(report_bytes)
    categories = _segment_categories(trace)
    policies = report["policies"]
    token_count = report["trace_summary"]["token_count"]
    width = 1600
    height = 790
    start_x = 300
    cell_width = 36
    cell_height = 44
    role_colors = {
        "boundary": "#7c3aed",
        "system": "#475569",
        "user": "#2563eb",
        "assistant": "#0f766e",
        "padding": "#334155",
    }
    pieces = [
        '  <text x="72" y="72" class="sans title">Synthetic supervision topology by policy</text>',
        f'  <text x="72" y="108" class="sans subtitle">Healthy hand-authored trace · {token_count} token positions · exact generated audit</text>',
        '  <rect x="72" y="132" width="1456" height="44" rx="10" fill="#10233f" stroke="#1d4ed8"/>',
        '  <text x="94" y="159" class="sans scope">NO TOKENIZER · NO CAUSAL SHIFT OR LOSS · NO MODEL / TRAINER / DATASET / GPU</text>',
        '  <text x="72" y="230" class="sans label">Trace ownership</text>',
    ]
    for position, category in enumerate(categories):
        x = start_x + (position * cell_width)
        color = role_colors[category]
        pieces.append(
            f'  <rect x="{x}" y="198" width="{cell_width - 2}" '
            f'height="{cell_height}" rx="4" fill="{color}">'
            f"<title>position {position}: {html.escape(category)}</title></rect>"
        )
        pieces.append(
            f'  <text x="{x + ((cell_width - 2) / 2):.1f}" y="264" '
            f'class="mono tiny" text-anchor="middle">{position}</text>'
        )
    legend_x = 300
    for category in ("system", "user", "assistant", "boundary", "padding"):
        pieces.extend(
            [
                f'  <rect x="{legend_x}" y="286" width="18" height="18" '
                f'rx="4" fill="{role_colors[category]}"/>',
                f'  <text x="{legend_x + 26}" y="300" class="sans small">'
                f"{html.escape(category)}</text>",
            ]
        )
        legend_x += 170

    row_specs = (
        ("all_tokens", 354, "#f59e0b"),
        ("assistant_only", 482, "#22c55e"),
    )
    for policy_name, y, selected_color in row_specs:
        policy = policies[policy_name]
        pieces.append(
            f'  <text x="72" y="{y + 28}" class="mono label">'
            f"{html.escape(policy_name)}</text>"
        )
        selected_positions = {
            position
            for run in policy["supervised_runs"]
            for position in range(run["start"], run["end"])
        }
        for position in range(len(categories)):
            x = start_x + (position * cell_width)
            selected = position in selected_positions
            fill = selected_color if selected else "#25344a"
            stroke = "#f8fafc" if selected else "#475569"
            pieces.append(
                f'  <rect x="{x}" y="{y}" width="{cell_width - 2}" '
                f'height="{cell_height}" rx="4" fill="{fill}" '
                f'stroke="{stroke}" stroke-width="1">'
                f"<title>position {position}: "
                f"{'supervised' if selected else 'ignored'}</title></rect>"
            )
        run_text = ", ".join(
            f"[{run['start']},{run['end']})"
            for run in policy["supervised_runs"]
        )
        run_label = (
            "run" if len(policy["supervised_runs"]) == 1 else "runs"
        )
        summary = (
            f"{policy['supervised_token_count']} supervised · "
            f"{run_label} {run_text or 'none'} · "
            f"{policy['ignored_token_count']} ignored"
        )
        pieces.append(
            f'  <text x="300" y="{y + 76}" class="sans small">'
            f"{html.escape(summary)}</text>"
        )

    pieces.extend(
        [
            '  <rect x="72" y="622" width="700" height="108" rx="14" fill="#111f34" stroke="#f59e0b"/>',
            '  <text x="96" y="656" class="sans label">all_tokens boundary behavior</text>',
            '  <text x="96" y="685" class="sans small">Boundary selection is intentional for this policy: '
            f'{policies["all_tokens"]["boundary_supervised_token_count"]} positions.</text>',
            '  <text x="96" y="710" class="sans small">That selection is not labeled as assistant-only leakage.</text>',
            '  <rect x="796" y="622" width="732" height="108" rx="14" fill="#111f34" stroke="#22c55e"/>',
            '  <text x="820" y="656" class="sans label">assistant_only audit</text>',
            '  <text x="820" y="685" class="sans small">'
            f'{len(policies["assistant_only"]["boundary_leakage_positions"])} boundary · '
            f'{len(policies["assistant_only"]["padding_leakage_positions"])} padding · '
            f'{len(policies["assistant_only"]["off_policy_supervision_positions"])} off-policy · '
            f'{len(policies["assistant_only"]["missing_eligible_positions"])} missing positions</text>',
            '  <text x="820" y="710" class="sans small">Selection topology only; it is not a model-quality claim.</text>',
            '  <text x="72" y="768" class="sans tiny">Source: fixtures/synthetic/healthy.v1.json · canonical input '
            f'{html.escape(report["input_sha256"][:12])}… · report {_sha256(report_bytes)[:12]}…</text>',
        ]
    )
    return _svg_document(
        width=width,
        height=height,
        title="Synthetic supervision topology by policy",
        description=(
            "A source-generated comparison of all-token and assistant-only "
            "label selection over a 33-position hand-authored synthetic trace."
        ),
        body="\n".join(pieces),
    )


def _format_positions(values: Sequence[int]) -> str:
    if not values:
        return "none"
    return ", ".join(str(value) for value in values)


def _format_runs(audit: PolicyAudit) -> str:
    if not audit.supervised_runs:
        return "none"
    return ", ".join(
        f"[{run.start},{run.end})" for run in audit.supervised_runs
    )


def _render_fault_svg(
    trace: SyntheticTrace,
    faults: tuple[FaultRun, ...],
) -> bytes:
    width = 1600
    height = 820
    start_x = 300
    cell_width = 27
    cell_height = 38
    baseline = build_labels(trace, ASSISTANT_ONLY)
    pieces = [
        '  <text x="72" y="70" class="sans title">Assistant-only fault injections and detected positions</text>',
        '  <text x="72" y="106" class="sans subtitle">Executed through public audit_label_topology · controlled in-memory mutations, not CLI inputs or outputs</text>',
        '  <rect x="72" y="130" width="1456" height="44" rx="10" fill="#10233f" stroke="#1d4ed8"/>',
        '  <text x="94" y="157" class="sans scope">SYNTHETIC LABEL MUTATIONS · NO TOKENIZER / LOSS / MODEL / TRAINER / DATASET / GPU</text>',
        '  <text x="72" y="224" class="sans small">Green = valid assistant target · red = injected off-policy supervision · amber outline = omitted eligible target</text>',
    ]
    row_y = (270, 424, 578)
    for fault, y in zip(faults, row_y, strict=True):
        pieces.append(
            f'  <text x="72" y="{y + 24}" class="sans label">'
            f"{html.escape(fault.title)}</text>"
        )
        mutation_map = dict(fault.mutations)
        for position, label in enumerate(fault.audit.labels):
            x = start_x + (position * cell_width)
            originally_selected = baseline[position] != IGNORE_INDEX
            selected = label != IGNORE_INDEX
            action = mutation_map.get(position)
            if action == "supervise" and not originally_selected:
                fill = "#dc2626"
                stroke = "#fecaca"
                stroke_width = 2
            elif action == "ignore" and originally_selected:
                fill = "#25344a"
                stroke = "#f59e0b"
                stroke_width = 3
            elif selected:
                fill = "#16a34a"
                stroke = "#bbf7d0"
                stroke_width = 1
            else:
                fill = "#25344a"
                stroke = "#475569"
                stroke_width = 1
            pieces.append(
                f'  <rect x="{x}" y="{y}" width="{cell_width - 2}" '
                f'height="{cell_height}" rx="3" fill="{fill}" '
                f'stroke="{stroke}" stroke-width="{stroke_width}">'
                f"<title>position {position}: "
                f"{'supervised' if selected else 'ignored'}</title></rect>"
            )
            pieces.append(
                f'  <text x="{x + ((cell_width - 2) / 2):.1f}" '
                f'y="{y + 57}" class="mono tiny" text-anchor="middle">'
                f"{position}</text>"
            )
        summary_x = 1210
        pieces.extend(
            [
                f'  <text x="{summary_x}" y="{y + 2}" class="mono small">'
                f"runs: {html.escape(_format_runs(fault.audit))}</text>",
                f'  <text x="{summary_x}" y="{y + 26}" class="mono small">'
                "boundary: "
                f"{html.escape(_format_positions(fault.audit.boundary_leakage_positions))}</text>",
                f'  <text x="{summary_x}" y="{y + 50}" class="mono small">'
                "padding: "
                f"{html.escape(_format_positions(fault.audit.padding_leakage_positions))}</text>",
                f'  <text x="{summary_x}" y="{y + 74}" class="mono small">'
                "off-policy: "
                f"{html.escape(_format_positions(fault.audit.off_policy_supervision_positions))}</text>",
                f'  <text x="{summary_x}" y="{y + 98}" class="mono small">'
                "missing: "
                f"{html.escape(_format_positions(fault.audit.missing_eligible_positions))}</text>",
            ]
        )
    pieces.extend(
        [
            '  <rect x="72" y="730" width="1456" height="54" rx="12" fill="#111f34" stroke="#64748b"/>',
            '  <text x="94" y="762" class="sans small">These rows prove detection behavior for deliberately altered label arrays. The CLI itself accepts traces, constructs its own valid policies, and did not receive these mutations.</text>',
            '  <text x="72" y="804" class="sans tiny">Fixture: healthy-multiturn · policy: assistant_only · positions are zero-based and half-open runs</text>',
        ]
    )
    return _svg_document(
        width=width,
        height=height,
        title="Assistant-only fault injections and detected positions",
        description=(
            "Three controlled in-memory synthetic label mutations audited "
            "through the public assistant-only topology API."
        ),
        body="\n".join(pieces),
    )


def _transcript_bytes(runs: tuple[CLIRun, ...]) -> bytes:
    lines = [
        "LOSS TOPOLOGY LAB / REAL SUBPROCESS CAPTURE",
        (
            "evidence sandbox: committed synthetic fixtures copied "
            "byte-for-byte; repository loss_topology package"
        ),
        (
            "scope: no tokenizer, causal shift, loss, model, trainer, "
            "dataset, network, or GPU"
        ),
        "",
    ]
    for run in runs:
        command = " ".join(run.argv)
        lines.append(f"$ {command}")
        lines.extend(run.stdout.decode("utf-8").rstrip("\n").splitlines())
        stderr_state = "empty" if run.stderr == b"" else "captured"
        lines.append(
            f"[recorder] exit={run.returncode} stderr={stderr_state}"
        )
        lines.append("")
    lines.extend(
        [
            (
                "note: exit 1 is the expected diagnostic status for the "
                "valid empty-assistant fixture; it is not a CLI error"
            ),
            (
                "fault-injection evidence is executed separately in memory "
                "through audit_label_topology; it is not CLI output"
            ),
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _render_transcript_svg(
    transcript: bytes,
    runs: tuple[CLIRun, ...],
) -> bytes:
    lines = transcript.decode("utf-8").splitlines()
    width = 1600
    height = 900
    pieces = [
        '  <rect x="54" y="42" width="1492" height="810" rx="18" fill="#050a13" stroke="#334155" stroke-width="2"/>',
        '  <circle cx="88" cy="76" r="8" fill="#ef4444"/>',
        '  <circle cx="116" cy="76" r="8" fill="#f59e0b"/>',
        '  <circle cx="144" cy="76" r="8" fill="#22c55e"/>',
        '  <text x="800" y="82" class="mono small" text-anchor="middle">captured subprocess session</text>',
        '  <line x1="54" y1="102" x2="1546" y2="102" stroke="#334155"/>',
    ]
    y = 136
    for line in lines:
        escaped = html.escape(line)
        fill = "#d1fae5"
        weight = 400
        if line.startswith("LOSS TOPOLOGY"):
            fill = "#f8fafc"
            weight = 760
        elif line.startswith("$ "):
            fill = "#93c5fd"
            weight = 650
        elif line.startswith("[recorder]"):
            fill = "#fcd34d"
        elif line.startswith("note:") or line.startswith("fault-injection"):
            fill = "#cbd5e1"
        pieces.append(
            f'  <text x="82" y="{y}" class="mono" fill="{fill}" '
            f'font-size="15" font-weight="{weight}">{escaped}</text>'
        )
        y += 30

    healthy = next(run for run in runs if run.identifier == "healthy")
    empty = next(run for run in runs if run.identifier == "empty-assistant")
    pieces.extend(
        [
            '  <rect x="82" y="742" width="692" height="82" rx="12" fill="#0d2d22" stroke="#22c55e"/>',
            '  <text x="104" y="773" class="sans label">healthy · exit 0 · pass</text>',
            f'  <text x="104" y="801" class="mono tiny">audit sha256 {_sha256(healthy.output_bytes)}</text>',
            '  <rect x="798" y="742" width="720" height="82" rx="12" fill="#332810" stroke="#f59e0b"/>',
            '  <text x="820" y="773" class="sans label">empty assistant · exit 1 · diagnostic fail</text>',
            f'  <text x="820" y="801" class="mono tiny">audit sha256 {_sha256(empty.output_bytes)}</text>',
            '  <text x="54" y="882" class="sans tiny">Exact raw transcript and execution metadata are bound in loss-topology-evidence.v1.json.</text>',
        ]
    )
    return _svg_document(
        width=width,
        height=height,
        title="LossTopology Lab real subprocess capture",
        description=(
            "A deterministic rendering of actual standard-library CLI "
            "subprocess output for the healthy and empty-assistant fixtures."
        ),
        body="\n".join(pieces),
    )


def _policy_manifest(audit: PolicyAudit) -> dict[str, object]:
    return {
        "boundary_leakage_positions": list(
            audit.boundary_leakage_positions
        ),
        "missing_eligible_positions": list(
            audit.missing_eligible_positions
        ),
        "off_policy_supervision_positions": list(
            audit.off_policy_supervision_positions
        ),
        "padding_leakage_positions": list(audit.padding_leakage_positions),
        "supervised_runs": [
            {"end": run.end, "start": run.start}
            for run in audit.supervised_runs
        ],
    }


def _manifest(
    *,
    root_fd: int,
    runs: tuple[CLIRun, ...],
    faults: tuple[FaultRun, ...],
    artifacts: dict[str, bytes],
) -> bytes:
    run_entries = []
    for run in runs:
        decoded = json.loads(run.output_bytes)
        run_entries.append(
            {
                "argv": list(run.argv),
                "canonical_input_sha256": decoded["input_sha256"],
                "expected_exit_code": run.returncode,
                "fixture_path": run.input_path,
                "fixture_raw_sha256": _sha256(run.input_bytes),
                "id": run.identifier,
                "observed_exit_code": run.returncode,
                "output_sha256": _sha256(run.output_bytes),
                "stderr_sha256": _sha256(run.stderr),
                "stdout_sha256": _sha256(run.stdout),
            }
        )
    fault_entries = []
    for fault in faults:
        fault_entries.append(
            {
                "detected": _policy_manifest(fault.audit),
                "execution_boundary": (
                    "public audit_label_topology in-memory; not CLI input "
                    "or output"
                ),
                "id": fault.identifier,
                "mutations": [
                    {"action": action, "position": position}
                    for position, action in fault.mutations
                ],
                "policy": ASSISTANT_ONLY,
            }
        )
    source_entries = []
    for path in SOURCE_PATHS:
        payload = _read_repo_file(root_fd, path)
        source_entries.append(
            {
                "bytes": len(payload),
                "path": path,
                "sha256": _sha256(payload),
            }
        )
    artifact_entries = []
    for name in ARTIFACT_NAMES:
        payload = artifacts[name]
        artifact_entries.append(
            {
                "bytes": len(payload),
                "path": f"{'/'.join(OUTPUT_PARTS)}/{name}",
                "sha256": _sha256(payload),
            }
        )
    value = {
        "artifacts": artifact_entries,
        "cli_executions": run_entries,
        "fault_injections": fault_entries,
        "kind": "loss-topology.visual-evidence-manifest",
        "schema_version": 1,
        "scope": {
            "causal_shift_or_loss_computed": False,
            "contains_personal_data": False,
            "contains_secrets": False,
            "faults_are_cli_inputs_or_outputs": False,
            "fixture_class": "committed synthetic pretokenized JSON",
            "model_or_dataset_loaded": False,
            "network_used": False,
            "tokenizer_executed": False,
            "tokenizer_mapping_attested": False,
            "trainer_imported_or_executed": False,
        },
        "sources": source_entries,
    }
    return _json_bytes(value)


def _build_bundle(repository_root: Path = REPOSITORY_ROOT) -> dict[str, bytes]:
    root_fd = _open_root(repository_root)
    try:
        fixture_payloads = {
            path: _read_repo_file(root_fd, path) for path in FIXTURE_PATHS
        }
        for path, payload in fixture_payloads.items():
            if _sha256(payload) != EXPECTED_FIXTURE_SHA256[path]:
                raise EvidenceError("evidence.fixture_identity")
        healthy_trace = parse_synthetic_trace(
            fixture_payloads["fixtures/synthetic/healthy.v1.json"]
        )
        runs = _run_cli_evidence(repository_root, fixture_payloads)
        healthy_run = next(
            run for run in runs if run.identifier == "healthy"
        )
        empty_run = next(
            run for run in runs if run.identifier == "empty-assistant"
        )
        healthy_report = json.loads(healthy_run.output_bytes)
        empty_report = json.loads(empty_run.output_bytes)
        if (
            healthy_report["status"] != "pass"
            or empty_report["status"] != "fail"
            or empty_report["diagnostics"]["issue_codes"]
            != ["assistant_target.empty"]
        ):
            raise EvidenceError("evidence.fixture_expectation")
        faults = _fault_runs(healthy_trace)
        transcript = _transcript_bytes(runs)
        artifacts = {
            FAULT_SVG_NAME: _render_fault_svg(healthy_trace, faults),
            TRANSCRIPT_SVG_NAME: _render_transcript_svg(transcript, runs),
            TRANSCRIPT_NAME: transcript,
            EMPTY_NAME: empty_run.output_bytes,
            HEALTHY_NAME: healthy_run.output_bytes,
            POLICY_SVG_NAME: _render_policy_svg(
                healthy_trace,
                healthy_run.output_bytes,
            ),
        }
        manifest = _manifest(
            root_fd=root_fd,
            runs=runs,
            faults=faults,
            artifacts=artifacts,
        )
        return {
            **artifacts,
            MANIFEST_NAME: manifest,
        }
    finally:
        os.close(root_fd)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise EvidenceError("cli.arguments")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="render-loss-topology-evidence",
        description=(
            "Generate or verify evidence derived only from the two committed "
            "synthetic LossTopology fixtures."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        bundle = _build_bundle(REPOSITORY_ROOT)
        if arguments.write:
            _publish_bundle(REPOSITORY_ROOT, bundle)
            mode = "written"
        else:
            _check_bundle(REPOSITORY_ROOT, bundle)
            mode = "verified"
    except EvidenceError as error:
        sys.stderr.write(
            json.dumps(
                {
                    "error": {"code": error.code},
                    "kind": "loss-topology.visual-evidence-error",
                    "status": "error",
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )
        return 2
    except Exception:
        sys.stderr.write(
            '{"error":{"code":"internal.error"},'
            '"kind":"loss-topology.visual-evidence-error",'
            '"status":"error"}\n'
        )
        return 2
    sys.stdout.write(
        json.dumps(
            {
                "artifacts": len(PUBLICATION_ORDER),
                "mode": mode,
                "status": "ok",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
