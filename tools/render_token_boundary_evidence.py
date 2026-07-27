"""Build deterministic, source-bound evidence for the token-boundary lab.

The generator is closed over seven committed synthetic fixtures, the local
hash-pinned tokenizer artifact, and the public stdin CLI. It does not import or
execute the quarantined trainer, access the network, or accept caller-selected
inputs or output paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Final, Never

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from token_boundary import analyze_boundary
from token_boundary.contract import (
    BoundaryCase,
    canonical_boundary_case_bytes,
    parse_boundary_case,
)
from token_boundary.errors import BoundaryEngineError
from token_boundary.report import canonical_boundary_report_bytes

OUTPUT_PARTS: Final = ("docs", "token-boundary", "generated")
MANIFEST_NAME: Final = "token-boundary-evidence.v1.json"
TRANSCRIPT_NAME: Final = "cli-session.txt"
TRANSCRIPT_SVG_NAME: Final = "cli-session.svg"
ARCHITECTURE_SVG_NAME: Final = "token-boundary-architecture.svg"
LANES_SVG_NAME: Final = "token-lanes.svg"
MECHANISMS_SVG_NAME: Final = "boundary-mechanisms.svg"
SWEEP_NAME: Final = "truncation-sweep.csv"
MATRIX_SVG_NAME: Final = "truncation-matrix.svg"
FIXTURE_PATHS: Final = (
    "fixtures/token-boundary/aligned.v1.json",
    "fixtures/token-boundary/merge-cross-boundary.v1.json",
    "fixtures/token-boundary/nfc-cross-boundary.v1.json",
    "fixtures/token-boundary/normalized-away.v1.json",
    "fixtures/token-boundary/partial-truncation.v1.json",
    "fixtures/token-boundary/right-strip-drift.v1.json",
    "fixtures/token-boundary/target-eliminated.v1.json",
)
EXPECTED_FIXTURE_SHA256: Final = {
    "fixtures/token-boundary/aligned.v1.json": (
        "47564f771516691fbe5fe8b2d55bc5c945ad383714c92d878830d82d84e7801f"
    ),
    "fixtures/token-boundary/merge-cross-boundary.v1.json": (
        "4c0ddbc30422e085ca8edf0aed09ffeda4a186c97029ad0ee31253bd74db3c97"
    ),
    "fixtures/token-boundary/nfc-cross-boundary.v1.json": (
        "d526241f40533c571fccc75586ec01631278f897f689ff4bad0c36aaf49ba7c3"
    ),
    "fixtures/token-boundary/normalized-away.v1.json": (
        "4d4fa0e8e911c9c2e4235c8de9faacf6f78e5b304d81317248965dd82b4a001f"
    ),
    "fixtures/token-boundary/partial-truncation.v1.json": (
        "2c2fc8b7f4cd7a590c18bacdb5502b6d95af3ac5dbde8652c6cf372f0bf49f4e"
    ),
    "fixtures/token-boundary/right-strip-drift.v1.json": (
        "3c17b4f20be66a6c2608e4ccc6c09cead04004988754101d7e5478df6ea323ef"
    ),
    "fixtures/token-boundary/target-eliminated.v1.json": (
        "1c86ebe0feae22c43c3ecb27c1a17489e4a4b55ce3b57573bcda7fe87ec5a5d7"
    ),
}
EXPECTED_RUNS: Final = {
    "aligned": {
        "exit": 0,
        "status": "pass",
        "primary": "aligned",
        "sha256": ("49f851a6d89d25d5aee9d8b68548883dfbc2962a5610e9b55c2dfde176ac6d76"),
    },
    "merge-cross-boundary": {
        "exit": 1,
        "status": "fail",
        "primary": "cross_boundary_token",
        "sha256": ("b4c7364274c4c0f43c70db7b8b3e852cc4d4fbefeefd0d77ff60d7eef460721b"),
    },
    "nfc-cross-boundary": {
        "exit": 1,
        "status": "fail",
        "primary": "cross_boundary_token",
        "sha256": ("1aeec08efe815ca11b599b1b9427c94a951aebfa2c004e3733fa06b3a662344a"),
    },
    "normalized-away": {
        "exit": 1,
        "status": "indeterminate",
        "primary": "indeterminate",
        "sha256": ("ef5b1e1ced0609f09c5bf4a8a05c31406d0e0fe0a662287583eaed57bc5bf475"),
    },
    "partial-truncation": {
        "exit": 1,
        "status": "fail",
        "primary": "partial_target_truncation",
        "sha256": ("35186327542ccaeb3d3f6ce1f6d1ff85b25c23bc02a36ace71cc9ab0cfcc0179"),
    },
    "right-strip-drift": {
        "exit": 1,
        "status": "fail",
        "primary": "boundary_drift",
        "sha256": ("ffe7b5194512f64c5d998acef0f3d7d026f29cb4f3fef26b28ca9b1baee95e01"),
    },
    "target-eliminated": {
        "exit": 1,
        "status": "fail",
        "primary": "target_eliminated",
        "sha256": ("d5c2121a22967b0a187882d978a66a919d653f8997649402e9e3e1246a73e48f"),
    },
}
REPORT_NAMES: Final = {
    identifier: f"{identifier}.boundary-report.json" for identifier in EXPECTED_RUNS
}
ARTIFACT_NAMES: Final = (
    ARCHITECTURE_SVG_NAME,
    MECHANISMS_SVG_NAME,
    TRANSCRIPT_SVG_NAME,
    TRANSCRIPT_NAME,
    *tuple(REPORT_NAMES.values()),
    LANES_SVG_NAME,
    MATRIX_SVG_NAME,
    SWEEP_NAME,
)
PUBLICATION_ORDER: Final = (*ARTIFACT_NAMES, MANIFEST_NAME)
SWEEP_LENGTHS: Final = tuple(range(2, 10))
SOURCE_PATHS: Final = (
    *FIXTURE_PATHS,
    "finetune.py",
    "provenance/legacy-snapshot.v1.json",
    "pyproject.toml",
    "token_boundary/__init__.py",
    "token_boundary/artifact.py",
    "token_boundary/artifacts/artifact-manifest.v1.json",
    "token_boundary/artifacts/local-boundary-bpe.v1.json",
    "token_boundary/cli.py",
    "token_boundary/contract.py",
    "token_boundary/engine.py",
    "token_boundary/errors.py",
    "token_boundary/provenance.py",
    "token_boundary/report.py",
    "tools/render_token_boundary_evidence.py",
)
MAX_SOURCE_BYTES: Final = 2 * 1024 * 1024
EMPTY_SHA256: Final = hashlib.sha256(b"").hexdigest()
SVG_NAMESPACE: Final = "http://www.w3.org/2000/svg"


class EvidenceError(RuntimeError):
    """Stable evidence generation failure."""

    __slots__ = ("code",)

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class CLIRun:
    """One real stdin subprocess and its canonical output."""

    identifier: str
    fixture_path: str
    display_input_path: str
    argv: tuple[str, ...]
    input_bytes: bytes
    stdout: bytes
    stderr: bytes
    returncode: int
    report: dict[str, object]


@dataclass(frozen=True, slots=True)
class SweepRun:
    """One fresh CLI execution with an authored max-length override."""

    base_identifier: str
    base_fixture_path: str
    fixture_raw_sha256: str
    max_length: int
    input_bytes: bytes
    stdout: bytes
    stderr: bytes
    returncode: int
    report: dict[str, object]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


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


def _read_regular(path: Path, *, max_bytes: int = MAX_SOURCE_BYTES) -> bytes:
    try:
        metadata = path.lstat()
    except OSError:
        raise EvidenceError("io.file_unavailable")
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > max_bytes:
        raise EvidenceError("io.file_not_regular")
    try:
        payload = path.read_bytes()
        after = path.lstat()
    except OSError:
        raise EvidenceError("io.file_unavailable")
    before_identity = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity or len(payload) != metadata.st_size:
        raise EvidenceError("io.file_changed")
    return payload


def _read_source(repository_root: Path, relative_path: str) -> bytes:
    current = repository_root
    for part in _safe_parts(relative_path)[:-1]:
        current /= part
        try:
            metadata = current.lstat()
        except OSError:
            raise EvidenceError("io.source_unavailable")
        if not stat.S_ISDIR(metadata.st_mode):
            raise EvidenceError("io.source_parent")
    return _read_regular(repository_root.joinpath(*_safe_parts(relative_path)))


def _output_root(repository_root: Path, *, create: bool) -> Path:
    current = repository_root
    for part in OUTPUT_PARTS:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if not create:
                raise EvidenceError("io.output_directory_missing")
            try:
                current.mkdir(mode=0o755)
                metadata = current.lstat()
            except OSError:
                raise EvidenceError("io.output_directory_unavailable")
        except OSError:
            raise EvidenceError("io.output_directory_unavailable")
        if not stat.S_ISDIR(metadata.st_mode):
            raise EvidenceError("io.output_directory_unavailable")
    return current


def _validate_bundle(bundle: dict[str, bytes]) -> None:
    if tuple(bundle) != PUBLICATION_ORDER:
        raise EvidenceError("bundle.order")
    if any(type(value) is not bytes for value in bundle.values()):
        raise EvidenceError("bundle.type")


def _validate_inventory(output_root: Path, *, allow_missing: bool) -> None:
    try:
        names = set(os.listdir(output_root))
    except OSError:
        raise EvidenceError("io.output_directory_unavailable")
    expected = set(PUBLICATION_ORDER)
    if allow_missing:
        if not names <= expected:
            raise EvidenceError("bundle.inventory")
    elif names != expected:
        raise EvidenceError("bundle.inventory")
    for name in names:
        _read_regular(output_root / name)


def _fsync_file(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        os.fsync(descriptor)
    except OSError:
        raise EvidenceError("io.stage_failed")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
        )
        os.fsync(descriptor)
    except OSError:
        raise EvidenceError("io.directory_sync_failed")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _remove_tree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except OSError:
        raise EvidenceError("io.cleanup_failed")


def _publish_bundle(repository_root: Path, bundle: dict[str, bytes]) -> None:
    """Publish a complete fixed-inventory bundle, rolling back on failure."""

    _validate_bundle(bundle)
    output_root = _output_root(repository_root, create=True)
    _validate_inventory(output_root, allow_missing=True)
    stage: Path | None = None
    backup: Path | None = None
    try:
        stage = Path(
            tempfile.mkdtemp(
                prefix=".token-boundary-evidence-stage-",
                dir=output_root.parent,
            )
        )
        backup = Path(
            tempfile.mkdtemp(
                prefix=".token-boundary-evidence-backup-",
                dir=output_root.parent,
            )
        )
    except OSError:
        for temporary in (stage, backup):
            if temporary is not None:
                try:
                    _remove_tree(temporary)
                except EvidenceError:
                    pass
        raise EvidenceError("io.stage_failed")
    assert stage is not None
    assert backup is not None

    published: list[str] = []
    try:
        for name, payload in bundle.items():
            target = stage / name
            try:
                target.write_bytes(payload)
                target.chmod(0o644)
            except OSError:
                raise EvidenceError("io.stage_failed")
            _fsync_file(target)

        try:
            for name in PUBLICATION_ORDER:
                destination = output_root / name
                previous = backup / name
                if destination.exists():
                    os.replace(destination, previous)
                os.replace(stage / name, destination)
                published.append(name)
            _fsync_directory(output_root)
            _validate_inventory(output_root, allow_missing=False)
        except (OSError, EvidenceError):
            rollback_failed = False
            for name in reversed(PUBLICATION_ORDER):
                destination = output_root / name
                previous = backup / name
                try:
                    if name in published and destination.exists():
                        destination.unlink()
                    if previous.exists():
                        os.replace(previous, destination)
                except OSError:
                    rollback_failed = True
            try:
                _fsync_directory(output_root)
            except EvidenceError:
                rollback_failed = True
            if rollback_failed:
                raise EvidenceError("io.rollback_failed")
            raise EvidenceError("io.publish_failed")
    finally:
        cleanup_error: EvidenceError | None = None
        for temporary in (stage, backup):
            try:
                _remove_tree(temporary)
            except EvidenceError as error:
                cleanup_error = error
        if cleanup_error is not None:
            raise cleanup_error


def _check_bundle(repository_root: Path, bundle: dict[str, bytes]) -> None:
    _validate_bundle(bundle)
    output_root = _output_root(repository_root, create=False)
    _validate_inventory(output_root, allow_missing=False)
    for name, expected in bundle.items():
        if _read_regular(output_root / name) != expected:
            raise EvidenceError("bundle.drift")


def _report_dict(payload: bytes) -> dict[str, object]:
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise EvidenceError("runtime.report_json")
    if type(decoded) is not dict:
        raise EvidenceError("runtime.report_json")
    assert isinstance(decoded, dict)
    return decoded


def _mapping(value: object, code: str) -> dict[str, object]:
    if type(value) is not dict:
        raise EvidenceError(code)
    assert isinstance(value, dict)
    return value


def _sequence(value: object, code: str) -> list[object]:
    if type(value) is not list:
        raise EvidenceError(code)
    assert isinstance(value, list)
    return value


def _string(value: object, code: str) -> str:
    if type(value) is not str:
        raise EvidenceError(code)
    assert isinstance(value, str)
    return value


def _integer(value: object, code: str) -> int:
    if type(value) is not int:
        raise EvidenceError(code)
    assert isinstance(value, int)
    return value


def _optional_integer(value: object, code: str) -> int | None:
    if value is None:
        return None
    return _integer(value, code)


def _run_process(
    *,
    repository_root: Path,
    sandbox: Path,
    input_bytes: bytes,
    display_name: str,
) -> tuple[bytes, bytes, int]:
    input_root = sandbox / "input"
    input_root.mkdir(mode=0o755, exist_ok=True)
    input_path = input_root / display_name
    try:
        input_path.write_bytes(input_bytes)
        captured_input = input_path.read_bytes()
    except OSError:
        raise EvidenceError("io.sandbox_input")
    if captured_input != input_bytes:
        raise EvidenceError("io.sandbox_input")

    environment = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": f"{Path(sys.executable).parent}:{os.defpath}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(repository_root),
        "TOKENIZERS_PARALLELISM": "false",
    }
    try:
        result = subprocess.run(
            (sys.executable, "-m", "token_boundary.cli"),
            cwd=sandbox,
            env=environment,
            input=captured_input,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        raise EvidenceError("runtime.cli_failed")
    return result.stdout, result.stderr, result.returncode


def _expected_output(case: BoundaryCase) -> bytes:
    return canonical_boundary_report_bytes(analyze_boundary(case))


def _fixture_identifier(path: str) -> str:
    name = PurePosixPath(path).name
    suffix = ".v1.json"
    if not name.endswith(suffix):
        raise EvidenceError("evidence.fixture_name")
    return name[: -len(suffix)]


def _run_fixture_cli(
    repository_root: Path,
    sandbox: Path,
    fixture_path: str,
    payload: bytes,
) -> CLIRun:
    identifier = _fixture_identifier(fixture_path)
    expectation = EXPECTED_RUNS[identifier]
    display_input = f"input/{PurePosixPath(fixture_path).name}"
    stdout, stderr, returncode = _run_process(
        repository_root=repository_root,
        sandbox=sandbox,
        input_bytes=payload,
        display_name=PurePosixPath(fixture_path).name,
    )
    case = parse_boundary_case(payload)
    expected = _expected_output(case)
    report = _report_dict(stdout)
    if (
        stdout != expected
        or stderr != b""
        or returncode != expectation["exit"]
        or report.get("status") != expectation["status"]
        or report.get("primary_classification") != expectation["primary"]
        or _sha256(stdout) != expectation["sha256"]
    ):
        raise EvidenceError("evidence.fixture_expectation")
    return CLIRun(
        identifier=identifier,
        fixture_path=fixture_path,
        display_input_path=display_input,
        argv=("python3", "-m", "token_boundary.cli"),
        input_bytes=payload,
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        report=report,
    )


def _run_fixture_suite(
    repository_root: Path,
    fixture_payloads: dict[str, bytes],
    sandbox: Path,
) -> tuple[CLIRun, ...]:
    return tuple(
        _run_fixture_cli(
            repository_root,
            sandbox,
            path,
            fixture_payloads[path],
        )
        for path in FIXTURE_PATHS
    )


def _run_sweep(
    repository_root: Path,
    fixture_payloads: dict[str, bytes],
    sandbox: Path,
    expected_artifact_identity: dict[str, object],
) -> tuple[SweepRun, ...]:
    runs: list[SweepRun] = []
    for fixture_path in FIXTURE_PATHS:
        identifier = _fixture_identifier(fixture_path)
        fixture_payload = fixture_payloads[fixture_path]
        base_case = parse_boundary_case(fixture_payload)
        for max_length in SWEEP_LENGTHS:
            case = replace(base_case, max_length=max_length)
            input_bytes = canonical_boundary_case_bytes(case)
            display_name = f"sweep-{identifier}-{max_length}.json"
            stdout, stderr, returncode = _run_process(
                repository_root=repository_root,
                sandbox=sandbox,
                input_bytes=input_bytes,
                display_name=display_name,
            )
            expected = _expected_output(case)
            report = _report_dict(stdout)
            expected_exit = 0 if report.get("status") == "pass" else 1
            identity = _input_identity(report)
            if stdout != expected or stderr != b"" or returncode != expected_exit:
                raise EvidenceError("evidence.sweep_expectation")
            if (
                _string(
                    identity.get("canonical_sha256"),
                    "report.input_identity",
                )
                != _sha256(input_bytes)
                or report.get("artifact_identity") != expected_artifact_identity
            ):
                raise EvidenceError("evidence.sweep_identity")
            runs.append(
                SweepRun(
                    base_identifier=identifier,
                    base_fixture_path=fixture_path,
                    fixture_raw_sha256=_sha256(fixture_payload),
                    max_length=max_length,
                    input_bytes=input_bytes,
                    stdout=stdout,
                    stderr=stderr,
                    returncode=returncode,
                    report=report,
                )
            )
    return tuple(runs)


def _diagnostics(report: dict[str, object]) -> dict[str, object]:
    return _mapping(report.get("diagnostics"), "report.diagnostics")


def _input_identity(report: dict[str, object]) -> dict[str, object]:
    return _mapping(report.get("input_identity"), "report.input_identity")


def _legacy_mask(report: dict[str, object]) -> dict[str, object]:
    return _mapping(report.get("legacy_mask"), "report.legacy_mask")


def _encodings(report: dict[str, object]) -> dict[str, object]:
    return _mapping(report.get("encodings"), "report.encodings")


def _encoding(
    report: dict[str, object],
    name: str,
) -> dict[str, object]:
    return _mapping(_encodings(report).get(name), "report.encoding")


def _attribution(
    report: dict[str, object],
    name: str,
) -> list[object]:
    attribution = _mapping(report.get("attribution"), "report.attribution")
    return _sequence(attribution.get(name), "report.attribution")


def _normalization(
    report: dict[str, object],
    name: str,
) -> dict[str, object]:
    normalization = _mapping(
        report.get("normalization"),
        "report.normalization",
    )
    return _mapping(normalization.get(name), "report.normalization")


def _join_values(value: object) -> str:
    return "|".join(str(item) for item in _sequence(value, "report.sequence"))


def _sweep_bytes(runs: tuple[SweepRun, ...]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        (
            "case_id",
            "base_fixture_path",
            "fixture_raw_sha256",
            "max_length",
            "canonical_input_sha256",
            "exit_code",
            "stdout_sha256",
            "status",
            "primary_classification",
            "legacy_cutoff",
            "oracle_cutoff",
            "full_target_tokens",
            "retained_target_tokens",
            "supervised_target_tokens",
            "truncated_target_tokens",
            "prompt_leakage_positions",
            "cross_boundary_positions",
            "elimination_causes",
            "indeterminate_reasons",
        )
    )
    for run in runs:
        report = run.report
        diagnostics = _diagnostics(report)
        identity = _input_identity(report)
        legacy = _legacy_mask(report)
        oracle_cutoff = _optional_integer(
            diagnostics.get("oracle_cutoff"),
            "report.oracle_cutoff",
        )
        writer.writerow(
            (
                run.base_identifier,
                run.base_fixture_path,
                run.fixture_raw_sha256,
                run.max_length,
                _string(
                    identity.get("canonical_sha256"),
                    "report.input_identity",
                ),
                run.returncode,
                _sha256(run.stdout),
                _string(report.get("status"), "report.status"),
                _string(
                    report.get("primary_classification"),
                    "report.primary",
                ),
                _integer(legacy.get("cutoff"), "report.cutoff"),
                "" if oracle_cutoff is None else oracle_cutoff,
                _integer(
                    diagnostics.get("full_target_token_count"),
                    "report.target_count",
                ),
                _integer(
                    diagnostics.get("retained_target_token_count"),
                    "report.target_count",
                ),
                _integer(
                    diagnostics.get("supervised_target_token_count"),
                    "report.target_count",
                ),
                _integer(
                    diagnostics.get("truncated_target_token_count"),
                    "report.target_count",
                ),
                _join_values(diagnostics.get("prompt_leakage_positions")),
                _join_values(diagnostics.get("cross_boundary_positions")),
                _join_values(diagnostics.get("elimination_causes")),
                _join_values(diagnostics.get("indeterminate_reasons")),
            )
        )
    return stream.getvalue().encode("utf-8")


def _transcript_bytes(runs: tuple[CLIRun, ...]) -> bytes:
    lines = [
        "TOKEN BOUNDARY LAB / REAL STDIN SUBPROCESS CAPTURE",
        (
            "recorder: seven committed synthetic fixtures; canonical stdout "
            "is preserved verbatim between markers"
        ),
        (
            "scope: local hash-pinned tokenizer artifact only; no inherited "
            "trainer, model, dataset, network, GPU, loss, or training"
        ),
        "",
    ]
    for run in runs:
        command = " ".join(run.argv)
        lines.append(f"$ {command} < {run.display_input_path}")
        lines.append(f"--- BEGIN CANONICAL STDOUT {REPORT_NAMES[run.identifier]} ---")
        lines.extend(run.stdout.decode("ascii").rstrip("\n").splitlines())
        lines.append("--- END CANONICAL STDOUT ---")
        lines.append(
            "[recorder] "
            f"exit={run.returncode} "
            f"status={run.report['status']} "
            f"primary={run.report['primary_classification']} "
            f"stdout_bytes={len(run.stdout)} "
            f"stdout_sha256={_sha256(run.stdout)} "
            f"stderr={'empty' if run.stderr == b'' else 'captured'} "
            f"stderr_sha256={_sha256(run.stderr)}"
        )
        lines.append("")
    lines.extend(
        [
            (
                "note: exit 1 is a complete valid fail or indeterminate "
                "report; process-boundary errors use exit 2"
            ),
            (
                "note: authored synthetic cases are counterexamples, not a "
                "sample, benchmark, or prevalence estimate"
            ),
        ]
    )
    return ("\n".join(lines) + "\n").encode("ascii")


def _svg_document(
    *,
    identifier: str,
    width: int,
    height: int,
    title: str,
    description: str,
    body: str,
) -> bytes:
    title_id = f"{identifier}-title"
    description_id = f"{identifier}-description"
    document = f"""<svg xmlns="{SVG_NAMESPACE}" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="{title_id} {description_id}">
  <title id="{title_id}">{html.escape(title)}</title>
  <desc id="{description_id}">{html.escape(description)}</desc>
  <rect width="{width}" height="{height}" fill="#07111f"/>
  <style>
    .sans {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
    .title {{ fill: #f8fafc; font-size: 32px; font-weight: 760; }}
    .subtitle {{ fill: #d7e1ee; font-size: 18px; }}
    .heading {{ fill: #f8fafc; font-size: 22px; font-weight: 720; }}
    .label {{ fill: #eef4fb; font-size: 17px; font-weight: 650; }}
    .body {{ fill: #d7e1ee; font-size: 16px; }}
    .small {{ fill: #d7e1ee; font-size: 14px; }}
    .tiny {{ fill: #d7e1ee; font-size: 12px; }}
    .scope {{ fill: #dbeafe; font-size: 14px; font-weight: 680; letter-spacing: .25px; }}
  </style>
{body}
</svg>
"""
    payload = document.encode("utf-8")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        raise EvidenceError("render.svg")
    if root.tag != f"{{{SVG_NAMESPACE}}}svg":
        raise EvidenceError("render.svg")
    return payload


def _line(
    *,
    x: int,
    y: int,
    text: str,
    css_class: str = "sans body",
    anchor: str | None = None,
    fill: str | None = None,
) -> str:
    attributes = [f'x="{x}"', f'y="{y}"', f'class="{css_class}"']
    if anchor is not None:
        attributes.append(f'text-anchor="{anchor}"')
    if fill is not None:
        attributes.append(f'fill="{fill}"')
    return f"  <text {' '.join(attributes)}>{html.escape(text)}</text>"


def _architecture_svg(
    runs: tuple[CLIRun, ...],
    legacy_manifest: dict[str, object],
) -> bytes:
    first = runs[0].report
    artifact = _mapping(
        first.get("artifact_identity"),
        "report.artifact_identity",
    )
    artifact_sha = _string(
        artifact.get("sha256"),
        "report.artifact_identity",
    )
    engine_version = _string(
        artifact.get("engine_version"),
        "report.artifact_identity",
    )
    snapshot = _mapping(legacy_manifest.get("snapshot"), "legacy.snapshot")
    files = _sequence(legacy_manifest.get("files"), "legacy.files")
    trainer = next(
        (
            _mapping(item, "legacy.file")
            for item in files
            if _mapping(item, "legacy.file").get("path") == "finetune.py"
        ),
        None,
    )
    if trainer is None:
        raise EvidenceError("legacy.trainer")

    width = 1000
    height = 1450
    pieces = [
        _line(
            x=50,
            y=64,
            text="Executable token-boundary audit architecture",
            css_class="sans title",
        ),
        _line(
            x=50,
            y=98,
            text=(
                "One bounded stdin document → one canonical report; all "
                "labels below are executable boundaries"
            ),
            css_class="sans subtitle",
        ),
        '  <rect x="50" y="122" width="900" height="50" rx="10" fill="#102744" stroke="#60a5fa"/>',
        _line(
            x=72,
            y=153,
            text=(
                "NO PATH / URL / ENV-SELECTED INPUT · NO NETWORK · "
                "NO MODEL / DATASET / GPU"
            ),
            css_class="sans scope",
        ),
    ]
    cards = (
        (
            "1",
            "stdin process boundary",
            "No arguments · regular file/FIFO · 32 KiB + sentinel · 5 s pipe deadline",
            "#2563eb",
        ),
        (
            "2",
            "closed synthetic case parser",
            "Exact v1 fields · duplicate/float/invalid UTF-8 rejection · canonical input SHA-256",
            "#2563eb",
        ),
        (
            "3",
            "fixed local tokenizer artifact",
            (
                f"tokenizers=={engine_version} · 1,533 bytes · "
                f"SHA-256 {artifact_sha[:16]}…"
            ),
            "#7c3aed",
        ),
        (
            "4",
            "three fresh tokenizer instances",
            "source truncated · source+target full · source+target right-truncated",
            "#7c3aed",
        ),
        (
            "5",
            "independent provenance replay",
            "NFC clusters + right-only strip · exact raw origins · fail closed on ambiguity",
            "#0f766e",
        ),
        (
            "6",
            "ownership oracle + inherited cutoff",
            "source / target / cross / injected ownership compared with len(T(source)) mask",
            "#0f766e",
        ),
        (
            "7",
            "classification and canonical process output",
            "aligned / drift / cross / partial / eliminated / indeterminate · exits 0 / 1 / 2",
            "#b45309",
        ),
    )
    y = 210
    for index, title, detail, color in cards:
        pieces.extend(
            [
                f'  <rect x="130" y="{y}" width="740" height="116" rx="16" fill="#101f33" stroke="{color}" stroke-width="2"/>',
                f'  <circle cx="178" cy="{y + 58}" r="26" fill="{color}"/>',
                _line(
                    x=178,
                    y=y + 66,
                    text=index,
                    css_class="sans label",
                    anchor="middle",
                ),
                _line(
                    x=222,
                    y=y + 44,
                    text=title,
                    css_class="sans heading",
                ),
                _line(
                    x=222,
                    y=y + 77,
                    text=detail,
                    css_class="sans small",
                ),
            ]
        )
        if index != "7":
            pieces.append(
                f'  <path d="M500 {y + 116} V{y + 150}" stroke="#64748b" stroke-width="3"/>'
            )
            pieces.append(
                f'  <path d="M491 {y + 141} L500 {y + 151} L509 {y + 141}" fill="none" stroke="#64748b" stroke-width="3"/>'
            )
        y += 158

    trainer_sha = _string(trainer.get("sha256"), "legacy.file")
    source_commit = _string(
        snapshot.get("source_git_commit"),
        "legacy.snapshot",
    )
    pieces.extend(
        [
            '  <rect x="50" y="1328" width="900" height="70" rx="13" fill="#2b1720" stroke="#f472b6" stroke-width="2" stroke-dasharray="8 6"/>',
            _line(
                x=72,
                y=1356,
                text="QUARANTINED SOURCE BINDING — never imported or executed",
                css_class="sans label",
            ),
            _line(
                x=72,
                y=1382,
                text=(
                    f"finetune.py SHA-256 {trainer_sha[:16]}… · snapshot "
                    f"{source_commit[:12]}… · algorithm id only"
                ),
                css_class="mono small",
            ),
            _line(
                x=50,
                y=1420,
                text=(
                    "Source-bound diagram generated from CLI/contract/engine/"
                    "provenance/report bytes and the legacy attestation."
                ),
                css_class="sans tiny",
            ),
        ]
    )
    return _svg_document(
        identifier="architecture",
        width=width,
        height=height,
        title="Executable token-boundary audit architecture",
        description=(
            "A vertical source-derived workflow from bounded stdin through "
            "artifact verification, three tokenizer encodings, provenance "
            "replay, ownership analysis, and canonical output. The inherited "
            "trainer is shown separately as attestation-only and quarantined."
        ),
        body="\n".join(pieces),
    )


def _display_raw(value: str) -> str:
    pieces: list[str] = []
    for character in value:
        codepoint = ord(character)
        if character == " ":
            pieces.append("[SPACE]")
        elif 0x300 <= codepoint <= 0x36F:
            pieces.append(f"U+{codepoint:04X}")
        elif character.isprintable():
            pieces.append(character)
        else:
            pieces.append(f"U+{codepoint:04X}")
    return "".join(pieces)


def _display_piece(value: object) -> str:
    piece = _string(value, "report.piece")
    if piece == " ":
        return "SPACE"
    return piece


def _ownership_code(value: object) -> str:
    ownership = _string(value, "report.ownership")
    return {
        "injected_prefix": "I",
        "source": "S",
        "target": "T",
        "cross_boundary": "X",
        "ambiguous": "?",
    }.get(ownership, "?")


def _token_lane(
    *,
    y: int,
    label: str,
    encoding: dict[str, object],
    attribution: list[object],
    legacy: dict[str, object],
    full_width: int = 650,
) -> list[str]:
    pieces = _sequence(encoding.get("pieces"), "report.pieces")
    if len(pieces) != len(attribution) or not pieces:
        raise EvidenceError("render.lane")
    masked = {
        _integer(item, "report.position")
        for item in _sequence(
            legacy.get("masked_positions"),
            "report.masked_positions",
        )
    }
    supervised = {
        _integer(item, "report.position")
        for item in _sequence(
            legacy.get("supervised_positions"),
            "report.supervised_positions",
        )
    }
    start_x = 255
    cell_width = min(82, full_width // len(pieces))
    ownership_colors = {
        "I": "#475569",
        "S": "#2563eb",
        "T": "#0f766e",
        "X": "#9333ea",
        "?": "#b45309",
    }
    result = [
        _line(x=66, y=y + 31, text=label, css_class="sans small"),
    ]
    for position, (piece, raw_attribution) in enumerate(
        zip(pieces, attribution, strict=True)
    ):
        item = _mapping(raw_attribution, "report.attribution")
        ownership = _ownership_code(item.get("ownership"))
        state = "M" if position in masked else "S" if position in supervised else "—"
        x = start_x + position * cell_width
        stroke = "#f59e0b" if ownership == "X" else "#94a3b8"
        result.extend(
            [
                f'  <rect x="{x}" y="{y}" width="{cell_width - 5}" height="58" rx="7" fill="{ownership_colors[ownership]}" stroke="{stroke}" stroke-width="2"/>',
                _line(
                    x=x + (cell_width - 5) // 2,
                    y=y + 23,
                    text=_display_piece(piece),
                    css_class="mono small",
                    anchor="middle",
                ),
                _line(
                    x=x + (cell_width - 5) // 2,
                    y=y + 46,
                    text=f"{position} · {ownership}/{state}",
                    css_class="mono tiny",
                    anchor="middle",
                ),
            ]
        )
    return result


def _lanes_svg(
    runs: tuple[CLIRun, ...],
    fixture_payloads: dict[str, bytes],
) -> bytes:
    width = 1000
    row_height = 244
    height = 230 + len(runs) * row_height
    pieces = [
        _line(
            x=46,
            y=60,
            text="Seven executed boundary differentials",
            css_class="sans title",
        ),
        _line(
            x=46,
            y=94,
            text=(
                "Exact full/retained token lanes · ownership + M(masked) / "
                "S(supervised) · positions are zero-based"
            ),
            css_class="sans subtitle",
        ),
        '  <rect x="46" y="120" width="908" height="66" rx="12" fill="#10233f" stroke="#475569"/>',
        _line(
            x=66,
            y=147,
            text="Ownership: I injected · S source · T target · X cross-boundary · ? ambiguous",
            css_class="sans small",
        ),
        _line(
            x=66,
            y=171,
            text="M masked · S supervised · — not retained · [SPACE] = U+0020 · U+0301 is a display escape.",
            css_class="sans small",
        ),
    ]
    y = 210
    for run in runs:
        report = run.report
        fixture = parse_boundary_case(fixture_payloads[run.fixture_path])
        legacy = _legacy_mask(report)
        diagnostics = _diagnostics(report)
        primary = _string(
            report.get("primary_classification"),
            "report.primary",
        )
        status = _string(report.get("status"), "report.status")
        authored = f"{_display_raw(fixture.source)} | {_display_raw(fixture.target)}"
        cutoff = _integer(legacy.get("cutoff"), "report.cutoff")
        oracle = _optional_integer(
            diagnostics.get("oracle_cutoff"),
            "report.oracle_cutoff",
        )
        retained = _integer(
            diagnostics.get("retained_target_token_count"),
            "report.target_count",
        )
        full = _integer(
            diagnostics.get("full_target_token_count"),
            "report.target_count",
        )
        truncated = _integer(
            diagnostics.get("truncated_target_token_count"),
            "report.target_count",
        )
        pieces.extend(
            [
                f'  <rect x="46" y="{y}" width="908" height="228" rx="14" fill="#0d1b2e" stroke="#334155"/>',
                _line(
                    x=66,
                    y=y + 29,
                    text=f"{run.identifier} · {status} · {primary}",
                    css_class="sans label",
                ),
                _line(
                    x=66,
                    y=y + 55,
                    text=(
                        f"authored {authored} · max_length {fixture.max_length} "
                        f"· cutoff {cutoff} / oracle "
                        f"{'n/a' if oracle is None else oracle}"
                    ),
                    css_class="mono tiny",
                ),
            ]
        )
        pieces.extend(
            _token_lane(
                y=y + 67,
                label="full",
                encoding=_encoding(report, "combined_full"),
                attribution=_attribution(report, "full"),
                legacy=legacy,
            )
        )
        pieces.extend(
            _token_lane(
                y=y + 128,
                label="retained",
                encoding=_encoding(report, "combined_truncated"),
                attribution=_attribution(report, "truncated"),
                legacy=legacy,
            )
        )
        pieces.append(
            _line(
                x=66,
                y=y + 214,
                text=(
                    f"target retained/full {retained}/{full} · truncated "
                    f"{truncated} · report {_sha256(run.stdout)[:12]}…"
                ),
                css_class="sans tiny",
            )
        )
        y += row_height
    return _svg_document(
        identifier="lanes",
        width=width,
        height=height,
        title="Seven executed boundary differentials",
        description=(
            "Full and retained tokenizer lanes for every committed authored "
            "case, with exact token pieces, ownership, mask state, cutoff, "
            "target retention, status, and primary classification."
        ),
        body="\n".join(pieces),
    )


def _mechanisms_svg(
    runs: tuple[CLIRun, ...],
    fixture_payloads: dict[str, bytes],
) -> bytes:
    by_id = {run.identifier: run for run in runs}
    merge = by_id["merge-cross-boundary"].report
    nfc = by_id["nfc-cross-boundary"].report
    strip = by_id["right-strip-drift"].report
    away = by_id["normalized-away"].report

    merge_attr = _mapping(
        _attribution(merge, "full")[1],
        "report.attribution",
    )
    nfc_attr = _mapping(_attribution(nfc, "full")[1], "report.attribution")
    if (
        _sequence(_encoding(merge, "combined_full").get("pieces"), "pieces")
        != ["<bos>", "ab"]
        or merge_attr.get("ownership") != "cross_boundary"
        or merge_attr.get("raw_origins") != [0, 1]
        or _normalization(nfc, "combined").get("cross_boundary_output_positions") != [0]
        or _sequence(_encoding(nfc, "combined_full").get("pieces"), "pieces")
        != ["<bos>", "é", "x"]
        or nfc_attr.get("raw_origins") != [0, 1]
        or _diagnostics(strip).get("prompt_leakage_positions") != [2]
        or _diagnostics(away).get("indeterminate_reasons")
        != ["target.no_attributable_token"]
    ):
        raise EvidenceError("render.mechanism_expectation")

    width = 1000
    height = 1230
    pieces = [
        _line(
            x=46,
            y=60,
            text="How independent tokenization shifts a boundary",
            css_class="sans title",
        ),
        _line(
            x=46,
            y=94,
            text="Four executed mechanisms · raw authored boundary shown with |",
            css_class="sans subtitle",
        ),
    ]
    cards = (
        (
            "BPE merge across boundary",
            "a | b",
            "standalone: <bos> a",
            "combined: <bos> ab",
            "token 1 owns origins [0,1] → cross_boundary; legacy cutoff 2 masks it",
            "#9333ea",
        ),
        (
            "NFC composition across boundary",
            "e | U+0301x",
            "standalone: <bos> e",
            "combined: <bos> é x",
            "normalized output 0 and token 1 own origins [0,1]; x remains target",
            "#7c3aed",
        ),
        (
            "Right-strip context drift",
            "c SPACE | d",
            "standalone right-strip: c SPACE → c",
            "combined: SPACE is internal → <bos> c SPACE d",
            "legacy cutoff 2 vs oracle 3 → source token 2 is supervised",
            "#2563eb",
        ),
        (
            "Target normalized away",
            "c | SPACE",
            "combined right-strip removes one trailing normalized code point",
            "combined tokens: <bos> c",
            "0 attributable target tokens → indeterminate, not an aligned pass",
            "#b45309",
        ),
    )
    y = 130
    for title, boundary, line_one, line_two, result, color in cards:
        pieces.extend(
            [
                f'  <rect x="46" y="{y}" width="908" height="242" rx="16" fill="#0d1b2e" stroke="{color}" stroke-width="2"/>',
                _line(
                    x=72,
                    y=y + 40,
                    text=title,
                    css_class="sans heading",
                ),
                f'  <rect x="72" y="{y + 58}" width="856" height="50" rx="10" fill="#111f34" stroke="#475569"/>',
                _line(
                    x=500,
                    y=y + 91,
                    text=boundary,
                    css_class="mono heading",
                    anchor="middle",
                ),
                _line(
                    x=72,
                    y=y + 140,
                    text=line_one,
                    css_class="mono body",
                ),
                _line(
                    x=72,
                    y=y + 172,
                    text=line_two,
                    css_class="mono body",
                ),
                _line(
                    x=72,
                    y=y + 211,
                    text=result,
                    css_class="sans label",
                ),
            ]
        )
        y += 258
    pieces.extend(
        [
            _line(
                x=46,
                y=1180,
                text=(
                    "Counts, pieces, raw origins, cutoffs, and reasons come "
                    "from the seven canonical CLI reports."
                ),
                css_class="sans small",
            ),
            _line(
                x=46,
                y=1212,
                text=(
                    "These authored cases isolate mechanisms; they do not "
                    "measure production prevalence or model impact."
                ),
                css_class="sans tiny",
            ),
        ]
    )
    del fixture_payloads
    return _svg_document(
        identifier="mechanisms",
        width=width,
        height=height,
        title="How independent tokenization shifts a boundary",
        description=(
            "Four source-derived panels show an executed BPE merge, NFC "
            "composition, right-strip context drift, and a target removed by "
            "normalization."
        ),
        body="\n".join(pieces),
    )


def _matrix_code(primary: str) -> str:
    return {
        "aligned": "A",
        "boundary_drift": "D",
        "cross_boundary_token": "X",
        "partial_target_truncation": "P",
        "target_eliminated": "E",
        "indeterminate": "?",
    }.get(primary, "?")


def _matrix_svg(runs: tuple[SweepRun, ...]) -> bytes:
    by_key = {(run.base_identifier, run.max_length): run for run in runs}
    expected_count = len(FIXTURE_PATHS) * len(SWEEP_LENGTHS)
    if len(by_key) != expected_count:
        raise EvidenceError("render.matrix")
    width = 1000
    panel_height = 650
    height = 1540
    label_width = 245
    cell_width = 168
    cell_height = 70
    colors = {
        "A": "#0f766e",
        "D": "#1d4ed8",
        "X": "#7e22ce",
        "P": "#a16207",
        "E": "#9f1239",
        "?": "#475569",
    }
    pieces = [
        _line(
            x=46,
            y=60,
            text="Target retention across max_length",
            css_class="sans title",
        ),
        _line(
            x=46,
            y=94,
            text=(
                "56 fresh stdin CLI executions · seven authored cases · "
                "direct labels show primary + retained/full"
            ),
            css_class="sans subtitle",
        ),
        '  <rect x="46" y="116" width="908" height="68" rx="12" fill="#10233f" stroke="#475569"/>',
        _line(
            x=66,
            y=143,
            text="A aligned · D drift · X cross-boundary · P partial · E eliminated · ? indeterminate",
            css_class="sans small",
        ),
        _line(
            x=66,
            y=168,
            text="Color is redundant: every cell contains its code and exact retained/full target count.",
            css_class="sans small",
        ),
    ]
    identifiers = tuple(EXPECTED_RUNS)
    for panel_index, lengths in enumerate((SWEEP_LENGTHS[:4], SWEEP_LENGTHS[4:])):
        panel_y = 220 + panel_index * panel_height
        pieces.append(
            _line(
                x=46,
                y=panel_y,
                text=(
                    f"max_length {lengths[0]}–{lengths[-1]} · panel {panel_index + 1}/2"
                ),
                css_class="sans heading",
            )
        )
        for column, max_length in enumerate(lengths):
            x = label_width + column * cell_width
            pieces.append(
                _line(
                    x=x + (cell_width - 8) // 2,
                    y=panel_y + 40,
                    text=str(max_length),
                    css_class="mono label",
                    anchor="middle",
                )
            )
        for row, identifier in enumerate(identifiers):
            y = panel_y + 58 + row * cell_height
            pieces.append(
                _line(
                    x=46,
                    y=y + 42,
                    text=identifier,
                    css_class="mono small",
                )
            )
            for column, max_length in enumerate(lengths):
                run = by_key[(identifier, max_length)]
                diagnostics = _diagnostics(run.report)
                primary = _string(
                    run.report.get("primary_classification"),
                    "report.primary",
                )
                code = _matrix_code(primary)
                retained = _integer(
                    diagnostics.get("retained_target_token_count"),
                    "report.target_count",
                )
                full = _integer(
                    diagnostics.get("full_target_token_count"),
                    "report.target_count",
                )
                x = label_width + column * cell_width
                pieces.extend(
                    [
                        f'  <rect x="{x}" y="{y}" width="{cell_width - 8}" height="{cell_height - 8}" rx="9" fill="{colors[code]}" stroke="#cbd5e1"/>',
                        _line(
                            x=x + (cell_width - 8) // 2,
                            y=y + 27,
                            text=f"{code} · {retained}/{full}",
                            css_class="mono label",
                            anchor="middle",
                        ),
                        _line(
                            x=x + (cell_width - 8) // 2,
                            y=y + 50,
                            text=(f"exit {run.returncode} · {run.report['status']}"),
                            css_class="mono tiny",
                            anchor="middle",
                        ),
                    ]
                )
    pieces.extend(
        [
            _line(
                x=46,
                y=1480,
                text=(
                    "This bounded sweep varies only max_length; source, target, "
                    "artifact, runtime, and analysis code stay fixed."
                ),
                css_class="sans small",
            ),
            _line(
                x=46,
                y=1504,
                text=(
                    "Inspect truncation-sweep.csv for every canonical input "
                    "identity, report hash, cutoff, and target count."
                ),
                css_class="sans tiny",
            ),
        ]
    )
    return _svg_document(
        identifier="matrix",
        width=width,
        height=height,
        title="Target retention across max_length",
        description=(
            "A two-panel matrix of 56 fresh CLI executions across seven "
            "authored boundary cases and max lengths two through nine. Each "
            "cell gives the primary classification and retained versus full "
            "target-token count."
        ),
        body="\n".join(pieces),
    )


def _transcript_svg(runs: tuple[CLIRun, ...]) -> bytes:
    width = 1000
    height = 1210
    pieces = [
        '  <rect x="38" y="34" width="924" height="1112" rx="18" fill="#030812" stroke="#475569" stroke-width="2"/>',
        '  <circle cx="72" cy="68" r="7" fill="#e87979"/>',
        '  <circle cx="98" cy="68" r="7" fill="#eab85d"/>',
        '  <circle cx="124" cy="68" r="7" fill="#66c2a5"/>',
        _line(
            x=500,
            y=74,
            text="real stdin subprocess capture",
            css_class="mono small",
            anchor="middle",
        ),
        '  <line x1="38" y1="94" x2="962" y2="94" stroke="#334155"/>',
        _line(
            x=66,
            y=128,
            text="TOKEN BOUNDARY LAB / EXECUTED FIXTURE INDEX",
            css_class="mono label",
        ),
        _line(
            x=66,
            y=157,
            text=(
                "Full canonical stdout is preserved in cli-session.txt and "
                "seven report files."
            ),
            css_class="sans small",
        ),
    ]
    y = 194
    for run in runs:
        pieces.extend(
            [
                f'  <rect x="62" y="{y}" width="876" height="116" rx="12" fill="#0d1b2e" stroke="#334155"/>',
                _line(
                    x=82,
                    y=y + 30,
                    text=(
                        f"$ python3 -m token_boundary.cli < {run.display_input_path}"
                    ),
                    css_class="mono small",
                    fill="#93c5fd",
                ),
                _line(
                    x=82,
                    y=y + 61,
                    text=(
                        f"exit {run.returncode} · {run.report['status']} · "
                        f"{run.report['primary_classification']}"
                    ),
                    css_class="sans label",
                ),
                _line(
                    x=82,
                    y=y + 89,
                    text=(
                        f"stdout {len(run.stdout)} bytes · "
                        f"sha256 {_sha256(run.stdout)[:16]}… · stderr empty"
                    ),
                    css_class="mono tiny",
                ),
            ]
        )
        y += 130
    pieces.extend(
        [
            _line(
                x=66,
                y=1130,
                text=(
                    "Exit 1 is a valid complete fail/indeterminate report; "
                    "exit 2 is reserved for process-boundary errors."
                ),
                css_class="sans small",
            ),
            _line(
                x=38,
                y=1185,
                text=(
                    "Source-generated index of actual runs; no hostname, "
                    "absolute path, timestamp, secret, or personal data."
                ),
                css_class="sans tiny",
            ),
        ]
    )
    return _svg_document(
        identifier="cli",
        width=width,
        height=height,
        title="Token-boundary real stdin subprocess capture",
        description=(
            "A terminal-style source-generated index of all seven actual "
            "stdin CLI runs, including command, exit status, report status, "
            "primary classification, output bytes, hash prefix, and empty "
            "stderr state."
        ),
        body="\n".join(pieces),
    )


def _source_entries(
    source_payloads: dict[str, bytes],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in SOURCE_PATHS:
        payload = source_payloads[path]
        entries.append(
            {
                "bytes": len(payload),
                "path": path,
                "sha256": _sha256(payload),
            }
        )
    return entries


def _artifact_entries(
    artifacts: dict[str, bytes],
) -> list[dict[str, object]]:
    prefix = "/".join(OUTPUT_PARTS)
    return [
        {
            "bytes": len(artifacts[name]),
            "path": f"{prefix}/{name}",
            "sha256": _sha256(artifacts[name]),
        }
        for name in ARTIFACT_NAMES
    ]


def _cli_entries(runs: tuple[CLIRun, ...]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    prefix = "/".join(OUTPUT_PARTS)
    for run in runs:
        identity = _input_identity(run.report)
        entries.append(
            {
                "argv": list(run.argv),
                "canonical_input_sha256": _string(
                    identity.get("canonical_sha256"),
                    "report.input_identity",
                ),
                "expected_exit_code": EXPECTED_RUNS[run.identifier]["exit"],
                "expected_primary_classification": EXPECTED_RUNS[run.identifier][
                    "primary"
                ],
                "expected_status": EXPECTED_RUNS[run.identifier]["status"],
                "fixture_path": run.fixture_path,
                "fixture_raw_sha256": _sha256(run.input_bytes),
                "id": run.identifier,
                "observed_exit_code": run.returncode,
                "observed_primary_classification": _string(
                    run.report.get("primary_classification"),
                    "report.primary",
                ),
                "observed_status": _string(
                    run.report.get("status"),
                    "report.status",
                ),
                "report_bytes": len(run.stdout),
                "report_path": f"{prefix}/{REPORT_NAMES[run.identifier]}",
                "report_sha256": _sha256(run.stdout),
                "stderr_bytes": len(run.stderr),
                "stderr_sha256": _sha256(run.stderr),
                "stdout_is_canonical_report": True,
            }
        )
    return entries


def _sweep_entries(runs: tuple[SweepRun, ...]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for run in runs:
        diagnostics = _diagnostics(run.report)
        identity = _input_identity(run.report)
        legacy = _legacy_mask(run.report)
        entries.append(
            {
                "base_fixture_path": run.base_fixture_path,
                "canonical_input_sha256": _string(
                    identity.get("canonical_sha256"),
                    "report.input_identity",
                ),
                "case_id": run.base_identifier,
                "cross_boundary_positions": _sequence(
                    diagnostics.get("cross_boundary_positions"),
                    "report.cross_boundary_positions",
                ),
                "elimination_causes": _sequence(
                    diagnostics.get("elimination_causes"),
                    "report.elimination_causes",
                ),
                "exit_code": run.returncode,
                "fixture_raw_sha256": run.fixture_raw_sha256,
                "full_target_token_count": _integer(
                    diagnostics.get("full_target_token_count"),
                    "report.target_count",
                ),
                "indeterminate_reasons": _sequence(
                    diagnostics.get("indeterminate_reasons"),
                    "report.indeterminate_reasons",
                ),
                "legacy_cutoff": _integer(
                    legacy.get("cutoff"),
                    "report.cutoff",
                ),
                "max_length": run.max_length,
                "oracle_cutoff": _optional_integer(
                    diagnostics.get("oracle_cutoff"),
                    "report.oracle_cutoff",
                ),
                "primary_classification": _string(
                    run.report.get("primary_classification"),
                    "report.primary",
                ),
                "prompt_leakage_positions": _sequence(
                    diagnostics.get("prompt_leakage_positions"),
                    "report.prompt_leakage_positions",
                ),
                "report_status": _string(
                    run.report.get("status"),
                    "report.status",
                ),
                "retained_target_token_count": _integer(
                    diagnostics.get("retained_target_token_count"),
                    "report.target_count",
                ),
                "stdout_bytes": len(run.stdout),
                "stdout_sha256": _sha256(run.stdout),
                "supervised_target_token_count": _integer(
                    diagnostics.get("supervised_target_token_count"),
                    "report.target_count",
                ),
                "truncated_target_token_count": _integer(
                    diagnostics.get("truncated_target_token_count"),
                    "report.target_count",
                ),
            }
        )
    return entries


def _legacy_binding(
    legacy_manifest: dict[str, object],
    legacy_manifest_bytes: bytes,
    trainer_bytes: bytes,
) -> dict[str, object]:
    snapshot = _mapping(legacy_manifest.get("snapshot"), "legacy.snapshot")
    files = _sequence(legacy_manifest.get("files"), "legacy.files")
    trainer: dict[str, object] | None = None
    for value in files:
        entry = _mapping(value, "legacy.file")
        if entry.get("path") == "finetune.py":
            trainer = entry
            break
    if trainer is None:
        raise EvidenceError("legacy.trainer")
    if (
        _integer(trainer.get("byte_length"), "legacy.file") != len(trainer_bytes)
        or _string(trainer.get("sha256"), "legacy.file") != _sha256(trainer_bytes)
        or _string(trainer.get("git_blob_sha1"), "legacy.file")
        != _git_blob_sha1(trainer_bytes)
    ):
        raise EvidenceError("legacy.source_drift")
    return {
        "algorithm_id": "standalone-source-token-count",
        "attestation_path": "provenance/legacy-snapshot.v1.json",
        "attestation_sha256": _sha256(legacy_manifest_bytes),
        "semantic_equivalence_claimed": False,
        "snapshot_source_git_commit": _string(
            snapshot.get("source_git_commit"),
            "legacy.snapshot",
        ),
        "source_git_blob_sha1": _string(
            trainer.get("git_blob_sha1"),
            "legacy.file",
        ),
        "source_imported_or_executed": False,
        "source_path": "finetune.py",
        "source_sha256": _string(trainer.get("sha256"), "legacy.file"),
    }


def _manifest_bytes(
    *,
    source_payloads: dict[str, bytes],
    runs: tuple[CLIRun, ...],
    sweep_runs: tuple[SweepRun, ...],
    sweep_bytes: bytes,
    legacy_manifest: dict[str, object],
    legacy_manifest_bytes: bytes,
    artifacts: dict[str, bytes],
) -> bytes:
    artifact_identity = _mapping(
        runs[0].report.get("artifact_identity"),
        "report.artifact_identity",
    )
    for fixture_run in runs[1:]:
        if fixture_run.report.get("artifact_identity") != artifact_identity:
            raise EvidenceError("runtime.artifact_identity")
    classification_counts: dict[str, int] = {}
    for sweep_run in sweep_runs:
        primary = _string(
            sweep_run.report.get("primary_classification"),
            "report.primary",
        )
        classification_counts[primary] = classification_counts.get(primary, 0) + 1
    value = {
        "artifacts": _artifact_entries(artifacts),
        "cli_executions": _cli_entries(runs),
        "kind": "token-boundary.visual-evidence-manifest",
        "legacy_algorithm_binding": _legacy_binding(
            legacy_manifest,
            legacy_manifest_bytes,
            source_payloads["finetune.py"],
        ),
        "runtime": {
            "artifact_bytes": _integer(
                artifact_identity.get("byte_count"),
                "report.artifact_identity",
            ),
            "artifact_id": _string(
                artifact_identity.get("artifact_id"),
                "report.artifact_identity",
            ),
            "artifact_sha256": _string(
                artifact_identity.get("sha256"),
                "report.artifact_identity",
            ),
            "engine": _string(
                artifact_identity.get("engine"),
                "report.artifact_identity",
            ),
            "engine_version": _string(
                artifact_identity.get("engine_version"),
                "report.artifact_identity",
            ),
            "normalizers": _sequence(
                artifact_identity.get("normalizers"),
                "report.artifact_identity",
            ),
            "post_processor": _string(
                artifact_identity.get("post_processor"),
                "report.artifact_identity",
            ),
            "truncation": _string(
                artifact_identity.get("truncation"),
                "report.artifact_identity",
            ),
        },
        "schema_version": 1,
        "scope": {
            "contains_personal_data": False,
            "contains_secrets": False,
            "deepseek_model_executed": False,
            "deepseek_tokenizer_attested": False,
            "fixture_class": (
                "seven committed authored synthetic source-target boundaries"
            ),
            "forward_pass_or_loss_computed": False,
            "gpu_used": False,
            "inherited_trainer_executed": False,
            "inherited_trainer_fixed": False,
            "local_synthetic_tokenizer_artifact_executed": True,
            "model_or_dataset_loaded": False,
            "model_quality_measured": False,
            "model_training_or_evaluation_performed": False,
            "network_used": False,
            "prevalence_measured": False,
            "sweep_is_benchmark_or_sample": False,
            "universal_masking_policy_claimed": False,
        },
        "sources": _source_entries(source_payloads),
        "truncation_sweep": {
            "classification_counts": dict(sorted(classification_counts.items())),
            "csv_bytes": len(sweep_bytes),
            "csv_path": (f"{'/'.join(OUTPUT_PARTS)}/{SWEEP_NAME}"),
            "csv_sha256": _sha256(sweep_bytes),
            "execution_boundary": ("fresh public stdin CLI subprocess for every cell"),
            "executions": _sweep_entries(sweep_runs),
            "max_lengths": list(SWEEP_LENGTHS),
            "run_count": len(sweep_runs),
        },
    }
    return _json_bytes(value)


def _load_legacy_manifest(
    payload: bytes,
) -> dict[str, object]:
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise EvidenceError("legacy.document")
    if type(decoded) is not dict:
        raise EvidenceError("legacy.document")
    assert isinstance(decoded, dict)
    if (
        decoded.get("kind") != "legacy-source-attestation"
        or decoded.get("schema_version") != 1
    ):
        raise EvidenceError("legacy.document")
    return decoded


def _build_bundle(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, bytes]:
    source_payloads = {
        path: _read_source(repository_root, path) for path in SOURCE_PATHS
    }
    fixture_payloads = {path: source_payloads[path] for path in FIXTURE_PATHS}
    for path, payload in fixture_payloads.items():
        if _sha256(payload) != EXPECTED_FIXTURE_SHA256[path]:
            raise EvidenceError("evidence.fixture_identity")
    legacy_manifest_bytes = source_payloads["provenance/legacy-snapshot.v1.json"]
    legacy_manifest = _load_legacy_manifest(legacy_manifest_bytes)
    try:
        temporary = tempfile.TemporaryDirectory(
            prefix=".token-boundary-evidence-run-",
            dir=repository_root,
        )
    except OSError:
        raise EvidenceError("io.sandbox_unavailable")
    with temporary as temporary_name:
        sandbox = Path(temporary_name)
        runs = _run_fixture_suite(
            repository_root,
            fixture_payloads,
            sandbox,
        )
        artifact_identity = _mapping(
            runs[0].report.get("artifact_identity"),
            "report.artifact_identity",
        )
        if any(
            run.report.get("artifact_identity") != artifact_identity for run in runs[1:]
        ):
            raise EvidenceError("runtime.artifact_identity")
        sweep_runs = _run_sweep(
            repository_root,
            fixture_payloads,
            sandbox,
            artifact_identity,
        )

    transcript = _transcript_bytes(runs)
    sweep = _sweep_bytes(sweep_runs)
    artifacts: dict[str, bytes] = {
        ARCHITECTURE_SVG_NAME: _architecture_svg(runs, legacy_manifest),
        MECHANISMS_SVG_NAME: _mechanisms_svg(runs, fixture_payloads),
        TRANSCRIPT_SVG_NAME: _transcript_svg(runs),
        TRANSCRIPT_NAME: transcript,
    }
    artifacts.update((REPORT_NAMES[run.identifier], run.stdout) for run in runs)
    artifacts.update(
        {
            LANES_SVG_NAME: _lanes_svg(runs, fixture_payloads),
            MATRIX_SVG_NAME: _matrix_svg(sweep_runs),
            SWEEP_NAME: sweep,
        }
    )
    if tuple(artifacts) != ARTIFACT_NAMES:
        raise EvidenceError("bundle.artifact_order")
    manifest = _manifest_bytes(
        source_payloads=source_payloads,
        runs=runs,
        sweep_runs=sweep_runs,
        sweep_bytes=sweep,
        legacy_manifest=legacy_manifest,
        legacy_manifest_bytes=legacy_manifest_bytes,
        artifacts=artifacts,
    )
    return {**artifacts, MANIFEST_NAME: manifest}


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise EvidenceError("cli.arguments")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="render-token-boundary-evidence",
        description=(
            "Generate or verify evidence for the seven fixed synthetic "
            "token-boundary fixtures."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser


def _error_bytes(code: str) -> bytes:
    return (
        json.dumps(
            {
                "error": {"code": code},
                "kind": "token-boundary.evidence-error",
                "status": "error",
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _emit_error(code: str) -> None:
    try:
        sys.stderr.buffer.write(_error_bytes(code))
    except (AttributeError, OSError, ValueError):
        pass
    # There is no safer fallback channel after stderr itself fails.
    except Exception:  # noqa: BLE001, S110
        pass


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
        summary = {
            "artifacts": len(ARTIFACT_NAMES),
            "cli_runs": len(EXPECTED_RUNS),
            "mode": mode,
            "status": "ok",
            "sweep_runs": len(FIXTURE_PATHS) * len(SWEEP_LENGTHS),
        }
        sys.stdout.write(
            json.dumps(summary, separators=(",", ":"), sort_keys=True) + "\n"
        )
        return 0
    except BoundaryEngineError as error:
        code = (
            error.code
            if error.code in {"runtime.unavailable", "runtime.version"}
            else "runtime.analysis_failed"
        )
        _emit_error(code)
        return 2
    except EvidenceError as error:
        _emit_error(error.code)
        return 2
    except KeyboardInterrupt:
        _emit_error("cli.interrupted")
        return 2
    # Do not serialize unexpected runtime, parser, or filesystem exceptions.
    except Exception:  # noqa: BLE001
        _emit_error("internal.error")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
