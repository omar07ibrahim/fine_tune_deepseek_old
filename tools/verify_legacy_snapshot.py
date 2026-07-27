#!/usr/bin/env python3
"""Verify the byte-level boundary around the inherited training snapshot.

The verifier is deliberately standard-library-only. It never imports the
legacy trainer, accesses the network, loads model code, or reads a dataset.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any


MANIFEST_PATH = PurePosixPath("provenance/legacy-snapshot.v1.json")
EXPECTED_LEGACY_PATHS = (
    "configs/ds_config_zero3.json",
    "finetune.py",
    "requirements.txt",
)
EXPECTED_LEGACY_IDENTITIES = {
    "configs/ds_config_zero3.json": {
        "byte_length": 1299,
        "line_count": 51,
        "git_mode": "100644",
        "sha256": (
            "c0efd31093f7d6e2ca8ea5fcd77e28a7aff6c3306893af56622850faa104d6a7"
        ),
        "git_blob_sha1": "a90c20c916c762a8b7d977d53b49bb617318bb5a",
    },
    "finetune.py": {
        "byte_length": 8667,
        "line_count": 230,
        "git_mode": "100644",
        "sha256": (
            "5de3316c8cf37edea97e83230fd90bf01092582dc25016c87fdc404aa1024e26"
        ),
        "git_blob_sha1": "d334caa2cec91ed97a1974cdf61e3ab0d3edf415",
    },
    "requirements.txt": {
        "byte_length": 111,
        "line_count": 10,
        "git_mode": "100644",
        "sha256": (
            "da3b0ecac531b2f304528f5778c136eef38b8deca8e7dea548c95e7a40e8ea77"
        ),
        "git_blob_sha1": "f6e03b76439065c4528f8ebbd2276388ab65d6ba",
    },
}
EXPECTED_OBSERVATION_IDS = ("trainer-line-lcs", "zero3-semantic-json")
EXPECTED_SOURCE_GIT_COMMIT = "6912653d881bedee71ef527bc5650db55f115779"
EXPECTED_TRAINER_LINE_COUNT = 230
EXPECTED_TRAINER_LCS_LINE_COUNT = 185
EXPECTED_ZERO3_SEMANTIC_SHA256 = (
    "ac305ab8aba093eb0a29f94629baf3c89ca266077f1246ae89a42b3648aaf23e"
)
EXPECTED_UPSTREAM_REPOSITORY = "https://github.com/deepseek-ai/DeepSeek-MoE"
EXPECTED_UPSTREAM_GIT_COMMIT = "66edeee5a4f75cbd76e0316229ad101805a90e01"
EXPECTED_UPSTREAM_FILE_IDENTITIES = (
    {
        "path": "finetune/finetune.py",
        "byte_length": 13274,
        "git_blob_sha1": "244ff4e11416ab4687df3a7343df36846ff45e81",
    },
    {
        "path": "finetune/configs/ds_config_zero3.json",
        "byte_length": 1348,
        "git_blob_sha1": "73f3b5f4c430d1ff5ab5ac9e82c11f436440e728",
    },
    {
        "path": "LICENSE-CODE",
        "byte_length": 1065,
        "git_blob_sha1": "d84f527e101b2cdd171e2b14253f84ea4fedabe9",
    },
)
EXPECTED_LICENSE_REFERENCE = (
    "https://github.com/deepseek-ai/DeepSeek-MoE/blob/"
    f"{EXPECTED_UPSTREAM_GIT_COMMIT}/LICENSE-CODE"
)
EXPECTED_NOTICE_PATH = "third_party/deepseek-moe/LICENSE-CODE"
EXPECTED_NOTICE_SHA256 = (
    "6e4c38e1172f42fdbff13edf9a7a017679fb82b0fde415a3e8b3c31c6ed4a4e4"
)
EXPECTED_NOTICE_BYTES = 1065
MAX_MANIFEST_BYTES = 128 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SHA1_RE = re.compile(r"[0-9a-f]{40}")


class AttestationError(RuntimeError):
    """Raised when an attested invariant is missing, ambiguous, or changed."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _decode_json(payload: bytes, *, decimals: bool = False) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AttestationError("JSON input is not valid UTF-8") from exc

    options: dict[str, Any] = {
        "object_pairs_hook": _unique_object,
        "parse_constant": _reject_constant,
    }
    if decimals:
        options["parse_float"] = Decimal

    try:
        return json.loads(text, **options)
    except (json.JSONDecodeError, ValueError) as exc:
        raise AttestationError(f"invalid or ambiguous JSON: {exc}") from exc


def _expect_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AttestationError(f"{label} must be a JSON object")
    return value


def _expect_sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AttestationError(f"{label} must be a JSON array")
    return value


def _expect_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise AttestationError(
            f"{label} has an unexpected schema "
            f"(missing={missing}, unexpected={unexpected})"
        )


def _expect_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AttestationError(f"{label} must be an integer >= {minimum}")
    return value


def _expect_sha(value: Any, label: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise AttestationError(f"{label} is not a lowercase hexadecimal digest")
    return value


def _safe_relative_path(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str):
        raise AttestationError(f"{label} must be a string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in value
    ):
        raise AttestationError(f"{label} is not a safe repository-relative path")
    return path


def _read_regular_file(
    root: Path,
    relative: PurePosixPath,
    *,
    byte_limit: int,
) -> bytes:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise AttestationError(f"{relative.as_posix()} must not use symlinks")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(current, flags)
    except OSError as exc:
        raise AttestationError(
            f"cannot open {relative.as_posix()} as a regular file"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise AttestationError(
                f"{relative.as_posix()} is not a regular file"
            )
        if metadata.st_size > byte_limit:
            raise AttestationError(
                f"{relative.as_posix()} exceeds the {byte_limit}-byte read limit"
            )
        chunks: list[bytes] = []
        remaining = byte_limit + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > byte_limit:
            raise AttestationError(
                f"{relative.as_posix()} exceeds the {byte_limit}-byte read limit"
            )
        if len(payload) != metadata.st_size:
            raise AttestationError(
                f"{relative.as_posix()} changed while being read"
            )
    except OSError as exc:
        raise AttestationError(
            f"cannot read {relative.as_posix()} as a regular file"
        ) from exc
    finally:
        os.close(descriptor)
    return payload


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode()
    try:
        digest = hashlib.sha1(usedforsecurity=False)
    except TypeError:  # pragma: no cover - compatibility with older Python.
        digest = hashlib.sha1()
    digest.update(header)
    digest.update(payload)
    return digest.hexdigest()


def _git_file_mode(root: Path, relative: PurePosixPath) -> str:
    mode = (root.joinpath(*relative.parts).stat().st_mode)
    return "100755" if mode & 0o111 else "100644"


def _canonical_number(value: Decimal) -> str:
    if not value.is_finite():
        raise AttestationError("semantic JSON contains a non-finite number")
    if value.is_zero():
        return "0"
    normalized = value.normalize()
    if abs(normalized) >= Decimal("1e9"):
        return format(normalized, "E")
    rendered = format(normalized, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _canonical_json(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return _canonical_number(Decimal(value))
    if isinstance(value, Decimal):
        return _canonical_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ",".join(_canonical_json(item) for item in value) + "]"
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise AttestationError("semantic JSON object keys must be strings")
        fields = (
            f"{json.dumps(key, ensure_ascii=False)}:{_canonical_json(value[key])}"
            for key in sorted(value)
        )
        return "{" + ",".join(fields) + "}"
    raise AttestationError(
        f"semantic JSON contains unsupported value type: {type(value).__name__}"
    )


def semantic_json_sha256(payload: bytes) -> str:
    """Return the v1 sorted, compact, newline-terminated semantic JSON hash.

    V1 rejects duplicate keys and non-finite numbers, sorts object keys,
    renders strings as unescaped UTF-8 where JSON permits, normalizes negative
    zero, and uses uppercase scientific notation at magnitudes of 1e9 or
    greater. That explicit rule reproduces the recorded audit digest without a
    runtime dependency on ``jq``.
    """

    parsed = _decode_json(payload, decimals=True)
    canonical = (_canonical_json(parsed) + "\n").encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_manifest(manifest: Any) -> Mapping[str, Any]:
    root = _expect_mapping(manifest, "manifest")
    _expect_exact_keys(
        root,
        {"schema_version", "kind", "snapshot", "files", "provenance"},
        "manifest",
    )
    if root["schema_version"] != 1:
        raise AttestationError("manifest.schema_version must be exactly 1")
    if root["kind"] != "legacy-source-attestation":
        raise AttestationError("manifest.kind is not supported")

    snapshot = _expect_mapping(root["snapshot"], "manifest.snapshot")
    _expect_exact_keys(
        snapshot,
        {"source_git_commit", "scope", "legacy_state", "supported_entrypoint"},
        "manifest.snapshot",
    )
    source_git_commit = _expect_sha(
        snapshot["source_git_commit"],
        "manifest.snapshot.source_git_commit",
        SHA1_RE,
    )
    if source_git_commit != EXPECTED_SOURCE_GIT_COMMIT:
        raise AttestationError("unexpected source Git commit")
    if not isinstance(snapshot["scope"], str) or not snapshot["scope"].strip():
        raise AttestationError("manifest.snapshot.scope must be non-empty")
    if snapshot["legacy_state"] != "quarantined":
        raise AttestationError("manifest.snapshot.legacy_state must be quarantined")
    if snapshot["supported_entrypoint"] is not None:
        raise AttestationError(
            "the inherited snapshot must not declare a supported entrypoint"
        )

    files = _expect_sequence(root["files"], "manifest.files")
    paths: list[str] = []
    for index, raw_entry in enumerate(files):
        label = f"manifest.files[{index}]"
        entry = _expect_mapping(raw_entry, label)
        _expect_exact_keys(
            entry,
            {
                "path",
                "byte_length",
                "line_count",
                "git_mode",
                "sha256",
                "git_blob_sha1",
            },
            label,
        )
        paths.append(_safe_relative_path(entry["path"], f"{label}.path").as_posix())
        _expect_int(entry["byte_length"], f"{label}.byte_length", minimum=1)
        _expect_int(entry["line_count"], f"{label}.line_count", minimum=1)
        if entry["git_mode"] != "100644":
            raise AttestationError(f"{label}.git_mode must be exactly 100644")
        _expect_sha(entry["sha256"], f"{label}.sha256", SHA256_RE)
        _expect_sha(entry["git_blob_sha1"], f"{label}.git_blob_sha1", SHA1_RE)
    if tuple(paths) != EXPECTED_LEGACY_PATHS:
        raise AttestationError(
            "manifest.files must contain the exact ordered legacy path set"
        )
    for index, raw_entry in enumerate(files):
        entry = _expect_mapping(raw_entry, f"manifest.files[{index}]")
        path = paths[index]
        expected_identity = EXPECTED_LEGACY_IDENTITIES[path]
        actual_identity = {
            "byte_length": entry["byte_length"],
            "line_count": entry["line_count"],
            "git_mode": entry["git_mode"],
            "sha256": entry["sha256"],
            "git_blob_sha1": entry["git_blob_sha1"],
        }
        if actual_identity != expected_identity:
            raise AttestationError(
                "manifest legacy identity differs from the trusted snapshot"
            )

    provenance = _expect_mapping(root["provenance"], "manifest.provenance")
    _expect_exact_keys(
        provenance,
        {
            "upstream_project",
            "upstream_repository",
            "upstream_git_commit",
            "upstream_file_identities",
            "upstream_code_license",
            "model_license_boundary",
            "dataset_license_boundary",
            "observations",
        },
        "manifest.provenance",
    )
    if provenance["upstream_project"] != "DeepSeek-MoE":
        raise AttestationError("unexpected upstream project")
    if provenance["upstream_repository"] != EXPECTED_UPSTREAM_REPOSITORY:
        raise AttestationError("unexpected upstream repository")
    if provenance["upstream_git_commit"] != EXPECTED_UPSTREAM_GIT_COMMIT:
        raise AttestationError("unexpected upstream Git commit")
    upstream_identities = _expect_sequence(
        provenance["upstream_file_identities"],
        "manifest.provenance.upstream_file_identities",
    )
    normalized_upstream: list[dict[str, Any]] = []
    for index, raw_identity in enumerate(upstream_identities):
        label = f"manifest.provenance.upstream_file_identities[{index}]"
        identity = _expect_mapping(raw_identity, label)
        _expect_exact_keys(
            identity,
            {"path", "byte_length", "git_blob_sha1"},
            label,
        )
        normalized_upstream.append(
            {
                "path": _safe_relative_path(
                    identity["path"],
                    f"{label}.path",
                ).as_posix(),
                "byte_length": _expect_int(
                    identity["byte_length"],
                    f"{label}.byte_length",
                    minimum=1,
                ),
                "git_blob_sha1": _expect_sha(
                    identity["git_blob_sha1"],
                    f"{label}.git_blob_sha1",
                    SHA1_RE,
                ),
            }
        )
    if tuple(normalized_upstream) != EXPECTED_UPSTREAM_FILE_IDENTITIES:
        raise AttestationError("upstream file identities changed")
    if (
        provenance["model_license_boundary"]
        != "separate-terms-not-vendored-or-attested"
    ):
        raise AttestationError("model-license boundary is missing or ambiguous")
    if (
        provenance["dataset_license_boundary"]
        != "no-dataset-vendored-or-attested"
    ):
        raise AttestationError("dataset-license boundary is missing or ambiguous")

    license_record = _expect_mapping(
        provenance["upstream_code_license"],
        "manifest.provenance.upstream_code_license",
    )
    _expect_exact_keys(
        license_record,
        {
            "name",
            "reference",
            "local_notice_path",
            "local_notice_byte_length",
            "local_notice_sha256",
        },
        "manifest.provenance.upstream_code_license",
    )
    if license_record["name"] != "MIT":
        raise AttestationError("upstream code license must be recorded as MIT")
    if license_record["reference"] != EXPECTED_LICENSE_REFERENCE:
        raise AttestationError("unexpected upstream code-license reference")
    if license_record["local_notice_path"] != EXPECTED_NOTICE_PATH:
        raise AttestationError("unexpected local third-party notice path")
    notice_length = _expect_int(
        license_record["local_notice_byte_length"],
        "manifest.provenance.upstream_code_license.local_notice_byte_length",
        minimum=1,
    )
    if notice_length != EXPECTED_NOTICE_BYTES:
        raise AttestationError("unexpected local third-party notice byte length")
    notice_sha = _expect_sha(
        license_record["local_notice_sha256"],
        "manifest.provenance.upstream_code_license.local_notice_sha256",
        SHA256_RE,
    )
    if notice_sha != EXPECTED_NOTICE_SHA256:
        raise AttestationError("unexpected local third-party notice identity")

    observations = _expect_sequence(
        provenance["observations"], "manifest.provenance.observations"
    )
    ids: list[str] = []
    for index, raw_observation in enumerate(observations):
        label = f"manifest.provenance.observations[{index}]"
        observation = _expect_mapping(raw_observation, label)
        if not isinstance(observation.get("id"), str):
            raise AttestationError(f"{label}.id must be a string")
        ids.append(observation["id"])
    if tuple(ids) != EXPECTED_OBSERVATION_IDS:
        raise AttestationError("provenance observations are missing or reordered")

    trainer = _expect_mapping(observations[0], "trainer-line-lcs observation")
    _expect_exact_keys(
        trainer,
        {
            "id",
            "local_path",
            "upstream_path",
            "method",
            "local_line_count",
            "matching_line_count",
            "relationship",
            "upstream_bytes_vendored",
            "reproducible_from_this_repository_alone",
        },
        "trainer-line-lcs observation",
    )
    if (
        trainer["local_path"] != "finetune.py"
        or trainer["upstream_path"] != "finetune/finetune.py"
        or trainer["method"] != "exact-line-longest-common-subsequence"
        or trainer["relationship"] != "substantial-line-overlap-observed"
        or trainer["upstream_bytes_vendored"] is not False
        or trainer["reproducible_from_this_repository_alone"] is not False
    ):
        raise AttestationError("trainer relationship observation is ambiguous")
    local_lines = _expect_int(
        trainer["local_line_count"],
        "trainer-line-lcs observation.local_line_count",
        minimum=1,
    )
    matching_lines = _expect_int(
        trainer["matching_line_count"],
        "trainer-line-lcs observation.matching_line_count",
        minimum=1,
    )
    if (
        local_lines != EXPECTED_TRAINER_LINE_COUNT
        or matching_lines != EXPECTED_TRAINER_LCS_LINE_COUNT
    ):
        raise AttestationError("trainer line-overlap observation changed")

    config = _expect_mapping(observations[1], "zero3-semantic-json observation")
    _expect_exact_keys(
        config,
        {
            "id",
            "local_path",
            "upstream_path",
            "method",
            "semantic_sha256",
            "relationship",
            "upstream_bytes_vendored",
            "reproducible_from_this_repository_alone",
        },
        "zero3-semantic-json observation",
    )
    if (
        config["local_path"] != "configs/ds_config_zero3.json"
        or config["upstream_path"] != "finetune/configs/ds_config_zero3.json"
        or config["method"] != "sorted-compact-json-utf8-line-v1"
        or config["relationship"] != "equal-semantic-hash-observed"
        or config["upstream_bytes_vendored"] is not False
        or config["reproducible_from_this_repository_alone"] is not False
    ):
        raise AttestationError("ZeRO-3 relationship observation is ambiguous")
    semantic_sha = _expect_sha(
        config["semantic_sha256"],
        "zero3-semantic-json observation.semantic_sha256",
        SHA256_RE,
    )
    if semantic_sha != EXPECTED_ZERO3_SEMANTIC_SHA256:
        raise AttestationError("ZeRO-3 semantic-hash observation changed")
    return root


def verify_repository(root: Path) -> dict[str, Any]:
    """Verify the manifest, legacy bytes, notice, and local semantic evidence."""

    root = root.resolve(strict=True)
    if not root.is_dir():
        raise AttestationError("repository root is not a directory")

    manifest_bytes = _read_regular_file(
        root, MANIFEST_PATH, byte_limit=MAX_MANIFEST_BYTES
    )
    manifest = _validate_manifest(_decode_json(manifest_bytes))

    verified_files: list[dict[str, Any]] = []
    payload_by_path: dict[str, bytes] = {}
    for raw_entry in manifest["files"]:
        entry = _expect_mapping(raw_entry, "manifest file entry")
        relative = _safe_relative_path(entry["path"], "manifest file path")
        path = relative.as_posix()
        trusted = EXPECTED_LEGACY_IDENTITIES[path]
        payload = _read_regular_file(
            root,
            relative,
            byte_limit=trusted["byte_length"],
        )
        payload_by_path[path] = payload
        actual_mode = _git_file_mode(root, relative)
        if actual_mode != trusted["git_mode"]:
            raise AttestationError(
                f"{path} Git file mode changed: expected {trusted['git_mode']}, "
                f"found {actual_mode}"
            )

        expected_length = trusted["byte_length"]
        if len(payload) != expected_length:
            raise AttestationError(
                f"{path} byte length changed: expected {expected_length}, "
                f"found {len(payload)}"
            )
        expected_lines = trusted["line_count"]
        actual_lines = len(payload.splitlines())
        if actual_lines != expected_lines:
            raise AttestationError(
                f"{path} line count changed: expected {expected_lines}, "
                f"found {actual_lines}"
            )

        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_sha256 != trusted["sha256"]:
            raise AttestationError(
                f"{path} SHA-256 changed: expected {trusted['sha256']}, "
                f"found {actual_sha256}"
            )
        actual_blob = _git_blob_sha1(payload)
        if actual_blob != trusted["git_blob_sha1"]:
            raise AttestationError(
                f"{path} Git blob identity changed: "
                f"expected {trusted['git_blob_sha1']}, found {actual_blob}"
            )
        verified_files.append(
            {
                "path": path,
                "byte_length": len(payload),
                "sha256": actual_sha256,
            }
        )

    provenance = _expect_mapping(manifest["provenance"], "manifest.provenance")
    license_record = _expect_mapping(
        provenance["upstream_code_license"], "upstream code license"
    )
    notice_path = _safe_relative_path(
        license_record["local_notice_path"], "local notice path"
    )
    notice = _read_regular_file(
        root,
        notice_path,
        byte_limit=EXPECTED_NOTICE_BYTES,
    )
    if len(notice) != EXPECTED_NOTICE_BYTES:
        raise AttestationError("local third-party notice byte length changed")
    if hashlib.sha256(notice).hexdigest() != EXPECTED_NOTICE_SHA256:
        raise AttestationError("local third-party notice SHA-256 changed")

    observations = _expect_sequence(provenance["observations"], "observations")
    config_observation = _expect_mapping(
        observations[1], "zero3-semantic-json observation"
    )
    semantic_sha = semantic_json_sha256(
        payload_by_path["configs/ds_config_zero3.json"]
    )
    if semantic_sha != config_observation["semantic_sha256"]:
        raise AttestationError(
            "local ZeRO-3 semantic hash changed: "
            f"expected {config_observation['semantic_sha256']}, "
            f"found {semantic_sha}"
        )

    return {
        "status": "verified",
        "manifest": MANIFEST_PATH.as_posix(),
        "source_git_commit": manifest["snapshot"]["source_git_commit"],
        "legacy_state": manifest["snapshot"]["legacy_state"],
        "files_verified": verified_files,
        "semantic_evidence": {
            "path": "configs/ds_config_zero3.json",
            "algorithm": config_observation["method"],
            "sha256": semantic_sha,
            "upstream_reverification": "not-performed-upstream-bytes-not-vendored",
        },
        "network_access": "not-used",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail closed if the inherited training snapshot has drifted."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to the parent of this tool directory)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the deterministic machine-readable report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = verify_repository(args.root)
    except (AttestationError, OSError) as exc:
        print(f"ATTESTATION FAILED: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("VERIFIED: inherited snapshot matches the locked manifest")
        print(f"source Git commit: {report['source_git_commit']}")
        print(f"legacy files: {len(report['files_verified'])}")
        print(
            "local semantic evidence: "
            f"{report['semantic_evidence']['sha256']}"
        )
        print("network access: not used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
