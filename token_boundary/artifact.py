"""Trusted byte and semantic verification for the fixed tokenizer artifact."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NoReturn

ARTIFACT_DIRECTORY: Final = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_FILENAME: Final = "local-boundary-bpe.v1.json"
MANIFEST_FILENAME: Final = "artifact-manifest.v1.json"
ARTIFACT_BYTE_COUNT: Final = 1533
ARTIFACT_SHA256: Final = (
    "29508edbb44ce9cbe77cdde972c0919fe3df8ee2ae1270e69545314f2e1f8358"
)
MANIFEST_BYTE_COUNT: Final = 1092
MANIFEST_SHA256: Final = (
    "2e52353b8dc21c00601c2f9ce925d1f2acb6b738e61a4d03d0732da10a98ef1d"
)
MAX_ARTIFACT_BYTES: Final = 16 * 1024
MAX_MANIFEST_BYTES: Final = 8 * 1024
RUNTIME_PACKAGE: Final = "tokenizers"
RUNTIME_VERSION: Final = "0.21.4"
ARTIFACT_ID: Final = "local-boundary-bpe-v1"

TRUSTED_MANIFEST: Final = {
    "schema_version": "token-boundary-artifact-manifest-v1",
    "artifact": {
        "byte_count": ARTIFACT_BYTE_COUNT,
        "format": "huggingface-tokenizers-json-v1",
        "id": ARTIFACT_ID,
        "origin": "locally-authored-synthetic",
        "path": ARTIFACT_FILENAME,
        "sha256": ARTIFACT_SHA256,
    },
    "intent": {
        "bpe_merges": ["a b"],
        "normalizers": ["NFC", "Strip(left=false,right=true)"],
        "post_processor": "prefix-bos-only",
        "pre_tokenizer": "none",
        "truncation": "caller-bounded-right",
    },
    "runtime": {
        "audit_mode": "offline-after-provisioning",
        "package": RUNTIME_PACKAGE,
        "provisioning_may_use_network": True,
        "version": RUNTIME_VERSION,
    },
    "scope": {
        "dataset_loaded": False,
        "deepseek_model_executed": False,
        "deepseek_tokenizer_attested": False,
        "gpu_used": False,
        "inherited_trainer_executed": False,
        "model_or_tokenizer_downloaded_by_audit": False,
        "pretrained_artifact": False,
        "training_or_forward_pass_executed": False,
    },
}


class ArtifactVerificationError(RuntimeError):
    """A stable, path-free failure for the fixed artifact boundary."""

    __slots__ = ("code",)

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"tokenizer artifact rejected: {code}")


class _JSONRejected(Exception):
    pass


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    """The exact trusted bytes and their closed identity."""

    artifact_id: str
    filename: str
    artifact_format: str
    runtime_package: str
    runtime_version: str
    sha256: str
    byte_count: int
    payload: bytes


def _closed_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _JSONRejected
        result[key] = value
    return result


def _reject_json_number(_: str) -> NoReturn:
    raise _JSONRejected


def _decode_json(payload: bytes, code: str) -> object:
    failed = False
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_closed_object,
            parse_float=_reject_json_number,
            parse_constant=_reject_json_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _JSONRejected, RecursionError):
        failed = True
        value = None
    if failed:
        raise ArtifactVerificationError(code)
    return value


def _read_fixed_files(directory: Path) -> tuple[bytes, bytes]:
    failure: str | None = None
    directory_descriptor = -1
    descriptors: list[int] = []
    contents: list[bytes] = []
    try:
        mode = os.lstat(directory).st_mode
        if not stat.S_ISDIR(mode):
            failure = "directory.shape"
        else:
            directory_descriptor = os.open(
                directory,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
        for filename, maximum in (
            (ARTIFACT_FILENAME, MAX_ARTIFACT_BYTES),
            (MANIFEST_FILENAME, MAX_MANIFEST_BYTES),
        ):
            if failure is not None:
                break
            descriptor = os.open(
                filename,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory_descriptor,
            )
            descriptors.append(descriptor)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                failure = "file.shape"
                break
            if metadata.st_size <= 0 or metadata.st_size > maximum:
                failure = "file.size"
                break
            chunks: list[bytes] = []
            remaining = metadata.st_size + 1
            while remaining > 0:
                chunk = os.read(descriptor, min(remaining, 64 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            if len(payload) != metadata.st_size:
                failure = "file.read"
                break
            contents.append(payload)
    except OSError:
        failure = "file.io"
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                failure = failure or "file.close"
        if directory_descriptor >= 0:
            try:
                os.close(directory_descriptor)
            except OSError:
                failure = failure or "directory.close"

    if failure is not None or len(contents) != 2:
        raise ArtifactVerificationError(failure or "file.read")
    return contents[0], contents[1]


def _validate_artifact_document(document: object) -> None:
    if type(document) is not dict:
        raise ArtifactVerificationError("artifact.document")
    assert isinstance(document, dict)
    if set(document) != {
        "version",
        "truncation",
        "padding",
        "added_tokens",
        "normalizer",
        "pre_tokenizer",
        "post_processor",
        "decoder",
        "model",
    }:
        raise ArtifactVerificationError("artifact.fields")
    if (
        document["version"] != "1.0"
        or document["truncation"] is not None
        or document["padding"] is not None
        or document["added_tokens"] != []
        or document["pre_tokenizer"] is not None
        or document["decoder"] is not None
    ):
        raise ArtifactVerificationError("artifact.pipeline")
    expected_normalizer = {
        "type": "Sequence",
        "normalizers": [
            {"type": "NFC"},
            {"type": "Strip", "strip_left": False, "strip_right": True},
        ],
    }
    if document["normalizer"] != expected_normalizer:
        raise ArtifactVerificationError("artifact.normalizer")
    model = document["model"]
    if type(model) is not dict:
        raise ArtifactVerificationError("artifact.model")
    assert isinstance(model, dict)
    expected_vocab = {
        "<unk>": 0,
        "<bos>": 1,
        " ": 2,
        "a": 3,
        "b": 4,
        "c": 5,
        "d": 6,
        "e": 7,
        "f": 8,
        "g": 9,
        "h": 10,
        "i": 11,
        "x": 12,
        "é": 13,
        "ab": 14,
    }
    expected_model = {
        "type": "BPE",
        "dropout": None,
        "unk_token": "<unk>",
        "continuing_subword_prefix": None,
        "end_of_word_suffix": None,
        "fuse_unk": False,
        "byte_fallback": False,
        "ignore_merges": False,
        "vocab": expected_vocab,
        "merges": [["a", "b"]],
    }
    if model != expected_model:
        raise ArtifactVerificationError("artifact.model")
    processor = document["post_processor"]
    if type(processor) is not dict:
        raise ArtifactVerificationError("artifact.processor")
    assert isinstance(processor, dict)
    expected_processor = {
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "<bos>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
        ],
        "pair": [
            {"Sequence": {"id": "A", "type_id": 0}},
            {"Sequence": {"id": "B", "type_id": 1}},
        ],
        "special_tokens": {"<bos>": {"id": "<bos>", "ids": [1], "tokens": ["<bos>"]}},
    }
    if processor != expected_processor:
        raise ArtifactVerificationError("artifact.processor")


def verify_local_artifact(directory: Path = ARTIFACT_DIRECTORY) -> VerifiedArtifact:
    """Verify fixed trusted bytes without importing the tokenizer runtime."""

    if not isinstance(directory, Path) or not directory.is_absolute():
        raise ArtifactVerificationError("directory.value")
    artifact_payload, manifest_payload = _read_fixed_files(directory)
    if (
        len(artifact_payload) != ARTIFACT_BYTE_COUNT
        or hashlib.sha256(artifact_payload).hexdigest() != ARTIFACT_SHA256
    ):
        raise ArtifactVerificationError("artifact.identity")
    if (
        len(manifest_payload) != MANIFEST_BYTE_COUNT
        or hashlib.sha256(manifest_payload).hexdigest() != MANIFEST_SHA256
    ):
        raise ArtifactVerificationError("manifest.identity")

    manifest = _decode_json(manifest_payload, "manifest.document")
    if manifest != TRUSTED_MANIFEST:
        raise ArtifactVerificationError("manifest.semantics")
    artifact = _decode_json(artifact_payload, "artifact.document")
    _validate_artifact_document(artifact)

    return VerifiedArtifact(
        artifact_id=ARTIFACT_ID,
        filename=ARTIFACT_FILENAME,
        artifact_format="huggingface-tokenizers-json-v1",
        runtime_package=RUNTIME_PACKAGE,
        runtime_version=RUNTIME_VERSION,
        sha256=ARTIFACT_SHA256,
        byte_count=ARTIFACT_BYTE_COUNT,
        payload=artifact_payload,
    )
