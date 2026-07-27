"""Bounded, canonical input contract for synthetic boundary cases."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Final

SCHEMA_VERSION: Final = "token-boundary-synthetic-case-v1"
MAX_DOCUMENT_BYTES: Final = 32 * 1024
MAX_TEXT_BYTES: Final = 8 * 1024
MAX_JSON_DEPTH: Final = 4
MAX_JSON_NODES: Final = 64
MIN_MODEL_LENGTH: Final = 2
MAX_MODEL_LENGTH: Final = 512
CASE_ID_PATTERN: Final = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
ROOT_FIELDS: Final = frozenset(
    {"schema_version", "case_id", "source", "target", "max_length"}
)


class BoundaryContractError(ValueError):
    """A stable, content-free input rejection."""

    __slots__ = ("code",)

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"boundary case rejected: {code}")


class _JSONRejected(Exception):
    pass


@dataclass(frozen=True, slots=True)
class BoundaryCase:
    """One immutable, caller-authored synthetic boundary case."""

    case_id: str
    source: str
    target: str
    max_length: int

    def __post_init__(self) -> None:
        _validate_case_fields(
            self.case_id,
            self.source,
            self.target,
            self.max_length,
        )


def _reject_float(_: str) -> float:
    raise _JSONRejected


def _reject_constant(_: str) -> float:
    raise _JSONRejected


def _bounded_integer(raw: str) -> int:
    if len(raw.lstrip("-")) > 6:
        raise _JSONRejected
    return int(raw)


def _closed_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _JSONRejected
        result[key] = value
    return result


def _decode_document(payload: bytes) -> object:
    if type(payload) is not bytes:
        raise BoundaryContractError("document.type")
    if not payload or len(payload) > MAX_DOCUMENT_BYTES:
        raise BoundaryContractError("document.size")

    decode_failed = False
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        decode_failed = True
        text = ""
    if decode_failed:
        raise BoundaryContractError("document.utf8")

    parse_failed = False
    try:
        value = json.loads(
            text,
            object_pairs_hook=_closed_object,
            parse_float=_reject_float,
            parse_int=_bounded_integer,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, _JSONRejected, RecursionError):
        parse_failed = True
        value = None
    if parse_failed:
        raise BoundaryContractError("document.json")
    _validate_tree(value)
    return value


def _validate_tree(root: object) -> None:
    nodes = 0
    stack: list[tuple[object, int]] = [(root, 1)]
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise BoundaryContractError("document.nodes")
        if depth > MAX_JSON_DEPTH:
            raise BoundaryContractError("document.depth")
        if isinstance(value, dict):
            stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)


def _valid_unicode_text(value: object) -> bool:
    if type(value) is not str or not value:
        return False
    assert isinstance(value, str)
    if "\x00" in value:
        return False
    return not any(0xD800 <= ord(character) <= 0xDFFF for character in value)


def _encoded_length(value: str) -> int:
    encoding_failed = False
    try:
        length = len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError:
        encoding_failed = True
        length = 0
    if encoding_failed:
        raise BoundaryContractError("text.unicode")
    return length


def _validate_case_fields(
    case_id: object,
    source: object,
    target: object,
    max_length: object,
) -> None:
    if type(case_id) is not str or CASE_ID_PATTERN.fullmatch(case_id) is None:
        raise BoundaryContractError("case_id")
    if not _valid_unicode_text(source) or not _valid_unicode_text(target):
        raise BoundaryContractError("text.value")
    assert isinstance(source, str)
    assert isinstance(target, str)
    if (
        _encoded_length(source) > MAX_TEXT_BYTES
        or _encoded_length(target) > MAX_TEXT_BYTES
    ):
        raise BoundaryContractError("text.size")
    if type(max_length) is not int:
        raise BoundaryContractError("max_length.type")
    assert isinstance(max_length, int)
    if not MIN_MODEL_LENGTH <= max_length <= MAX_MODEL_LENGTH:
        raise BoundaryContractError("max_length.range")


def parse_boundary_case(payload: bytes) -> BoundaryCase:
    """Parse one closed JSON case without retaining rejected content."""

    document = _decode_document(payload)
    if type(document) is not dict:
        raise BoundaryContractError("document.root")
    assert isinstance(document, dict)
    if set(document) != ROOT_FIELDS:
        raise BoundaryContractError("document.fields")
    if document["schema_version"] != SCHEMA_VERSION:
        raise BoundaryContractError("schema_version")
    return BoundaryCase(
        case_id=document["case_id"],
        source=document["source"],
        target=document["target"],
        max_length=document["max_length"],
    )


def canonical_boundary_case_bytes(case: BoundaryCase) -> bytes:
    """Return the only canonical document for a validated case."""

    if type(case) is not BoundaryCase:
        raise BoundaryContractError("case.type")
    document = {
        "case_id": case.case_id,
        "max_length": case.max_length,
        "schema_version": SCHEMA_VERSION,
        "source": case.source,
        "target": case.target,
    }
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def boundary_case_sha256(case: BoundaryCase) -> str:
    """Hash the canonical semantic identity of a boundary case."""

    return hashlib.sha256(canonical_boundary_case_bytes(case)).hexdigest()
