"""Versioned contract for synthetic conversations and pretokenized traces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any


SCHEMA_VERSION = 1
TRACE_KIND = "loss-topology.synthetic-pretokenized-trace"
MAX_INPUT_BYTES = 128 * 1024
MAX_JSON_DEPTH = 8
MAX_JSON_NODES = 20_000
MAX_MESSAGES = 64
MAX_MESSAGE_CONTENT_BYTES = 8 * 1024
MAX_TOTAL_CONTENT_BYTES = 64 * 1024
MAX_TOKENS = 4_096
MAX_TOKEN_ID = 2_147_483_647
MAX_SEGMENTS = (3 * MAX_MESSAGES) + 3
EXAMPLE_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
ROLES = frozenset({"system", "user", "assistant"})
MESSAGE_SEGMENT_KINDS = (
    "message_start",
    "message_content",
    "message_end",
)


class ContractError(RuntimeError):
    """A stable, redacted contract failure.

    ``code`` is safe to expose to a caller. Input content, paths, keys, values,
    and parser excerpts are intentionally never included.
    """

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _JSONViolation(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class Message:
    """One immutable message in a strict SFT conversation."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class TokenSegment:
    """A half-open token span with an explicit template/content boundary."""

    kind: str
    message_index: int | None
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class PretokenizedTrace:
    """Caller-supplied token IDs and their asserted segment topology."""

    token_ids: tuple[int, ...]
    padding_token_id: int
    segments: tuple[TokenSegment, ...]

    @property
    def padding_start(self) -> int:
        return self.segments[-1].start


@dataclass(frozen=True, slots=True)
class SyntheticTrace:
    """Validated v1 envelope.

    Validation proves internal structural consistency only. It cannot prove
    that a tokenizer would map ``messages`` to ``token_ids``.
    """

    schema_version: int
    kind: str
    example_id: str
    messages: tuple[Message, ...]
    trace: PretokenizedTrace


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _JSONViolation("json.duplicate_key")
        result[key] = value
    return result


def _parse_int(value: str) -> int:
    digits = value[1:] if value.startswith("-") else value
    if len(digits) > 10:
        raise _JSONViolation("json.integer_out_of_range")
    parsed = int(value)
    if abs(parsed) > MAX_TOKEN_ID:
        raise _JSONViolation("json.integer_out_of_range")
    return parsed


def _reject_float(_: str) -> None:
    raise _JSONViolation("json.floating_point_forbidden")


def _reject_constant(_: str) -> None:
    raise _JSONViolation("json.non_finite_number")


def _check_json_shape(value: Any) -> None:
    nodes = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise ContractError("json.node_limit")
        if depth > MAX_JSON_DEPTH:
            raise ContractError("json.depth_limit")
        if isinstance(current, Mapping):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)


def _decode_json(payload: bytes) -> Any:
    if len(payload) > MAX_INPUT_BYTES:
        raise ContractError("input.byte_limit")
    text: str | None = None
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        pass
    if text is None:
        raise ContractError("input.utf8")

    failure: str | None = None
    value: Any = None
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_int=_parse_int,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except _JSONViolation as error:
        failure = error.code
    except (json.JSONDecodeError, RecursionError, ValueError):
        failure = "json.invalid"
    if failure is not None:
        raise ContractError(failure)
    _check_json_shape(value)
    return value


def _object(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(code)
    return value


def _array(value: Any, code: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractError(code)
    return value


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    code: str,
) -> None:
    if frozenset(value) != expected:
        raise ContractError(code)


def _exact_int(
    value: Any,
    code: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_TOKEN_ID,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise ContractError(code)
    return value


def _parse_messages(value: Any) -> tuple[Message, ...]:
    conversation = _object(value, "conversation.type")
    _exact_fields(conversation, frozenset({"messages"}), "conversation.fields")
    raw_messages = _array(conversation["messages"], "messages.type")
    if not 1 <= len(raw_messages) <= MAX_MESSAGES:
        raise ContractError("messages.count")

    messages: list[Message] = []
    total_content_bytes = 0
    for raw_message in raw_messages:
        message = _object(raw_message, "message.type")
        _exact_fields(
            message,
            frozenset({"role", "content"}),
            "message.fields",
        )
        role = message["role"]
        content = message["content"]
        if not isinstance(role, str) or role not in ROLES:
            raise ContractError("message.role")
        if not isinstance(content, str):
            raise ContractError("message.content_type")
        if "\x00" in content:
            raise ContractError("message.content_nul")
        content_bytes: int | None = None
        try:
            content_bytes = len(content.encode("utf-8"))
        except UnicodeEncodeError:
            pass
        if content_bytes is None:
            raise ContractError("message.content_unicode")
        if content_bytes > MAX_MESSAGE_CONTENT_BYTES:
            raise ContractError("message.content_limit")
        total_content_bytes += content_bytes
        if total_content_bytes > MAX_TOTAL_CONTENT_BYTES:
            raise ContractError("messages.content_limit")
        messages.append(Message(role=role, content=content))

    first_conversation_index = 0
    if messages[0].role == "system":
        first_conversation_index = 1
    if first_conversation_index >= len(messages):
        raise ContractError("messages.turn_order")
    for index, message in enumerate(messages):
        if index == 0 and message.role == "system":
            continue
        offset = index - first_conversation_index
        expected_role = "user" if offset % 2 == 0 else "assistant"
        if message.role != expected_role:
            raise ContractError("messages.turn_order")
    if messages[-1].role != "assistant":
        raise ContractError("messages.turn_order")
    return tuple(messages)


def _parse_segment(raw_value: Any) -> TokenSegment:
    value = _object(raw_value, "segment.type")
    _exact_fields(
        value,
        frozenset({"kind", "message_index", "start", "end"}),
        "segment.fields",
    )
    kind = value["kind"]
    if not isinstance(kind, str):
        raise ContractError("segment.kind")
    message_index = value["message_index"]
    if message_index is not None:
        message_index = _exact_int(
            message_index,
            "segment.message_index",
            maximum=MAX_MESSAGES - 1,
        )
    start = _exact_int(value["start"], "segment.start", maximum=MAX_TOKENS)
    end = _exact_int(value["end"], "segment.end", maximum=MAX_TOKENS)
    return TokenSegment(
        kind=kind,
        message_index=message_index,
        start=start,
        end=end,
    )


def _expected_segment_identities(
    message_count: int,
) -> tuple[tuple[str, int | None], ...]:
    identities: list[tuple[str, int | None]] = [("prefix_special", None)]
    for message_index in range(message_count):
        identities.extend(
            (kind, message_index) for kind in MESSAGE_SEGMENT_KINDS
        )
    identities.extend((("suffix_special", None), ("padding", None)))
    return tuple(identities)


def _parse_trace(
    value: Any,
    messages: tuple[Message, ...],
) -> PretokenizedTrace:
    trace = _object(value, "trace.type")
    _exact_fields(
        trace,
        frozenset({"token_ids", "padding_token_id", "segments"}),
        "trace.fields",
    )
    raw_token_ids = _array(trace["token_ids"], "token_ids.type")
    if not 1 <= len(raw_token_ids) <= MAX_TOKENS:
        raise ContractError("token_ids.count")
    token_ids = tuple(
        _exact_int(item, "token_ids.item") for item in raw_token_ids
    )
    padding_token_id = _exact_int(
        trace["padding_token_id"],
        "padding_token_id",
    )

    raw_segments = _array(trace["segments"], "segments.type")
    expected_identities = _expected_segment_identities(len(messages))
    if (
        len(raw_segments) > MAX_SEGMENTS
        or len(raw_segments) != len(expected_identities)
    ):
        raise ContractError("segments.count")
    segments = tuple(_parse_segment(item) for item in raw_segments)

    previous_end = 0
    for segment, expected_identity in zip(
        segments,
        expected_identities,
        strict=True,
    ):
        if (segment.kind, segment.message_index) != expected_identity:
            raise ContractError("segments.sequence")
        if segment.start != previous_end:
            raise ContractError("segments.coverage")
        if segment.end < segment.start or segment.end > len(token_ids):
            raise ContractError("segments.range")
        if (
            segment.kind in {"message_start", "message_end"}
            and segment.length == 0
        ):
            raise ContractError("segments.boundary_empty")
        previous_end = segment.end
    if previous_end != len(token_ids):
        raise ContractError("segments.coverage")

    padding = segments[-1]
    if padding.start == 0:
        raise ContractError("trace.non_padding_empty")
    if any(
        token_id != padding_token_id
        for token_id in token_ids[padding.start : padding.end]
    ):
        raise ContractError("trace.padding_mismatch")
    return PretokenizedTrace(
        token_ids=token_ids,
        padding_token_id=padding_token_id,
        segments=segments,
    )


def parse_synthetic_trace(payload: bytes) -> SyntheticTrace:
    """Parse and validate one bounded v1 synthetic trace.

    This validates the supplied mapping between messages, token IDs, and
    segments. It intentionally does not attest that any real tokenizer
    produced that mapping.
    """

    if type(payload) is not bytes:
        raise ContractError("input.type")
    decoded = _decode_json(payload)
    root = _object(decoded, "root.type")
    _exact_fields(
        root,
        frozenset(
            {
                "schema_version",
                "kind",
                "example_id",
                "conversation",
                "trace",
            }
        ),
        "root.fields",
    )
    schema_version = _exact_int(
        root["schema_version"],
        "schema.version",
        minimum=SCHEMA_VERSION,
        maximum=SCHEMA_VERSION,
    )
    if root["kind"] != TRACE_KIND:
        raise ContractError("schema.kind")
    example_id = root["example_id"]
    if (
        not isinstance(example_id, str)
        or EXAMPLE_ID_RE.fullmatch(example_id) is None
    ):
        raise ContractError("example_id")
    messages = _parse_messages(root["conversation"])
    trace = _parse_trace(root["trace"], messages)
    return SyntheticTrace(
        schema_version=schema_version,
        kind=TRACE_KIND,
        example_id=example_id,
        messages=messages,
        trace=trace,
    )


def trace_to_primitive(trace: SyntheticTrace) -> dict[str, Any]:
    """Return the normalized JSON-compatible representation of ``trace``."""

    return {
        "schema_version": trace.schema_version,
        "kind": trace.kind,
        "example_id": trace.example_id,
        "conversation": {
            "messages": [
                {"role": message.role, "content": message.content}
                for message in trace.messages
            ]
        },
        "trace": {
            "token_ids": list(trace.trace.token_ids),
            "padding_token_id": trace.trace.padding_token_id,
            "segments": [
                {
                    "kind": segment.kind,
                    "message_index": segment.message_index,
                    "start": segment.start,
                    "end": segment.end,
                }
                for segment in trace.trace.segments
            ],
        },
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def canonical_trace_bytes(trace: SyntheticTrace) -> bytes:
    """Return canonical UTF-8 JSON for a validated synthetic trace."""

    return _canonical_json_bytes(trace_to_primitive(trace))


def trace_sha256(trace: SyntheticTrace) -> str:
    """Hash the canonical, validated trace representation."""

    return hashlib.sha256(canonical_trace_bytes(trace)).hexdigest()
