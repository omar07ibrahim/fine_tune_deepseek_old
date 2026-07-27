"""Conservative raw-code-point provenance for the fixed normalizer pipeline."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Final

MAX_TEXT_BYTES: Final = 16 * 1024
CLUSTER_REPLAY_REASON: Final = "normalization.cluster_replay"
AMBIGUOUS_PROVENANCE_REASON: Final = "normalization.provenance_ambiguous"


class ProvenanceError(ValueError):
    """A stable, content-free normalization input rejection."""

    __slots__ = ("code",)

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"normalization input rejected: {code}")


@dataclass(frozen=True, slots=True)
class NormalizationTrace:
    """The fixed NFC-plus-right-strip result and conservative raw origins."""

    normalized: str
    origins: tuple[tuple[int, ...] | None, ...]
    raw_codepoint_count: int
    nfc_codepoint_count: int
    trailing_normalized_codepoints_removed: int
    cross_boundary_output_positions: tuple[int, ...]
    complete: bool
    indeterminate_reasons: tuple[str, ...]


def _validate_input(text: object, boundary: object) -> tuple[str, int]:
    if type(text) is not str:
        raise ProvenanceError("text.type")
    assert isinstance(text, str)
    if (
        not text
        or "\x00" in text
        or any(0xD800 <= ord(character) <= 0xDFFF for character in text)
    ):
        raise ProvenanceError("text.value")

    encoding_failed = False
    try:
        encoded_length = len(text.encode("utf-8", errors="strict"))
    except UnicodeEncodeError:
        encoding_failed = True
        encoded_length = 0
    if encoding_failed:
        raise ProvenanceError("text.value")
    if encoded_length > MAX_TEXT_BYTES:
        raise ProvenanceError("text.size")

    if type(boundary) is not int:
        raise ProvenanceError("boundary.type")
    assert isinstance(boundary, int)
    if not 0 <= boundary <= len(text):
        raise ProvenanceError("boundary.range")
    return text, boundary


def _canonical_cluster_spans(text: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    start = 0
    for position in range(1, len(text)):
        if unicodedata.combining(text[position]) == 0:
            spans.append((start, position))
            start = position
    spans.append((start, len(text)))
    return tuple(spans)


def _cluster_origins(
    raw_cluster: str,
    normalized_cluster: str,
    start: int,
) -> tuple[tuple[tuple[int, ...] | None, ...], bool]:
    raw_positions = tuple(range(start, start + len(raw_cluster)))
    if len(raw_cluster) == 1 and len(normalized_cluster) == 1:
        return ((raw_positions,), False)
    if len(raw_cluster) > 1 and len(normalized_cluster) == 1:
        return ((raw_positions,), False)
    if (
        len(normalized_cluster) > 1
        and normalized_cluster == raw_cluster
        and all(
            unicodedata.normalize("NFC", character) == character
            for character in raw_cluster
        )
    ):
        return (tuple((position,) for position in raw_positions), False)
    return (tuple(None for _ in normalized_cluster), True)


def trace_normalization(text: str, boundary: int) -> NormalizationTrace:
    """Replay fixed NFC then ``rstrip`` with conservative raw provenance."""

    text, boundary = _validate_input(text, boundary)
    replay_parts: list[str] = []
    replay_origins: list[tuple[int, ...] | None] = []
    reasons: list[str] = []

    for start, end in _canonical_cluster_spans(text):
        raw_cluster = text[start:end]
        normalized_cluster = unicodedata.normalize("NFC", raw_cluster)
        cluster_origins, ambiguous = _cluster_origins(
            raw_cluster,
            normalized_cluster,
            start,
        )
        replay_parts.append(normalized_cluster)
        replay_origins.extend(cluster_origins)
        if ambiguous:
            reasons.append(AMBIGUOUS_PROVENANCE_REASON)

    replay = "".join(replay_parts)
    nfc_text = unicodedata.normalize("NFC", text)
    replay_matches = replay == nfc_text
    if not replay_matches:
        reasons = [CLUSTER_REPLAY_REASON]
        nfc_origins: tuple[tuple[int, ...] | None, ...] = tuple(None for _ in nfc_text)
    else:
        nfc_origins = tuple(replay_origins)

    normalized = nfc_text.rstrip()
    retained_count = len(normalized)
    origins = nfc_origins[:retained_count]
    removed_count = len(nfc_text) - retained_count
    cross_positions = tuple(
        position
        for position, origin in enumerate(origins)
        if origin is not None
        and any(raw_position < boundary for raw_position in origin)
        and any(raw_position >= boundary for raw_position in origin)
    )
    complete = replay_matches and all(origin is not None for origin in nfc_origins)

    return NormalizationTrace(
        normalized=normalized,
        origins=origins,
        raw_codepoint_count=len(text),
        nfc_codepoint_count=len(nfc_text),
        trailing_normalized_codepoints_removed=removed_count,
        cross_boundary_output_positions=cross_positions,
        complete=complete,
        indeterminate_reasons=tuple(sorted(set(reasons))),
    )
