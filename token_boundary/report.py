"""Canonical, privacy-minimal tokenizer-boundary differential reports."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Final, TypeAlias

SCHEMA_VERSION: Final = "token-boundary-differential-report-v1"
REPORT_KIND: Final = "token-boundary.differential-report"

CLASSIFICATION_PRECEDENCE: Final = (
    "indeterminate",
    "cross_boundary_token",
    "target_eliminated",
    "partial_target_truncation",
    "boundary_drift",
    "aligned",
)
CLASSIFICATIONS: Final = frozenset(CLASSIFICATION_PRECEDENCE)
OWNERSHIPS: Final = frozenset(
    {
        "injected_prefix",
        "source",
        "target",
        "cross_boundary",
        "ambiguous",
    }
)
REPORT_STATUSES: Final = frozenset({"pass", "fail", "indeterminate"})

CASE_ID_PATTERN: Final = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
CODE_PATTERN: Final = re.compile(r"[a-z0-9][a-z0-9._-]{0,95}\Z")
SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}\Z")
MIN_MODEL_LENGTH: Final = 2
MAX_MODEL_LENGTH: Final = 512
MAX_TOKEN_ID: Final = 2**63 - 1
MAX_TEXT_CODEPOINTS: Final = 8 * 1024
MAX_COMBINED_CODEPOINTS: Final = 2 * MAX_TEXT_CODEPOINTS
MAX_NORMALIZED_CODEPOINTS: Final = 4 * MAX_COMBINED_CODEPOINTS
MAX_ENCODING_TOKENS: Final = MAX_NORMALIZED_CODEPOINTS + 1

TOKEN_PIECES: Final = (
    "<unk>",
    "<bos>",
    " ",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "x",
    "\u00e9",
    "ab",
)

EXPECTED_ARTIFACT_ID: Final = "local-boundary-bpe-v1"
EXPECTED_ARTIFACT_SHA256: Final = (
    "29508edbb44ce9cbe77cdde972c0919fe3df8ee2ae1270e69545314f2e1f8358"
)
EXPECTED_ARTIFACT_BYTES: Final = 1533
EXPECTED_ENGINE: Final = "tokenizers"
EXPECTED_ENGINE_VERSION: Final = "0.21.4"
EXPECTED_NORMALIZERS: Final = (
    "NFC",
    "Strip(left=false,right=true)",
)
EXPECTED_POST_PROCESSOR: Final = "prefix-bos-only"
EXPECTED_TRUNCATION: Final = "caller-bounded-right"
EXPECTED_LEGACY_ALGORITHM: Final = "standalone-source-token-count"
ELIMINATION_CAUSE_ORDER: Final = ("truncation", "legacy_cutoff")
INDETERMINATE_REASONS: Final = frozenset(
    {
        "encoding.offset_provenance",
        "encoding.piece_alignment",
        "encoding.unknown_token",
        "normalization.cluster_replay",
        "normalization.provenance_ambiguous",
        "normalization.runtime_mismatch",
        "target.no_attributable_token",
        "truncation.not_exact_prefix",
    }
)

SCOPE_FALSE_FIELDS: Final = (
    "dataset_loaded",
    "deepseek_model_executed",
    "deepseek_tokenizer_attested",
    "gpu_used",
    "inherited_trainer_executed",
    "inherited_trainer_fixed",
    "model_or_tokenizer_downloaded_by_audit",
    "network_used",
    "forward_pass_or_loss_computed",
    "training_or_evaluation_performed",
    "model_quality_measured",
    "prevalence_measured",
    "universal_masking_policy_claimed",
)

JSONValue: TypeAlias = (
    None | bool | int | str | list["JSONValue"] | dict[str, "JSONValue"]
)


class BoundaryReportError(ValueError):
    """A stable, content-free report invariant failure."""

    __slots__ = ("code",)

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"boundary report rejected: {code}")


def _exact_nonnegative_int(value: object, code: str) -> int:
    if type(value) is not int:
        raise BoundaryReportError(code)
    assert isinstance(value, int)
    if value < 0:
        raise BoundaryReportError(code)
    return value


def _exact_positive_int(value: object, code: str) -> int:
    result = _exact_nonnegative_int(value, code)
    if result == 0:
        raise BoundaryReportError(code)
    return result


def _exact_string(value: object, code: str) -> str:
    if type(value) is not str:
        raise BoundaryReportError(code)
    assert isinstance(value, str)
    if (
        not value
        or "\x00" in value
        or any(0xD800 <= ord(character) <= 0xDFFF for character in value)
    ):
        raise BoundaryReportError(code)
    return value


def _exact_bool(value: object, code: str) -> bool:
    if type(value) is not bool:
        raise BoundaryReportError(code)
    assert isinstance(value, bool)
    return value


def _exact_tuple(value: object, code: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise BoundaryReportError(code)
    assert isinstance(value, tuple)
    return value


def _position_tuple(
    value: object,
    code: str,
    *,
    upper_bound: int | None = None,
) -> tuple[int, ...]:
    items = _exact_tuple(value, code)
    positions: list[int] = []
    previous: int | None = None
    for item in items:
        position = _exact_nonnegative_int(item, code)
        if previous is not None and position <= previous:
            raise BoundaryReportError(code)
        if upper_bound is not None and position >= upper_bound:
            raise BoundaryReportError(code)
        positions.append(position)
        previous = position
    return tuple(positions)


def _code_tuple(
    value: object,
    code: str,
    *,
    canonical_order: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    items = _exact_tuple(value, code)
    codes: list[str] = []
    for item in items:
        item_value = _exact_string(item, code)
        if CODE_PATTERN.fullmatch(item_value) is None:
            raise BoundaryReportError(code)
        codes.append(item_value)
    result = tuple(codes)
    if len(set(result)) != len(result):
        raise BoundaryReportError(code)
    if canonical_order is None:
        if result != tuple(sorted(result)):
            raise BoundaryReportError(code)
    else:
        if any(item not in canonical_order for item in result):
            raise BoundaryReportError(code)
        expected = tuple(item for item in canonical_order if item in result)
        if result != expected:
            raise BoundaryReportError(code)
    return result


@dataclass(frozen=True, slots=True)
class EncodingSnapshot:
    """One exact runtime encoding vector bundle."""

    ids: tuple[int, ...]
    pieces: tuple[str, ...]
    offsets: tuple[tuple[int, int], ...]
    type_ids: tuple[int, ...]
    special_tokens_mask: tuple[int, ...]
    attention_mask: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_encoding_snapshot(self)


def _validate_encoding_snapshot(snapshot: EncodingSnapshot) -> None:
    ids = _exact_tuple(snapshot.ids, "encoding.ids")
    if not 1 <= len(ids) <= MAX_ENCODING_TOKENS:
        raise BoundaryReportError("encoding.length")
    checked_ids: list[int] = []
    for token_id in ids:
        value = _exact_nonnegative_int(token_id, "encoding.ids")
        if value > MAX_TOKEN_ID:
            raise BoundaryReportError("encoding.ids")
        checked_ids.append(value)

    pieces = _exact_tuple(snapshot.pieces, "encoding.pieces")
    for piece in pieces:
        _exact_string(piece, "encoding.pieces")
    if len(pieces) == len(checked_ids) and any(
        token_id >= len(TOKEN_PIECES) or TOKEN_PIECES[token_id] != piece
        for token_id, piece in zip(checked_ids, pieces, strict=True)
    ):
        raise BoundaryReportError("encoding.token_piece")

    offsets = _exact_tuple(snapshot.offsets, "encoding.offsets")
    for offset in offsets:
        pair = _exact_tuple(offset, "encoding.offsets")
        if len(pair) != 2:
            raise BoundaryReportError("encoding.offsets")
        start = _exact_nonnegative_int(pair[0], "encoding.offsets")
        end = _exact_nonnegative_int(pair[1], "encoding.offsets")
        if start > end or end > MAX_COMBINED_CODEPOINTS:
            raise BoundaryReportError("encoding.offsets")

    type_ids = _exact_tuple(snapshot.type_ids, "encoding.type_ids")
    for type_id in type_ids:
        if _exact_nonnegative_int(type_id, "encoding.type_ids") != 0:
            raise BoundaryReportError("encoding.type_ids")

    special = _exact_tuple(
        snapshot.special_tokens_mask,
        "encoding.special_tokens_mask",
    )
    for mask_item in special:
        if type(mask_item) is not int or mask_item not in (0, 1):
            raise BoundaryReportError("encoding.special_tokens_mask")

    attention = _exact_tuple(snapshot.attention_mask, "encoding.attention_mask")
    for attention_item in attention:
        if type(attention_item) is not int or attention_item != 1:
            raise BoundaryReportError("encoding.attention_mask")

    lengths = {
        len(ids),
        len(pieces),
        len(offsets),
        len(type_ids),
        len(special),
        len(attention),
    }
    if len(lengths) != 1:
        raise BoundaryReportError("encoding.length")
    if (
        checked_ids[0] != 1
        or pieces[0] != "<bos>"
        or offsets[0] != (0, 0)
        or special[0] != 1
        or any(value != 0 for value in special[1:])
    ):
        raise BoundaryReportError("encoding.special_layout")
    if any(start == end for start, end in snapshot.offsets[1:]):
        raise BoundaryReportError("encoding.offsets")
    previous_offset: tuple[int, int] | None = None
    for special_item, offset in zip(
        snapshot.special_tokens_mask,
        snapshot.offsets,
        strict=True,
    ):
        if special_item == 1:
            continue
        if previous_offset is not None and (
            offset[0] < previous_offset[0] or offset[1] < previous_offset[1]
        ):
            raise BoundaryReportError("encoding.offset_order")
        previous_offset = offset


@dataclass(frozen=True, slots=True)
class NormalizationSnapshot:
    """Counts and provenance state for NFC plus right-strip normalization."""

    raw_codepoint_count: int
    nfc_codepoint_count: int
    normalized_codepoint_count: int
    trailing_normalized_codepoints_removed: int
    cross_boundary_output_positions: tuple[int, ...]
    provenance_complete: bool

    def __post_init__(self) -> None:
        _validate_normalization_snapshot(self)


def _validate_normalization_snapshot(snapshot: NormalizationSnapshot) -> None:
    raw_count = _exact_nonnegative_int(
        snapshot.raw_codepoint_count,
        "normalization.raw_codepoint_count",
    )
    nfc_count = _exact_nonnegative_int(
        snapshot.nfc_codepoint_count,
        "normalization.nfc_codepoint_count",
    )
    normalized_count = _exact_nonnegative_int(
        snapshot.normalized_codepoint_count,
        "normalization.normalized_codepoint_count",
    )
    removed_count = _exact_nonnegative_int(
        snapshot.trailing_normalized_codepoints_removed,
        "normalization.trailing_removed",
    )
    if normalized_count + removed_count != nfc_count:
        raise BoundaryReportError("normalization.counts")
    if (
        raw_count > MAX_COMBINED_CODEPOINTS
        or nfc_count > MAX_NORMALIZED_CODEPOINTS
        or normalized_count > MAX_NORMALIZED_CODEPOINTS
    ):
        raise BoundaryReportError("normalization.bounds")
    _position_tuple(
        snapshot.cross_boundary_output_positions,
        "normalization.cross_boundary_positions",
        upper_bound=normalized_count,
    )
    _exact_bool(
        snapshot.provenance_complete,
        "normalization.provenance_complete",
    )


@dataclass(frozen=True, slots=True)
class TokenAttribution:
    """Ownership of one encoded position in combined-text coordinates."""

    position: int
    ownership: str
    raw_origins: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_token_attribution(self)


def _validate_token_attribution(attribution: TokenAttribution) -> None:
    _exact_nonnegative_int(attribution.position, "attribution.position")
    ownership = _exact_string(attribution.ownership, "attribution.ownership")
    if ownership not in OWNERSHIPS:
        raise BoundaryReportError("attribution.ownership")
    origins = _position_tuple(attribution.raw_origins, "attribution.raw_origins")
    if ownership == "injected_prefix" and origins:
        raise BoundaryReportError("attribution.injected_origins")
    if ownership in {"source", "target", "cross_boundary"} and not origins:
        raise BoundaryReportError("attribution.raw_origins")


@dataclass(frozen=True, slots=True)
class InputIdentity:
    """Content identity and privacy-safe counts for one synthetic case."""

    case_id: str
    canonical_sha256: str
    source_codepoints: int
    target_codepoints: int
    boundary_codepoint: int
    max_length: int

    def __post_init__(self) -> None:
        _validate_input_identity(self)


def _validate_input_identity(identity: InputIdentity) -> None:
    case_id = _exact_string(identity.case_id, "input.case_id")
    if CASE_ID_PATTERN.fullmatch(case_id) is None:
        raise BoundaryReportError("input.case_id")
    digest = _exact_string(identity.canonical_sha256, "input.canonical_sha256")
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise BoundaryReportError("input.canonical_sha256")
    source_count = _exact_positive_int(
        identity.source_codepoints,
        "input.source_codepoints",
    )
    target_count = _exact_positive_int(
        identity.target_codepoints,
        "input.target_codepoints",
    )
    if source_count > MAX_TEXT_CODEPOINTS or target_count > MAX_TEXT_CODEPOINTS:
        raise BoundaryReportError("input.codepoint_bounds")
    boundary = _exact_nonnegative_int(
        identity.boundary_codepoint,
        "input.boundary_codepoint",
    )
    if boundary != source_count:
        raise BoundaryReportError("input.boundary_codepoint")
    max_length = _exact_nonnegative_int(identity.max_length, "input.max_length")
    if not MIN_MODEL_LENGTH <= max_length <= MAX_MODEL_LENGTH:
        raise BoundaryReportError("input.max_length")


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    """The exact local tokenizer and runtime used by the differential."""

    artifact_id: str
    sha256: str
    byte_count: int
    engine: str
    engine_version: str
    normalizers: tuple[str, ...]
    post_processor: str
    truncation: str

    def __post_init__(self) -> None:
        _validate_artifact_identity(self)


def _validate_artifact_identity(identity: ArtifactIdentity) -> None:
    artifact_id = _exact_string(identity.artifact_id, "artifact.artifact_id")
    digest = _exact_string(identity.sha256, "artifact.sha256")
    byte_count = _exact_positive_int(identity.byte_count, "artifact.byte_count")
    engine = _exact_string(identity.engine, "artifact.engine")
    engine_version = _exact_string(
        identity.engine_version,
        "artifact.engine_version",
    )
    normalizers = _exact_tuple(identity.normalizers, "artifact.normalizers")
    for normalizer in normalizers:
        _exact_string(normalizer, "artifact.normalizers")
    post_processor = _exact_string(
        identity.post_processor,
        "artifact.post_processor",
    )
    truncation = _exact_string(identity.truncation, "artifact.truncation")

    if (
        artifact_id != EXPECTED_ARTIFACT_ID
        or digest != EXPECTED_ARTIFACT_SHA256
        or SHA256_PATTERN.fullmatch(digest) is None
        or byte_count != EXPECTED_ARTIFACT_BYTES
        or engine != EXPECTED_ENGINE
        or engine_version != EXPECTED_ENGINE_VERSION
        or identity.normalizers != EXPECTED_NORMALIZERS
        or post_processor != EXPECTED_POST_PROCESSOR
        or truncation != EXPECTED_TRUNCATION
    ):
        raise BoundaryReportError("artifact.identity")


@dataclass(frozen=True, slots=True)
class LegacyMask:
    """The inherited length-cutoff mask and its exact position partition."""

    algorithm: str
    cutoff: int
    masked_positions: tuple[int, ...]
    supervised_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_legacy_mask(self)


def _validate_legacy_mask(mask: LegacyMask) -> None:
    algorithm = _exact_string(mask.algorithm, "legacy_mask.algorithm")
    if algorithm != EXPECTED_LEGACY_ALGORITHM:
        raise BoundaryReportError("legacy_mask.algorithm")
    _exact_nonnegative_int(mask.cutoff, "legacy_mask.cutoff")
    _position_tuple(mask.masked_positions, "legacy_mask.masked_positions")
    _position_tuple(
        mask.supervised_positions,
        "legacy_mask.supervised_positions",
    )
    if set(mask.masked_positions) & set(mask.supervised_positions):
        raise BoundaryReportError("legacy_mask.overlap")


@dataclass(frozen=True, slots=True)
class Diagnostics:
    """Derived ownership, truncation, and indeterminacy diagnostics."""

    oracle_cutoff: int | None
    prompt_leakage_positions: tuple[int, ...]
    masked_retained_target_positions: tuple[int, ...]
    cross_boundary_positions: tuple[int, ...]
    ambiguous_positions: tuple[int, ...]
    full_target_token_count: int
    retained_target_token_count: int
    supervised_target_token_count: int
    truncated_target_token_count: int
    elimination_causes: tuple[str, ...]
    indeterminate_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_diagnostics(self)


def _validate_diagnostics(diagnostics: Diagnostics) -> None:
    if diagnostics.oracle_cutoff is not None:
        _exact_nonnegative_int(
            diagnostics.oracle_cutoff,
            "diagnostics.oracle_cutoff",
        )
    for position_values, code in (
        (
            diagnostics.prompt_leakage_positions,
            "diagnostics.prompt_leakage_positions",
        ),
        (
            diagnostics.masked_retained_target_positions,
            "diagnostics.masked_retained_target_positions",
        ),
        (
            diagnostics.cross_boundary_positions,
            "diagnostics.cross_boundary_positions",
        ),
        (
            diagnostics.ambiguous_positions,
            "diagnostics.ambiguous_positions",
        ),
    ):
        _position_tuple(position_values, code)
    for count_value, code in (
        (
            diagnostics.full_target_token_count,
            "diagnostics.full_target_token_count",
        ),
        (
            diagnostics.retained_target_token_count,
            "diagnostics.retained_target_token_count",
        ),
        (
            diagnostics.supervised_target_token_count,
            "diagnostics.supervised_target_token_count",
        ),
        (
            diagnostics.truncated_target_token_count,
            "diagnostics.truncated_target_token_count",
        ),
    ):
        _exact_nonnegative_int(count_value, code)
    _code_tuple(
        diagnostics.elimination_causes,
        "diagnostics.elimination_causes",
        canonical_order=ELIMINATION_CAUSE_ORDER,
    )
    _code_tuple(
        diagnostics.indeterminate_reasons,
        "diagnostics.indeterminate_reasons",
    )
    if any(
        reason not in INDETERMINATE_REASONS
        for reason in diagnostics.indeterminate_reasons
    ):
        raise BoundaryReportError("diagnostics.indeterminate_reasons")


@dataclass(frozen=True, slots=True)
class BoundaryReport:
    """A closed differential report with redundant invariants rechecked."""

    status: str
    primary_classification: str
    classifications: tuple[str, ...]
    input_identity: InputIdentity
    artifact_identity: ArtifactIdentity
    source_normalization: NormalizationSnapshot
    combined_normalization: NormalizationSnapshot
    source_truncated: EncodingSnapshot
    combined_full: EncodingSnapshot
    combined_truncated: EncodingSnapshot
    legacy_mask: LegacyMask
    full_attribution: tuple[TokenAttribution, ...]
    truncated_attribution: tuple[TokenAttribution, ...]
    diagnostics: Diagnostics

    def __post_init__(self) -> None:
        _validate_boundary_report(self)


def _require_exact_instance(value: object, expected: type[object], code: str) -> None:
    if type(value) is not expected:
        raise BoundaryReportError(code)


def _validate_attribution_vector(
    value: object,
    *,
    expected_length: int,
    raw_codepoint_count: int,
    boundary: int,
    special_tokens_mask: tuple[int, ...],
    offsets: tuple[tuple[int, int], ...],
    code: str,
) -> tuple[TokenAttribution, ...]:
    items = _exact_tuple(value, code)
    if len(items) != expected_length:
        raise BoundaryReportError(f"{code}.length")
    result: list[TokenAttribution] = []
    claimed_origins: set[int] = set()
    previous_origin_max: int | None = None
    for expected_position, item in enumerate(items):
        if type(item) is not TokenAttribution:
            raise BoundaryReportError(f"{code}.type")
        assert isinstance(item, TokenAttribution)
        _validate_token_attribution(item)
        if item.position != expected_position:
            raise BoundaryReportError(f"{code}.positions")
        if any(origin >= raw_codepoint_count for origin in item.raw_origins):
            raise BoundaryReportError(f"{code}.raw_origins")
        is_injected_prefix = item.ownership == "injected_prefix"
        if (special_tokens_mask[expected_position] == 1) != is_injected_prefix:
            raise BoundaryReportError(f"{code}.special")

        has_source = any(origin < boundary for origin in item.raw_origins)
        has_target = any(origin >= boundary for origin in item.raw_origins)
        expected_ownership: str | None = None
        if item.ownership == "source":
            expected_ownership = "source" if has_source and not has_target else None
        elif item.ownership == "target":
            expected_ownership = "target" if has_target and not has_source else None
        elif item.ownership == "cross_boundary":
            expected_ownership = "cross_boundary" if has_source and has_target else None
        elif item.ownership == "ambiguous" and item.raw_origins:
            start, end = offsets[expected_position]
            if 0 <= start < end <= raw_codepoint_count and set(
                range(start, end)
            ).issubset(item.raw_origins):
                raise BoundaryReportError(f"{code}.ownership")
        if item.ownership in {"source", "target", "cross_boundary"}:
            if expected_ownership != item.ownership:
                raise BoundaryReportError(f"{code}.ownership")
            start, end = offsets[expected_position]
            if not 0 <= start < end <= raw_codepoint_count or not set(
                range(start, end)
            ).issubset(item.raw_origins):
                raise BoundaryReportError(f"{code}.offset")
        if item.raw_origins:
            if claimed_origins.intersection(item.raw_origins):
                raise BoundaryReportError(f"{code}.origin_overlap")
            if (
                previous_origin_max is not None
                and item.raw_origins[0] <= previous_origin_max
            ):
                raise BoundaryReportError(f"{code}.origin_order")
            claimed_origins.update(item.raw_origins)
            previous_origin_max = item.raw_origins[-1]
        result.append(item)
    return tuple(result)


def _encoding_is_exact_prefix(
    full: EncodingSnapshot,
    truncated: EncodingSnapshot,
) -> bool:
    size = len(truncated.ids)
    if size > len(full.ids):
        return False
    return (
        truncated.ids == full.ids[:size]
        and truncated.pieces == full.pieces[:size]
        and truncated.offsets == full.offsets[:size]
        and truncated.type_ids == full.type_ids[:size]
        and truncated.special_tokens_mask == full.special_tokens_mask[:size]
        and truncated.attention_mask == full.attention_mask[:size]
    )


def _expected_classifications(report: BoundaryReport) -> tuple[str, ...]:
    diagnostics = report.diagnostics
    supervised_set = set(report.legacy_mask.supervised_positions)
    supervised_ambiguous = any(
        item.ownership == "ambiguous" and item.position in supervised_set
        for item in report.truncated_attribution
    )
    truncated_ownership_unknown = (
        "truncation.not_exact_prefix" in diagnostics.indeterminate_reasons
    )
    observed: set[str] = set()
    if diagnostics.indeterminate_reasons or diagnostics.ambiguous_positions:
        observed.add("indeterminate")
    if diagnostics.cross_boundary_positions:
        observed.add("cross_boundary_token")
    if not truncated_ownership_unknown:
        if (
            diagnostics.full_target_token_count > 0
            and diagnostics.supervised_target_token_count == 0
            and not supervised_ambiguous
        ):
            observed.add("target_eliminated")
        if (
            0
            < diagnostics.retained_target_token_count
            < diagnostics.full_target_token_count
        ):
            observed.add("partial_target_truncation")
        if (
            diagnostics.prompt_leakage_positions
            or diagnostics.masked_retained_target_positions
            or (
                diagnostics.oracle_cutoff is not None
                and report.legacy_mask.cutoff != diagnostics.oracle_cutoff
            )
        ):
            observed.add("boundary_drift")
    if not observed:
        observed.add("aligned")
    return tuple(
        classification
        for classification in CLASSIFICATION_PRECEDENCE
        if classification in observed
    )


def _validate_boundary_report(report: BoundaryReport) -> None:
    status = _exact_string(report.status, "report.status")
    if status not in REPORT_STATUSES:
        raise BoundaryReportError("report.status")
    primary = _exact_string(
        report.primary_classification,
        "report.primary_classification",
    )
    if primary not in CLASSIFICATIONS:
        raise BoundaryReportError("report.primary_classification")
    classifications = _code_tuple(
        report.classifications,
        "report.classifications",
        canonical_order=CLASSIFICATION_PRECEDENCE,
    )
    if not classifications or primary != classifications[0]:
        raise BoundaryReportError("report.primary_classification")
    if "aligned" in classifications and len(classifications) != 1:
        raise BoundaryReportError("report.classifications")

    for value, expected, code in (
        (report.input_identity, InputIdentity, "report.input_identity"),
        (report.artifact_identity, ArtifactIdentity, "report.artifact_identity"),
        (
            report.source_normalization,
            NormalizationSnapshot,
            "report.source_normalization",
        ),
        (
            report.combined_normalization,
            NormalizationSnapshot,
            "report.combined_normalization",
        ),
        (
            report.source_truncated,
            EncodingSnapshot,
            "report.source_truncated",
        ),
        (report.combined_full, EncodingSnapshot, "report.combined_full"),
        (
            report.combined_truncated,
            EncodingSnapshot,
            "report.combined_truncated",
        ),
        (report.legacy_mask, LegacyMask, "report.legacy_mask"),
        (report.diagnostics, Diagnostics, "report.diagnostics"),
    ):
        _require_exact_instance(value, expected, code)

    _validate_input_identity(report.input_identity)
    _validate_artifact_identity(report.artifact_identity)
    _validate_normalization_snapshot(report.source_normalization)
    _validate_normalization_snapshot(report.combined_normalization)
    _validate_encoding_snapshot(report.source_truncated)
    _validate_encoding_snapshot(report.combined_full)
    _validate_encoding_snapshot(report.combined_truncated)
    _validate_legacy_mask(report.legacy_mask)
    _validate_diagnostics(report.diagnostics)

    input_identity = report.input_identity
    total_raw_count = (
        input_identity.source_codepoints + input_identity.target_codepoints
    )
    if (
        report.source_normalization.raw_codepoint_count
        != input_identity.source_codepoints
        or report.combined_normalization.raw_codepoint_count != total_raw_count
    ):
        raise BoundaryReportError("report.normalization_counts")
    if report.source_normalization.cross_boundary_output_positions:
        raise BoundaryReportError("report.source_cross_boundary")
    for special, (start, end) in zip(
        report.source_truncated.special_tokens_mask,
        report.source_truncated.offsets,
        strict=True,
    ):
        if special == 0 and not 0 <= start < end <= input_identity.source_codepoints:
            raise BoundaryReportError("report.source_offsets")

    source_size = len(report.source_truncated.ids)
    full_size = len(report.combined_full.ids)
    retained_size = len(report.combined_truncated.ids)
    if (
        source_size > input_identity.max_length
        or retained_size > input_identity.max_length
    ):
        raise BoundaryReportError("report.truncation_length")
    if report.legacy_mask.cutoff != source_size:
        raise BoundaryReportError("report.legacy_cutoff")
    expected_masked = tuple(range(min(source_size, retained_size)))
    expected_supervised = tuple(range(min(source_size, retained_size), retained_size))
    if report.legacy_mask.masked_positions != expected_masked:
        raise BoundaryReportError("report.masked_positions")
    if report.legacy_mask.supervised_positions != expected_supervised:
        raise BoundaryReportError("report.supervised_positions")

    full_attribution = _validate_attribution_vector(
        report.full_attribution,
        expected_length=full_size,
        raw_codepoint_count=total_raw_count,
        boundary=input_identity.boundary_codepoint,
        special_tokens_mask=report.combined_full.special_tokens_mask,
        offsets=report.combined_full.offsets,
        code="report.full_attribution",
    )
    truncated_attribution = _validate_attribution_vector(
        report.truncated_attribution,
        expected_length=retained_size,
        raw_codepoint_count=total_raw_count,
        boundary=input_identity.boundary_codepoint,
        special_tokens_mask=report.combined_truncated.special_tokens_mask,
        offsets=report.combined_truncated.offsets,
        code="report.truncated_attribution",
    )

    exact_prefix = _encoding_is_exact_prefix(
        report.combined_full,
        report.combined_truncated,
    )
    if exact_prefix:
        if "truncation.not_exact_prefix" in report.diagnostics.indeterminate_reasons:
            raise BoundaryReportError("report.encoding_prefix_reason")
        if truncated_attribution != full_attribution[:retained_size]:
            raise BoundaryReportError("report.attribution_prefix")
    elif "truncation.not_exact_prefix" not in (
        report.diagnostics.indeterminate_reasons
    ):
        raise BoundaryReportError("report.encoding_prefix")

    masked_set = set(report.legacy_mask.masked_positions)
    supervised_set = set(report.legacy_mask.supervised_positions)
    expected_prompt_leakage = tuple(
        item.position
        for item in truncated_attribution
        if item.ownership == "source" and item.position in supervised_set
    )
    expected_masked_target = tuple(
        item.position
        for item in truncated_attribution
        if item.ownership == "target" and item.position in masked_set
    )
    expected_cross = tuple(
        item.position for item in full_attribution if item.ownership == "cross_boundary"
    )
    expected_ambiguous = tuple(
        item.position for item in full_attribution if item.ownership == "ambiguous"
    )
    first_definite_target = next(
        (item.position for item in truncated_attribution if item.ownership == "target"),
        None,
    )
    expected_oracle = (
        None
        if first_definite_target is not None
        and any(
            item.ownership == "ambiguous" and item.position < first_definite_target
            for item in truncated_attribution
        )
        else first_definite_target
    )
    diagnostics = report.diagnostics
    if diagnostics.prompt_leakage_positions != expected_prompt_leakage:
        raise BoundaryReportError("report.prompt_leakage")
    if diagnostics.masked_retained_target_positions != expected_masked_target:
        raise BoundaryReportError("report.masked_target")
    if diagnostics.cross_boundary_positions != expected_cross:
        raise BoundaryReportError("report.cross_boundary_positions")
    if diagnostics.ambiguous_positions != expected_ambiguous:
        raise BoundaryReportError("report.ambiguous_positions")
    if diagnostics.oracle_cutoff != expected_oracle:
        raise BoundaryReportError("report.oracle_cutoff")
    unknown_token_present = any(
        token_id == 0
        for snapshot in (
            report.source_truncated,
            report.combined_full,
            report.combined_truncated,
        )
        for token_id in snapshot.ids
    )
    unknown_token_reason = "encoding.unknown_token" in diagnostics.indeterminate_reasons
    if unknown_token_present != unknown_token_reason:
        raise BoundaryReportError("report.unknown_token_reason")
    source_piece_count = sum(
        len(piece)
        for piece, special in zip(
            report.source_truncated.pieces,
            report.source_truncated.special_tokens_mask,
            strict=True,
        )
        if special == 0
    )
    combined_piece_count = sum(
        len(piece)
        for piece, special in zip(
            report.combined_full.pieces,
            report.combined_full.special_tokens_mask,
            strict=True,
        )
        if special == 0
    )
    normalization_incomplete = (
        not report.source_normalization.provenance_complete
        or not report.combined_normalization.provenance_complete
    )
    source_piece_mismatch = source_piece_count > (
        report.source_normalization.normalized_codepoint_count
    ) or (
        source_size < input_identity.max_length
        and source_piece_count != report.source_normalization.normalized_codepoint_count
    )
    piece_alignment_observed = (
        source_piece_mismatch
        or combined_piece_count
        != report.combined_normalization.normalized_codepoint_count
        or (
            report.combined_normalization.provenance_complete
            and not unknown_token_present
            and any(
                item.ownership == "ambiguous" and not item.raw_origins
                for item in full_attribution
            )
        )
    )
    piece_alignment_reason = (
        "encoding.piece_alignment" in diagnostics.indeterminate_reasons
    )
    if piece_alignment_observed and not piece_alignment_reason:
        raise BoundaryReportError("report.piece_alignment_reason")
    if (
        piece_alignment_reason
        and not piece_alignment_observed
        and not normalization_incomplete
    ):
        raise BoundaryReportError("report.piece_alignment_reason")

    offset_issue_observed = any(
        item.ownership == "ambiguous"
        and bool(item.raw_origins)
        and (
            not 0 <= start < end <= total_raw_count
            or not set(range(start, end)).issubset(item.raw_origins)
        )
        for item, (start, end) in zip(
            full_attribution,
            report.combined_full.offsets,
            strict=True,
        )
    )
    offset_issue_reason = (
        "encoding.offset_provenance" in diagnostics.indeterminate_reasons
    )
    if offset_issue_observed != offset_issue_reason:
        raise BoundaryReportError("report.offset_reason")

    normalization_reason_codes = {
        "normalization.cluster_replay",
        "normalization.provenance_ambiguous",
        "normalization.runtime_mismatch",
    }
    normalization_reason_present = bool(
        normalization_reason_codes.intersection(diagnostics.indeterminate_reasons)
    )
    if normalization_incomplete != normalization_reason_present:
        raise BoundaryReportError("report.normalization_reason")
    if expected_ambiguous and not diagnostics.indeterminate_reasons:
        raise BoundaryReportError("report.ambiguous_reason")
    if (
        "normalization.runtime_mismatch" in diagnostics.indeterminate_reasons
        and report.source_normalization.provenance_complete
        and report.combined_normalization.provenance_complete
    ):
        raise BoundaryReportError("report.runtime_mismatch_reason")
    if (
        report.combined_normalization.cross_boundary_output_positions
        and not expected_cross
        and not diagnostics.indeterminate_reasons
    ):
        raise BoundaryReportError("report.normalization_cross_boundary")

    expected_full_target_count = sum(
        item.ownership == "target" for item in full_attribution
    )
    expected_retained_target_count = sum(
        item.ownership == "target" for item in truncated_attribution
    )
    expected_supervised_target_count = sum(
        item.ownership == "target" and item.position in supervised_set
        for item in truncated_attribution
    )
    supervised_ambiguous = any(
        item.ownership == "ambiguous" and item.position in supervised_set
        for item in truncated_attribution
    )
    if diagnostics.full_target_token_count != expected_full_target_count:
        raise BoundaryReportError("report.full_target_count")
    if diagnostics.retained_target_token_count != expected_retained_target_count:
        raise BoundaryReportError("report.retained_target_count")
    if diagnostics.supervised_target_token_count != expected_supervised_target_count:
        raise BoundaryReportError("report.supervised_target_count")
    if expected_retained_target_count > expected_full_target_count:
        raise BoundaryReportError("report.target_count_order")
    expected_truncated_count = (
        expected_full_target_count - expected_retained_target_count
    )
    if diagnostics.truncated_target_token_count != expected_truncated_count:
        raise BoundaryReportError("report.truncated_target_count")

    truncated_ownership_unknown = (
        "truncation.not_exact_prefix" in diagnostics.indeterminate_reasons
    )
    target_eliminated = (
        not truncated_ownership_unknown
        and expected_full_target_count > 0
        and expected_supervised_target_count == 0
        and not supervised_ambiguous
    )
    expected_causes: list[str] = []
    if target_eliminated:
        if expected_retained_target_count < expected_full_target_count:
            expected_causes.append("truncation")
        if expected_retained_target_count > 0:
            expected_causes.append("legacy_cutoff")
    if diagnostics.elimination_causes != tuple(expected_causes):
        raise BoundaryReportError("report.elimination_causes")

    target_material_absent = (
        expected_full_target_count == 0
        and not expected_cross
        and not expected_ambiguous
    )
    target_absence_reason = (
        "target.no_attributable_token" in diagnostics.indeterminate_reasons
    )
    if target_material_absent != target_absence_reason:
        raise BoundaryReportError("report.target_material")
    if not diagnostics.indeterminate_reasons and (
        source_piece_mismatch
        or combined_piece_count
        != report.combined_normalization.normalized_codepoint_count
    ):
        raise BoundaryReportError("report.normalized_piece_count")

    expected_classifications = _expected_classifications(report)
    if classifications != expected_classifications:
        raise BoundaryReportError("report.classifications")
    expected_status = (
        "indeterminate"
        if expected_classifications[0] == "indeterminate"
        else "pass"
        if expected_classifications[0] == "aligned"
        else "fail"
    )
    if status != expected_status:
        raise BoundaryReportError("report.status")


def _encoding_to_primitive(snapshot: EncodingSnapshot) -> dict[str, JSONValue]:
    return {
        "ids": list(snapshot.ids),
        "pieces": list(snapshot.pieces),
        "offsets": [[start, end] for start, end in snapshot.offsets],
        "type_ids": list(snapshot.type_ids),
        "special_tokens_mask": list(snapshot.special_tokens_mask),
        "attention_mask": list(snapshot.attention_mask),
    }


def _normalization_to_primitive(
    snapshot: NormalizationSnapshot,
) -> dict[str, JSONValue]:
    return {
        "raw_codepoint_count": snapshot.raw_codepoint_count,
        "nfc_codepoint_count": snapshot.nfc_codepoint_count,
        "normalized_codepoint_count": snapshot.normalized_codepoint_count,
        "trailing_normalized_codepoints_removed": (
            snapshot.trailing_normalized_codepoints_removed
        ),
        "cross_boundary_output_positions": list(
            snapshot.cross_boundary_output_positions
        ),
        "provenance_complete": snapshot.provenance_complete,
    }


def _attribution_to_primitive(
    attribution: tuple[TokenAttribution, ...],
) -> list[JSONValue]:
    return [
        {
            "position": item.position,
            "ownership": item.ownership,
            "raw_origins": list(item.raw_origins),
        }
        for item in attribution
    ]


def boundary_report_to_primitive(report: BoundaryReport) -> dict[str, JSONValue]:
    """Return the exact closed JSON-compatible report document."""

    if type(report) is not BoundaryReport:
        raise BoundaryReportError("report.type")
    _validate_boundary_report(report)
    diagnostics = report.diagnostics
    scope: dict[str, JSONValue] = {field: False for field in SCOPE_FALSE_FIELDS}
    scope.update(
        {
            "artifact_origin": "locally-authored-synthetic",
            "case_class": "caller-authored-synthetic",
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "status": report.status,
        "primary_classification": report.primary_classification,
        "classifications": list(report.classifications),
        "input_identity": {
            "case_id": report.input_identity.case_id,
            "canonical_sha256": report.input_identity.canonical_sha256,
            "source_codepoints": report.input_identity.source_codepoints,
            "target_codepoints": report.input_identity.target_codepoints,
            "boundary_codepoint": report.input_identity.boundary_codepoint,
            "max_length": report.input_identity.max_length,
        },
        "artifact_identity": {
            "artifact_id": report.artifact_identity.artifact_id,
            "sha256": report.artifact_identity.sha256,
            "byte_count": report.artifact_identity.byte_count,
            "engine": report.artifact_identity.engine,
            "engine_version": report.artifact_identity.engine_version,
            "normalizers": list(report.artifact_identity.normalizers),
            "post_processor": report.artifact_identity.post_processor,
            "truncation": report.artifact_identity.truncation,
        },
        "normalization": {
            "source": _normalization_to_primitive(report.source_normalization),
            "combined": _normalization_to_primitive(report.combined_normalization),
        },
        "encodings": {
            "source_truncated": _encoding_to_primitive(report.source_truncated),
            "combined_full": _encoding_to_primitive(report.combined_full),
            "combined_truncated": _encoding_to_primitive(report.combined_truncated),
        },
        "legacy_mask": {
            "algorithm": report.legacy_mask.algorithm,
            "cutoff": report.legacy_mask.cutoff,
            "masked_positions": list(report.legacy_mask.masked_positions),
            "supervised_positions": list(report.legacy_mask.supervised_positions),
        },
        "attribution": {
            "full": _attribution_to_primitive(report.full_attribution),
            "truncated": _attribution_to_primitive(report.truncated_attribution),
        },
        "diagnostics": {
            "oracle_cutoff": diagnostics.oracle_cutoff,
            "prompt_leakage_positions": list(diagnostics.prompt_leakage_positions),
            "masked_retained_target_positions": list(
                diagnostics.masked_retained_target_positions
            ),
            "cross_boundary_positions": list(diagnostics.cross_boundary_positions),
            "ambiguous_positions": list(diagnostics.ambiguous_positions),
            "full_target_token_count": diagnostics.full_target_token_count,
            "retained_target_token_count": (diagnostics.retained_target_token_count),
            "supervised_target_token_count": (
                diagnostics.supervised_target_token_count
            ),
            "truncated_target_token_count": (diagnostics.truncated_target_token_count),
            "elimination_causes": list(diagnostics.elimination_causes),
            "indeterminate_reasons": list(diagnostics.indeterminate_reasons),
        },
        "scope": scope,
    }


def canonical_boundary_report_bytes(report: BoundaryReport) -> bytes:
    """Return sorted compact ASCII JSON terminated by one newline."""

    return (
        json.dumps(
            boundary_report_to_primitive(report),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def boundary_report_sha256(report: BoundaryReport) -> str:
    """Return the SHA-256 identity of the complete canonical report."""

    return hashlib.sha256(canonical_boundary_report_bytes(report)).hexdigest()
