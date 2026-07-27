"""Offline differential analysis over the one trusted tokenizer artifact."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Final

from tokenizers import Encoding, Tokenizer  # type: ignore[import-untyped]

from .artifact import (
    ARTIFACT_ID,
    ARTIFACT_SHA256,
    RUNTIME_PACKAGE,
    RUNTIME_VERSION,
    ArtifactVerificationError,
    VerifiedArtifact,
    verify_local_artifact,
)
from .contract import BoundaryCase, boundary_case_sha256
from .errors import BoundaryEngineError
from .provenance import NormalizationTrace, ProvenanceError, trace_normalization
from .report import (
    CLASSIFICATION_PRECEDENCE,
    ArtifactIdentity,
    BoundaryReport,
    BoundaryReportError,
    Diagnostics,
    EncodingSnapshot,
    InputIdentity,
    LegacyMask,
    NormalizationSnapshot,
    TokenAttribution,
)

UNKNOWN_ID: Final = 0
BOS_ID: Final = 1
UNKNOWN_PIECE: Final = "<unk>"
BOS_PIECE: Final = "<bos>"
LEGACY_ALGORITHM: Final = "standalone-source-token-count"


@dataclass(frozen=True, slots=True)
class _CapturedEncoding:
    snapshot: EncodingSnapshot
    normalized: str


def _installed_runtime_version() -> str:
    lookup_failed = False
    try:
        detected = metadata.version(RUNTIME_PACKAGE)
    except metadata.PackageNotFoundError:
        lookup_failed = True
        detected = ""
    if lookup_failed:
        raise BoundaryEngineError("runtime.unavailable")
    if detected != RUNTIME_VERSION:
        raise BoundaryEngineError("runtime.version")
    return detected


def _new_tokenizer(payload: bytes) -> Tokenizer:
    load_failed = False
    try:
        tokenizer = Tokenizer.from_str(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, ValueError):
        load_failed = True
        tokenizer = None
    if load_failed or tokenizer is None:
        raise BoundaryEngineError("runtime.artifact_load")
    return tokenizer


def _encoding_snapshot(encoding: Encoding) -> EncodingSnapshot:
    rejected = False
    try:
        snapshot = EncodingSnapshot(
            ids=tuple(encoding.ids),
            pieces=tuple(encoding.tokens),
            offsets=tuple((start, end) for start, end in encoding.offsets),
            type_ids=tuple(encoding.type_ids),
            special_tokens_mask=tuple(encoding.special_tokens_mask),
            attention_mask=tuple(encoding.attention_mask),
        )
    except (
        AttributeError,
        BoundaryReportError,
        OverflowError,
        TypeError,
        ValueError,
    ):
        rejected = True
        snapshot = None
    if rejected or snapshot is None:
        raise BoundaryEngineError("runtime.encoding")
    return snapshot


def _capture_encoding(
    artifact: VerifiedArtifact,
    text: str,
    max_length: int | None,
) -> _CapturedEncoding:
    tokenizer = _new_tokenizer(artifact.payload)
    if max_length is not None:
        configuration_failed = False
        try:
            tokenizer.enable_truncation(
                max_length=max_length,
                stride=0,
                strategy="longest_first",
                direction="right",
            )
        except (TypeError, ValueError):
            configuration_failed = True
        if configuration_failed:
            raise BoundaryEngineError("runtime.truncation")

    normalizer = tokenizer.normalizer
    if normalizer is None:
        raise BoundaryEngineError("runtime.normalizer")

    execution_failed = False
    try:
        normalized = normalizer.normalize_str(text)
        encoding = tokenizer.encode(text, add_special_tokens=True)
    except (TypeError, ValueError):
        execution_failed = True
        normalized = ""
        encoding = None
    if execution_failed or encoding is None:
        raise BoundaryEngineError("runtime.encode")
    return _CapturedEncoding(
        snapshot=_encoding_snapshot(encoding),
        normalized=normalized,
    )


def _exact_prefix(full: EncodingSnapshot, truncated: EncodingSnapshot) -> bool:
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


def _normalization_snapshot(
    trace: NormalizationTrace,
    *,
    runtime_matches: bool,
) -> NormalizationSnapshot:
    return NormalizationSnapshot(
        raw_codepoint_count=trace.raw_codepoint_count,
        nfc_codepoint_count=trace.nfc_codepoint_count,
        normalized_codepoint_count=len(trace.normalized),
        trailing_normalized_codepoints_removed=(
            trace.trailing_normalized_codepoints_removed
        ),
        cross_boundary_output_positions=(
            trace.cross_boundary_output_positions if runtime_matches else ()
        ),
        provenance_complete=trace.complete and runtime_matches,
    )


def _ownership(raw_origins: tuple[int, ...], boundary: int) -> str:
    has_source = any(position < boundary for position in raw_origins)
    has_target = any(position >= boundary for position in raw_origins)
    if has_source and has_target:
        return "cross_boundary"
    if has_source:
        return "source"
    if has_target:
        return "target"
    return "ambiguous"


def _attribute_encoding(
    snapshot: EncodingSnapshot,
    trace: NormalizationTrace,
    *,
    boundary: int,
    raw_codepoint_count: int,
    runtime_matches: bool,
    require_complete_consumption: bool,
) -> tuple[tuple[TokenAttribution, ...], tuple[str, ...]]:
    attributions: list[TokenAttribution] = []
    reasons: set[str] = set(trace.indeterminate_reasons)
    if not runtime_matches:
        reasons.add("normalization.runtime_mismatch")

    cursor = 0
    alignment_failed = False
    provenance_usable = trace.complete and runtime_matches
    for position, (
        token_id,
        piece,
        offset,
        special_mask,
    ) in enumerate(
        zip(
            snapshot.ids,
            snapshot.pieces,
            snapshot.offsets,
            snapshot.special_tokens_mask,
            strict=True,
        )
    ):
        if special_mask == 1:
            attributions.append(
                TokenAttribution(
                    position=position,
                    ownership="injected_prefix",
                    raw_origins=(),
                )
            )
            continue

        if alignment_failed or token_id == UNKNOWN_ID or piece == UNKNOWN_PIECE:
            alignment_failed = True
            reasons.add(
                "encoding.unknown_token"
                if token_id == UNKNOWN_ID or piece == UNKNOWN_PIECE
                else "encoding.piece_alignment"
            )
            attributions.append(
                TokenAttribution(
                    position=position,
                    ownership="ambiguous",
                    raw_origins=(),
                )
            )
            continue
        if not piece or not trace.normalized.startswith(piece, cursor):
            alignment_failed = True
            reasons.add("encoding.piece_alignment")
            attributions.append(
                TokenAttribution(
                    position=position,
                    ownership="ambiguous",
                    raw_origins=(),
                )
            )
            continue

        piece_origins = trace.origins[cursor : cursor + len(piece)]
        cursor += len(piece)
        if not provenance_usable or any(origin is None for origin in piece_origins):
            reasons.add("normalization.provenance_ambiguous")
            attributions.append(
                TokenAttribution(
                    position=position,
                    ownership="ambiguous",
                    raw_origins=(),
                )
            )
            continue

        raw_origins = tuple(
            sorted(
                {
                    raw_position
                    for origin in piece_origins
                    if origin is not None
                    for raw_position in origin
                }
            )
        )
        start, end = offset
        offset_is_valid = 0 <= start < end <= raw_codepoint_count and set(
            range(start, end)
        ).issubset(raw_origins)
        if not raw_origins or not offset_is_valid:
            reasons.add("encoding.offset_provenance")
            ownership = "ambiguous"
        else:
            ownership = _ownership(raw_origins, boundary)
        attributions.append(
            TokenAttribution(
                position=position,
                ownership=ownership,
                raw_origins=raw_origins,
            )
        )

    if require_complete_consumption and cursor != len(trace.normalized):
        reasons.add("encoding.piece_alignment")
    return tuple(attributions), tuple(sorted(reasons))


def _verified_artifact() -> VerifiedArtifact:
    verification_code: str | None = None
    try:
        verified = verify_local_artifact()
    except ArtifactVerificationError as error:
        verification_code = error.code
        verified = None
    if verification_code is not None or verified is None:
        raise BoundaryEngineError(verification_code or "artifact.verification")
    return verified


def _normalization_trace(text: str, boundary: int) -> NormalizationTrace:
    failure_code: str | None = None
    try:
        trace = trace_normalization(text, boundary)
    except ProvenanceError as error:
        failure_code = error.code
        trace = None
    if failure_code is not None or trace is None:
        raise BoundaryEngineError(f"provenance.{failure_code or 'trace'}")
    return trace


def _classification_tuple(observed: set[str]) -> tuple[str, ...]:
    if not observed:
        observed.add("aligned")
    return tuple(
        classification
        for classification in CLASSIFICATION_PRECEDENCE
        if classification in observed
    )


def analyze_boundary(case: BoundaryCase) -> BoundaryReport:
    """Execute the legacy cutoff and an independent ownership oracle."""

    if type(case) is not BoundaryCase:
        raise BoundaryEngineError("case.type")

    runtime_version = _installed_runtime_version()
    artifact = _verified_artifact()
    combined_text = case.source + case.target
    boundary = len(case.source)

    source_trace = _normalization_trace(case.source, boundary)
    combined_trace = _normalization_trace(combined_text, boundary)
    source_capture = _capture_encoding(
        artifact,
        case.source,
        case.max_length,
    )
    full_capture = _capture_encoding(artifact, combined_text, None)
    truncated_capture = _capture_encoding(
        artifact,
        combined_text,
        case.max_length,
    )

    source_runtime_matches = source_capture.normalized == source_trace.normalized
    full_runtime_matches = full_capture.normalized == combined_trace.normalized
    truncated_runtime_matches = (
        truncated_capture.normalized == combined_trace.normalized
    )

    _, source_reasons = _attribute_encoding(
        source_capture.snapshot,
        source_trace,
        boundary=boundary,
        raw_codepoint_count=len(case.source),
        runtime_matches=source_runtime_matches,
        require_complete_consumption=(
            len(source_capture.snapshot.ids) < case.max_length
        ),
    )
    full_attribution, full_reasons = _attribute_encoding(
        full_capture.snapshot,
        combined_trace,
        boundary=boundary,
        raw_codepoint_count=len(combined_text),
        runtime_matches=full_runtime_matches,
        require_complete_consumption=True,
    )
    exact_prefix = _exact_prefix(
        full_capture.snapshot,
        truncated_capture.snapshot,
    )
    if exact_prefix:
        truncated_attribution = full_attribution[: len(truncated_capture.snapshot.ids)]
        truncated_reasons: tuple[str, ...] = ()
    else:
        truncated_attribution = tuple(
            TokenAttribution(
                position=position,
                ownership=(
                    "injected_prefix"
                    if position == 0
                    and token_id == BOS_ID
                    and piece == BOS_PIECE
                    and offset == (0, 0)
                    and special_mask == 1
                    else "ambiguous"
                ),
                raw_origins=(),
            )
            for position, (token_id, piece, offset, special_mask) in enumerate(
                zip(
                    truncated_capture.snapshot.ids,
                    truncated_capture.snapshot.pieces,
                    truncated_capture.snapshot.offsets,
                    truncated_capture.snapshot.special_tokens_mask,
                    strict=True,
                )
            )
        )
        truncated_reasons = ("truncation.not_exact_prefix",)

    indeterminate_reasons = set(source_reasons)
    indeterminate_reasons.update(full_reasons)
    indeterminate_reasons.update(truncated_reasons)
    if not truncated_runtime_matches:
        indeterminate_reasons.add("normalization.runtime_mismatch")
    if not exact_prefix:
        indeterminate_reasons.add("truncation.not_exact_prefix")

    legacy_cutoff = len(source_capture.snapshot.ids)
    retained_size = len(truncated_capture.snapshot.ids)
    masked_positions = tuple(range(min(legacy_cutoff, retained_size)))
    supervised_positions = tuple(
        range(min(legacy_cutoff, retained_size), retained_size)
    )
    masked_set = set(masked_positions)
    supervised_set = set(supervised_positions)

    prompt_leakage_positions = tuple(
        item.position
        for item in truncated_attribution
        if item.ownership == "source" and item.position in supervised_set
    )
    masked_retained_target_positions = tuple(
        item.position
        for item in truncated_attribution
        if item.ownership == "target" and item.position in masked_set
    )
    cross_boundary_positions = tuple(
        item.position for item in full_attribution if item.ownership == "cross_boundary"
    )
    ambiguous_positions = tuple(
        item.position for item in full_attribution if item.ownership == "ambiguous"
    )
    first_definite_target = next(
        (item.position for item in truncated_attribution if item.ownership == "target"),
        None,
    )
    oracle_cutoff = (
        None
        if first_definite_target is not None
        and any(
            item.ownership == "ambiguous" and item.position < first_definite_target
            for item in truncated_attribution
        )
        else first_definite_target
    )

    full_target_count = sum(item.ownership == "target" for item in full_attribution)
    retained_target_count = sum(
        item.ownership == "target" for item in truncated_attribution
    )
    supervised_target_count = sum(
        item.ownership == "target" and item.position in supervised_set
        for item in truncated_attribution
    )
    supervised_ambiguous = any(
        item.ownership == "ambiguous" and item.position in supervised_set
        for item in truncated_attribution
    )
    if retained_target_count > full_target_count:
        raise BoundaryEngineError("runtime.target_count")
    truncated_target_count = full_target_count - retained_target_count

    if (
        full_target_count == 0
        and not cross_boundary_positions
        and not ambiguous_positions
    ):
        indeterminate_reasons.add("target.no_attributable_token")

    observed: set[str] = set()
    if indeterminate_reasons or ambiguous_positions:
        observed.add("indeterminate")
    if cross_boundary_positions:
        observed.add("cross_boundary_token")

    elimination_causes: list[str] = []
    target_eliminated = (
        exact_prefix
        and full_target_count > 0
        and supervised_target_count == 0
        and not supervised_ambiguous
    )
    if target_eliminated:
        observed.add("target_eliminated")
        if retained_target_count < full_target_count:
            elimination_causes.append("truncation")
        if retained_target_count > 0:
            elimination_causes.append("legacy_cutoff")
    if exact_prefix and 0 < retained_target_count < full_target_count:
        observed.add("partial_target_truncation")

    boundary_drift = bool(
        prompt_leakage_positions
        or masked_retained_target_positions
        or (oracle_cutoff is not None and legacy_cutoff != oracle_cutoff)
    )
    if boundary_drift:
        observed.add("boundary_drift")

    classifications = _classification_tuple(observed)
    primary = classifications[0]
    status = (
        "indeterminate"
        if primary == "indeterminate"
        else "pass"
        if primary == "aligned"
        else "fail"
    )

    report_failed = False
    try:
        report = BoundaryReport(
            status=status,
            primary_classification=primary,
            classifications=classifications,
            input_identity=InputIdentity(
                case_id=case.case_id,
                canonical_sha256=boundary_case_sha256(case),
                source_codepoints=len(case.source),
                target_codepoints=len(case.target),
                boundary_codepoint=boundary,
                max_length=case.max_length,
            ),
            artifact_identity=ArtifactIdentity(
                artifact_id=ARTIFACT_ID,
                sha256=ARTIFACT_SHA256,
                byte_count=artifact.byte_count,
                engine=RUNTIME_PACKAGE,
                engine_version=runtime_version,
                normalizers=(
                    "NFC",
                    "Strip(left=false,right=true)",
                ),
                post_processor="prefix-bos-only",
                truncation="caller-bounded-right",
            ),
            source_normalization=_normalization_snapshot(
                source_trace,
                runtime_matches=source_runtime_matches,
            ),
            combined_normalization=_normalization_snapshot(
                combined_trace,
                runtime_matches=(full_runtime_matches and truncated_runtime_matches),
            ),
            source_truncated=source_capture.snapshot,
            combined_full=full_capture.snapshot,
            combined_truncated=truncated_capture.snapshot,
            legacy_mask=LegacyMask(
                algorithm=LEGACY_ALGORITHM,
                cutoff=legacy_cutoff,
                masked_positions=masked_positions,
                supervised_positions=supervised_positions,
            ),
            full_attribution=full_attribution,
            truncated_attribution=truncated_attribution,
            diagnostics=Diagnostics(
                oracle_cutoff=oracle_cutoff,
                prompt_leakage_positions=prompt_leakage_positions,
                masked_retained_target_positions=(masked_retained_target_positions),
                cross_boundary_positions=cross_boundary_positions,
                ambiguous_positions=ambiguous_positions,
                full_target_token_count=full_target_count,
                retained_target_token_count=retained_target_count,
                supervised_target_token_count=supervised_target_count,
                truncated_target_token_count=truncated_target_count,
                elimination_causes=tuple(elimination_causes),
                indeterminate_reasons=tuple(sorted(indeterminate_reasons)),
            ),
        )
    except BoundaryReportError:
        report_failed = True
        report = None
    if report_failed or report is None:
        raise BoundaryEngineError("report.invariant")
    return report
