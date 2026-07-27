from __future__ import annotations

import ast
import hashlib
import json
import sys
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from token_boundary.report import (
    CLASSIFICATION_PRECEDENCE,
    EXPECTED_ARTIFACT_SHA256,
    REPORT_KIND,
    SCHEMA_VERSION,
    SCOPE_FALSE_FIELDS,
    ArtifactIdentity,
    BoundaryReport,
    BoundaryReportError,
    Diagnostics,
    EncodingSnapshot,
    InputIdentity,
    LegacyMask,
    NormalizationSnapshot,
    TokenAttribution,
    boundary_report_sha256,
    boundary_report_to_primitive,
    canonical_boundary_report_bytes,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _encoding(
    ids: tuple[int, ...],
    pieces: tuple[str, ...],
    offsets: tuple[tuple[int, int], ...],
) -> EncodingSnapshot:
    size = len(ids)
    return EncodingSnapshot(
        ids=ids,
        pieces=pieces,
        offsets=offsets,
        type_ids=(0,) * size,
        special_tokens_mask=(1,) + (0,) * (size - 1),
        attention_mask=(1,) * size,
    )


def _artifact_identity() -> ArtifactIdentity:
    return ArtifactIdentity(
        artifact_id="local-boundary-bpe-v1",
        sha256=EXPECTED_ARTIFACT_SHA256,
        byte_count=1533,
        engine="tokenizers",
        engine_version="0.21.4",
        normalizers=("NFC", "Strip(left=false,right=true)"),
        post_processor="prefix-bos-only",
        truncation="caller-bounded-right",
    )


def _aligned_report() -> BoundaryReport:
    source = _encoding(
        (1, 5),
        ("<bos>", "c"),
        ((0, 0), (0, 1)),
    )
    combined = _encoding(
        (1, 5, 6, 7),
        ("<bos>", "c", "d", "e"),
        ((0, 0), (0, 1), (1, 2), (2, 3)),
    )
    attribution = (
        TokenAttribution(0, "injected_prefix", ()),
        TokenAttribution(1, "source", (0,)),
        TokenAttribution(2, "target", (1,)),
        TokenAttribution(3, "target", (2,)),
    )
    return BoundaryReport(
        status="pass",
        primary_classification="aligned",
        classifications=("aligned",),
        input_identity=InputIdentity(
            case_id="aligned",
            canonical_sha256="a" * 64,
            source_codepoints=1,
            target_codepoints=2,
            boundary_codepoint=1,
            max_length=8,
        ),
        artifact_identity=_artifact_identity(),
        source_normalization=NormalizationSnapshot(
            raw_codepoint_count=1,
            nfc_codepoint_count=1,
            normalized_codepoint_count=1,
            trailing_normalized_codepoints_removed=0,
            cross_boundary_output_positions=(),
            provenance_complete=True,
        ),
        combined_normalization=NormalizationSnapshot(
            raw_codepoint_count=3,
            nfc_codepoint_count=3,
            normalized_codepoint_count=3,
            trailing_normalized_codepoints_removed=0,
            cross_boundary_output_positions=(),
            provenance_complete=True,
        ),
        source_truncated=source,
        combined_full=combined,
        combined_truncated=combined,
        legacy_mask=LegacyMask(
            algorithm="standalone-source-token-count",
            cutoff=2,
            masked_positions=(0, 1),
            supervised_positions=(2, 3),
        ),
        full_attribution=attribution,
        truncated_attribution=attribution,
        diagnostics=Diagnostics(
            oracle_cutoff=2,
            prompt_leakage_positions=(),
            masked_retained_target_positions=(),
            cross_boundary_positions=(),
            ambiguous_positions=(),
            full_target_token_count=2,
            retained_target_token_count=2,
            supervised_target_token_count=2,
            truncated_target_token_count=0,
            elimination_causes=(),
            indeterminate_reasons=(),
        ),
    )


def _source_truncated_report() -> BoundaryReport:
    source_truncated = _encoding(
        (1, 3),
        ("<bos>", "a"),
        ((0, 0), (0, 1)),
    )
    combined_full = _encoding(
        (1, 3, 3, 3),
        ("<bos>", "a", "a", "a"),
        ((0, 0), (0, 1), (1, 2), (2, 3)),
    )
    full_attribution = (
        TokenAttribution(0, "injected_prefix", ()),
        TokenAttribution(1, "source", (0,)),
        TokenAttribution(2, "source", (1,)),
        TokenAttribution(3, "target", (2,)),
    )
    return BoundaryReport(
        status="fail",
        primary_classification="target_eliminated",
        classifications=("target_eliminated",),
        input_identity=InputIdentity(
            case_id="source-window-full",
            canonical_sha256="b" * 64,
            source_codepoints=2,
            target_codepoints=1,
            boundary_codepoint=2,
            max_length=2,
        ),
        artifact_identity=_artifact_identity(),
        source_normalization=NormalizationSnapshot(
            raw_codepoint_count=2,
            nfc_codepoint_count=2,
            normalized_codepoint_count=2,
            trailing_normalized_codepoints_removed=0,
            cross_boundary_output_positions=(),
            provenance_complete=True,
        ),
        combined_normalization=NormalizationSnapshot(
            raw_codepoint_count=3,
            nfc_codepoint_count=3,
            normalized_codepoint_count=3,
            trailing_normalized_codepoints_removed=0,
            cross_boundary_output_positions=(),
            provenance_complete=True,
        ),
        source_truncated=source_truncated,
        combined_full=combined_full,
        combined_truncated=source_truncated,
        legacy_mask=LegacyMask(
            algorithm="standalone-source-token-count",
            cutoff=2,
            masked_positions=(0, 1),
            supervised_positions=(),
        ),
        full_attribution=full_attribution,
        truncated_attribution=full_attribution[:2],
        diagnostics=Diagnostics(
            oracle_cutoff=None,
            prompt_leakage_positions=(),
            masked_retained_target_positions=(),
            cross_boundary_positions=(),
            ambiguous_positions=(),
            full_target_token_count=1,
            retained_target_token_count=0,
            supervised_target_token_count=0,
            truncated_target_token_count=1,
            elimination_causes=("truncation",),
            indeterminate_reasons=(),
        ),
    )


class TokenBoundaryReportSerializationTests(unittest.TestCase):
    def test_primitive_has_the_exact_nested_shape_and_scope(self) -> None:
        primitive = boundary_report_to_primitive(_aligned_report())

        self.assertEqual(
            set(primitive),
            {
                "schema_version",
                "kind",
                "status",
                "primary_classification",
                "classifications",
                "input_identity",
                "artifact_identity",
                "normalization",
                "encodings",
                "legacy_mask",
                "attribution",
                "diagnostics",
                "scope",
            },
        )
        self.assertEqual(primitive["schema_version"], SCHEMA_VERSION)
        self.assertEqual(primitive["kind"], REPORT_KIND)
        self.assertEqual(
            set(primitive["normalization"]),  # type: ignore[arg-type]
            {"source", "combined"},
        )
        self.assertEqual(
            set(primitive["encodings"]),  # type: ignore[arg-type]
            {"source_truncated", "combined_full", "combined_truncated"},
        )
        self.assertEqual(
            set(primitive["attribution"]),  # type: ignore[arg-type]
            {"full", "truncated"},
        )

        scope = primitive["scope"]
        self.assertIs(type(scope), dict)
        assert isinstance(scope, dict)
        self.assertEqual(
            set(scope),
            {
                *SCOPE_FALSE_FIELDS,
                "artifact_origin",
                "case_class",
            },
        )
        for field in SCOPE_FALSE_FIELDS:
            self.assertIs(scope[field], False)
        self.assertEqual(
            scope["artifact_origin"],
            "locally-authored-synthetic",
        )
        self.assertEqual(scope["case_class"], "caller-authored-synthetic")

    def test_canonical_bytes_are_sorted_compact_ascii_and_stable(self) -> None:
        report = _aligned_report()
        primitive = boundary_report_to_primitive(report)
        expected = (
            json.dumps(
                primitive,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")

        first = canonical_boundary_report_bytes(report)
        second = canonical_boundary_report_bytes(report)
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertTrue(first.isascii())
        self.assertTrue(first.endswith(b"\n"))
        self.assertNotIn(b": ", first)
        self.assertEqual(
            boundary_report_sha256(report),
            hashlib.sha256(first).hexdigest(),
        )

    def test_report_omits_raw_case_text_and_keeps_only_synthetic_token_facts(
        self,
    ) -> None:
        primitive = boundary_report_to_primitive(_aligned_report())
        identity = primitive["input_identity"]

        self.assertIs(type(identity), dict)
        assert isinstance(identity, dict)
        self.assertEqual(
            set(identity),
            {
                "case_id",
                "canonical_sha256",
                "source_codepoints",
                "target_codepoints",
                "boundary_codepoint",
                "max_length",
            },
        )
        document = canonical_boundary_report_bytes(_aligned_report())
        for forbidden in (
            b"/home/",
            b"/tmp/",
            b"objective",
            b"PRIVATE",
            b"finetune.py",
        ):
            self.assertNotIn(forbidden, document)

    def test_serializers_require_and_revalidate_the_exact_report_type(
        self,
    ) -> None:
        for value in (object(), None, {"status": "pass"}):
            with (
                self.subTest(type=type(value).__name__),
                self.assertRaisesRegex(
                    BoundaryReportError,
                    "boundary report rejected: report.type",
                ),
            ):
                boundary_report_to_primitive(value)  # type: ignore[arg-type]

        corrupted = _aligned_report()
        object.__setattr__(corrupted, "status", "fail")
        with self.assertRaisesRegex(
            BoundaryReportError,
            "boundary report rejected: report.status",
        ):
            canonical_boundary_report_bytes(corrupted)


class TokenBoundaryReportValueTests(unittest.TestCase):
    def test_all_values_are_frozen_slotted_and_nested_data_are_tuples(
        self,
    ) -> None:
        report = _aligned_report()

        self.assertFalse(hasattr(report, "__dict__"))
        self.assertFalse(hasattr(report.combined_full, "__dict__"))
        self.assertIs(type(report.classifications), tuple)
        self.assertIs(type(report.combined_full.ids), tuple)
        self.assertIs(type(report.full_attribution), tuple)
        with self.assertRaises(FrozenInstanceError):
            report.status = "fail"  # type: ignore[misc]

    def test_encoding_requires_exact_equal_length_vectors(self) -> None:
        valid = _aligned_report().combined_full
        invalid_changes: tuple[tuple[str, object, str], ...] = (
            ("ids", [1, 5, 6, 7], "encoding.ids"),
            ("pieces", ("<bos>", "c"), "encoding.length"),
            (
                "pieces",
                ("<bos>", "c", "PRIVATE-TARGET", "e"),
                "encoding.token_piece",
            ),
            (
                "offsets",
                ((0, 0), (0, 1), (2, 1), (2, 3)),
                "encoding.offsets",
            ),
            (
                "offsets",
                ((0, 0), (1, 2), (0, 1), (2, 3)),
                "encoding.offset_order",
            ),
            (
                "special_tokens_mask",
                (True, 0, 0, 0),
                "encoding.special_tokens_mask",
            ),
            (
                "attention_mask",
                (1, 1, 1, 2),
                "encoding.attention_mask",
            ),
            (
                "attention_mask",
                (1, 1, 1, 0),
                "encoding.attention_mask",
            ),
        )
        for field, value, code in invalid_changes:
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(
                    BoundaryReportError,
                    f"boundary report rejected: {code}",
                ),
            ):
                replace(valid, **{field: value})  # type: ignore[arg-type]

    def test_normalization_input_and_artifact_identities_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            BoundaryReportError,
            "normalization.counts",
        ):
            NormalizationSnapshot(3, 3, 1, 1, (), True)
        with self.assertRaisesRegex(
            BoundaryReportError,
            "normalization.cross_boundary_positions",
        ):
            NormalizationSnapshot(3, 3, 3, 0, (3,), True)
        with self.assertRaisesRegex(BoundaryReportError, "input.max_length"):
            replace(_aligned_report().input_identity, max_length=True)
        with self.assertRaisesRegex(
            BoundaryReportError,
            "input.codepoint_bounds",
        ):
            replace(
                _aligned_report().input_identity,
                target_codepoints=10**5000,
            )
        with self.assertRaisesRegex(
            BoundaryReportError,
            "input.canonical_sha256",
        ):
            replace(
                _aligned_report().input_identity,
                canonical_sha256="A" * 64,
            )
        with self.assertRaisesRegex(BoundaryReportError, "artifact.identity"):
            replace(_artifact_identity(), engine_version="0.21.5")

    def test_attribution_ownership_and_reason_codes_are_closed(self) -> None:
        with self.assertRaisesRegex(
            BoundaryReportError,
            "attribution.injected_origins",
        ):
            TokenAttribution(0, "injected_prefix", (0,))
        with self.assertRaisesRegex(
            BoundaryReportError,
            "attribution.raw_origins",
        ):
            TokenAttribution(1, "target", ())
        with self.assertRaisesRegex(
            BoundaryReportError,
            "attribution.ownership",
        ):
            TokenAttribution(1, "private-owner", (0,))
        with self.assertRaisesRegex(
            BoundaryReportError,
            "diagnostics.indeterminate_reasons",
        ) as caught:
            replace(
                _aligned_report().diagnostics,
                indeterminate_reasons=("PRIVATE-CONTENT",),
            )
        self.assertNotIn("PRIVATE-CONTENT", str(caught.exception))
        with self.assertRaisesRegex(
            BoundaryReportError,
            "diagnostics.indeterminate_reasons",
        ):
            replace(
                _aligned_report().diagnostics,
                indeterminate_reasons=("arbitrary.reason",),
            )

    def test_position_tuples_are_strictly_increasing_and_nonnegative(self) -> None:
        for positions in ((1, 1), (2, 1), (-1,)):
            with (
                self.subTest(positions=positions),
                self.assertRaisesRegex(
                    BoundaryReportError,
                    "diagnostics.cross_boundary_positions",
                ),
            ):
                replace(
                    _aligned_report().diagnostics,
                    cross_boundary_positions=positions,
                )


class TokenBoundaryReportCrossFieldTests(unittest.TestCase):
    def test_status_primary_and_classifications_are_fully_derived(self) -> None:
        report = _aligned_report()
        invalid_changes: tuple[tuple[str, object, str], ...] = (
            ("status", "fail", "report.status"),
            (
                "primary_classification",
                "boundary_drift",
                "report.primary_classification",
            ),
            (
                "classifications",
                ("aligned", "boundary_drift"),
                "report.classifications",
            ),
            (
                "classifications",
                ("boundary_drift",),
                "report.primary_classification",
            ),
        )
        for field, value, code in invalid_changes:
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(
                    BoundaryReportError,
                    f"boundary report rejected: {code}",
                ),
            ):
                replace(report, **{field: value})  # type: ignore[arg-type]

        indeterminate_diagnostics = replace(
            report.diagnostics,
            indeterminate_reasons=("normalization.runtime_mismatch",),
        )
        indeterminate = replace(
            report,
            status="indeterminate",
            primary_classification="indeterminate",
            classifications=("indeterminate",),
            combined_normalization=replace(
                report.combined_normalization,
                provenance_complete=False,
            ),
            diagnostics=indeterminate_diagnostics,
        )
        self.assertEqual(
            boundary_report_to_primitive(indeterminate)["status"],
            "indeterminate",
        )

    def test_indeterminate_reasons_require_observable_evidence(self) -> None:
        report = _aligned_report()
        unsupported_reasons = (
            "encoding.offset_provenance",
            "encoding.piece_alignment",
            "normalization.cluster_replay",
            "normalization.provenance_ambiguous",
            "normalization.runtime_mismatch",
        )

        for reason in unsupported_reasons:
            with (
                self.subTest(reason=reason),
                self.assertRaises(BoundaryReportError),
            ):
                replace(
                    report,
                    status="indeterminate",
                    primary_classification="indeterminate",
                    classifications=("indeterminate",),
                    diagnostics=replace(
                        report.diagnostics,
                        indeterminate_reasons=(reason,),
                    ),
                )

    def test_earlier_ambiguity_suppresses_numeric_cutoff_drift(self) -> None:
        report = _aligned_report()
        corrupted_encoding = replace(
            report.combined_full,
            offsets=((0, 0), (0, 1), (0, 1), (2, 3)),
        )
        ambiguous_target = replace(
            report.full_attribution[2],
            ownership="ambiguous",
        )
        attribution = (
            report.full_attribution[:2]
            + (ambiguous_target,)
            + report.full_attribution[3:]
        )
        diagnostics = replace(
            report.diagnostics,
            oracle_cutoff=None,
            ambiguous_positions=(2,),
            full_target_token_count=1,
            retained_target_token_count=1,
            supervised_target_token_count=1,
            indeterminate_reasons=("encoding.offset_provenance",),
        )
        accepted = replace(
            report,
            status="indeterminate",
            primary_classification="indeterminate",
            classifications=("indeterminate",),
            combined_full=corrupted_encoding,
            combined_truncated=corrupted_encoding,
            full_attribution=attribution,
            truncated_attribution=attribution,
            diagnostics=diagnostics,
        )

        self.assertIsNone(accepted.diagnostics.oracle_cutoff)
        self.assertNotIn("boundary_drift", accepted.classifications)
        with self.assertRaisesRegex(
            BoundaryReportError,
            "report.oracle_cutoff",
        ):
            replace(
                accepted,
                diagnostics=replace(
                    accepted.diagnostics,
                    oracle_cutoff=3,
                ),
            )

        truncated_encoding = replace(
            corrupted_encoding,
            ids=corrupted_encoding.ids[:3],
            pieces=corrupted_encoding.pieces[:3],
            offsets=corrupted_encoding.offsets[:3],
            type_ids=corrupted_encoding.type_ids[:3],
            special_tokens_mask=corrupted_encoding.special_tokens_mask[:3],
            attention_mask=corrupted_encoding.attention_mask[:3],
        )
        uncertain_elimination = replace(
            report,
            status="indeterminate",
            primary_classification="indeterminate",
            classifications=("indeterminate",),
            input_identity=replace(
                report.input_identity,
                max_length=3,
            ),
            combined_full=corrupted_encoding,
            combined_truncated=truncated_encoding,
            legacy_mask=replace(
                report.legacy_mask,
                supervised_positions=(2,),
            ),
            full_attribution=attribution,
            truncated_attribution=attribution[:3],
            diagnostics=replace(
                diagnostics,
                retained_target_token_count=0,
                supervised_target_token_count=0,
                truncated_target_token_count=1,
            ),
        )
        self.assertEqual(uncertain_elimination.classifications, ("indeterminate",))
        self.assertEqual(uncertain_elimination.diagnostics.elimination_causes, ())

    def test_source_window_truncation_is_not_piece_misalignment(self) -> None:
        report = _source_truncated_report()

        self.assertEqual(report.status, "fail")
        self.assertEqual(report.classifications, ("target_eliminated",))
        self.assertEqual(report.diagnostics.indeterminate_reasons, ())
        with self.assertRaisesRegex(
            BoundaryReportError,
            "report.piece_alignment_reason",
        ):
            replace(
                report,
                input_identity=replace(
                    report.input_identity,
                    max_length=3,
                ),
            )

    def test_legacy_mask_is_the_exact_cutoff_partition(self) -> None:
        report = _aligned_report()
        alternative_mask = LegacyMask(
            algorithm="standalone-source-token-count",
            cutoff=1,
            masked_positions=(0,),
            supervised_positions=(1, 2, 3),
        )
        with self.assertRaisesRegex(
            BoundaryReportError,
            "report.legacy_cutoff",
        ):
            replace(report, legacy_mask=alternative_mask)

        incomplete_partition = LegacyMask(
            algorithm="standalone-source-token-count",
            cutoff=2,
            masked_positions=(0,),
            supervised_positions=(2, 3),
        )
        with self.assertRaisesRegex(
            BoundaryReportError,
            "report.masked_positions",
        ):
            replace(report, legacy_mask=incomplete_partition)

    def test_attribution_positions_ownership_and_counts_are_recomputed(self) -> None:
        report = _aligned_report()
        ambiguous_prefix = replace(
            report.full_attribution[0],
            ownership="ambiguous",
        )
        with self.assertRaisesRegex(
            BoundaryReportError,
            "report.full_attribution.special",
        ):
            replace(
                report,
                full_attribution=(ambiguous_prefix,) + report.full_attribution[1:],
            )

        wrong_position = replace(report.full_attribution[2], position=3)
        with self.assertRaisesRegex(
            BoundaryReportError,
            "report.full_attribution.positions",
        ):
            replace(
                report,
                full_attribution=(
                    report.full_attribution[:2]
                    + (wrong_position,)
                    + report.full_attribution[3:]
                ),
            )

        wrong_origin = replace(report.full_attribution[2], raw_origins=(0,))
        with self.assertRaisesRegex(
            BoundaryReportError,
            "report.full_attribution.ownership",
        ):
            replace(
                report,
                full_attribution=(
                    report.full_attribution[:2]
                    + (wrong_origin,)
                    + report.full_attribution[3:]
                ),
            )

        overlapping_ambiguous = replace(
            report.full_attribution[2],
            ownership="ambiguous",
            raw_origins=(0,),
        )
        with self.assertRaisesRegex(
            BoundaryReportError,
            "report.full_attribution.origin_overlap",
        ):
            replace(
                report,
                full_attribution=(
                    report.full_attribution[:2]
                    + (overlapping_ambiguous,)
                    + report.full_attribution[3:]
                ),
            )

        reordered_origins = (
            report.full_attribution[0],
            TokenAttribution(1, "cross_boundary", (0, 2)),
            TokenAttribution(2, "target", (1,)),
            TokenAttribution(3, "ambiguous", ()),
        )
        with self.assertRaisesRegex(
            BoundaryReportError,
            "report.full_attribution.origin_order",
        ):
            replace(
                report,
                full_attribution=reordered_origins,
            )

        unjustified_ambiguous = replace(
            report.full_attribution[2],
            ownership="ambiguous",
        )
        with self.assertRaisesRegex(
            BoundaryReportError,
            "report.full_attribution.ownership",
        ):
            replace(
                report,
                full_attribution=(
                    report.full_attribution[:2]
                    + (unjustified_ambiguous,)
                    + report.full_attribution[3:]
                ),
            )

        out_of_range_offset = replace(
            report.combined_full,
            offsets=((0, 0), (0, 1), (1, 2), (9, 10)),
        )
        with self.assertRaisesRegex(
            BoundaryReportError,
            "report.full_attribution.offset",
        ):
            replace(report, combined_full=out_of_range_offset)

        with self.assertRaisesRegex(
            BoundaryReportError,
            "report.full_target_count",
        ):
            replace(
                report,
                diagnostics=replace(
                    report.diagnostics,
                    full_target_token_count=3,
                ),
            )
        with self.assertRaisesRegex(
            BoundaryReportError,
            "report.truncated_target_count",
        ):
            replace(
                report,
                diagnostics=replace(
                    report.diagnostics,
                    truncated_target_token_count=1,
                ),
            )

    def test_nonprefix_encoding_is_representable_only_as_indeterminate(self) -> None:
        report = _aligned_report()
        nonprefix = replace(
            report.combined_truncated,
            ids=(1, 5, 6, 8),
            pieces=("<bos>", "c", "d", "f"),
        )
        with self.assertRaisesRegex(
            BoundaryReportError,
            "report.encoding_prefix",
        ):
            replace(report, combined_truncated=nonprefix)

        nonprefix_diagnostics = replace(
            report.diagnostics,
            oracle_cutoff=None,
            retained_target_token_count=0,
            supervised_target_token_count=0,
            truncated_target_token_count=2,
            indeterminate_reasons=("truncation.not_exact_prefix",),
        )
        with self.assertRaisesRegex(
            BoundaryReportError,
            "report.encoding_prefix_reason",
        ):
            replace(
                report,
                status="indeterminate",
                primary_classification="indeterminate",
                classifications=("indeterminate",),
                diagnostics=nonprefix_diagnostics,
            )
        ambiguous_truncated = (
            report.truncated_attribution[0],
            TokenAttribution(1, "ambiguous", ()),
            TokenAttribution(2, "ambiguous", ()),
            TokenAttribution(3, "ambiguous", ()),
        )
        accepted = replace(
            report,
            status="indeterminate",
            primary_classification="indeterminate",
            classifications=("indeterminate",),
            combined_truncated=nonprefix,
            truncated_attribution=ambiguous_truncated,
            diagnostics=nonprefix_diagnostics,
        )
        self.assertEqual(accepted.classifications, ("indeterminate",))
        self.assertEqual(accepted.diagnostics.elimination_causes, ())

    def test_raw_counts_bounds_and_source_cross_positions_are_cross_checked(
        self,
    ) -> None:
        report = _aligned_report()
        with self.assertRaisesRegex(
            BoundaryReportError,
            "report.normalization_counts",
        ):
            replace(
                report,
                combined_normalization=replace(
                    report.combined_normalization,
                    raw_codepoint_count=4,
                ),
            )
        with self.assertRaisesRegex(
            BoundaryReportError,
            "report.source_cross_boundary",
        ):
            replace(
                report,
                source_normalization=replace(
                    report.source_normalization,
                    cross_boundary_output_positions=(0,),
                ),
            )
        with self.assertRaisesRegex(
            BoundaryReportError,
            "report.normalization_cross_boundary",
        ):
            replace(
                report,
                combined_normalization=replace(
                    report.combined_normalization,
                    cross_boundary_output_positions=(0,),
                ),
            )

        too_long = _encoding(
            (1, 5, 5, 5, 5, 5, 5, 5, 5),
            ("<bos>", "c", "c", "c", "c", "c", "c", "c", "c"),
            (
                (0, 0),
                (0, 1),
                (0, 1),
                (0, 1),
                (0, 1),
                (0, 1),
                (0, 1),
                (0, 1),
                (0, 1),
            ),
        )
        with self.assertRaisesRegex(
            BoundaryReportError,
            "report.truncation_length",
        ):
            replace(report, source_truncated=too_long)

    def test_classification_precedence_is_public_and_exact(self) -> None:
        self.assertEqual(
            CLASSIFICATION_PRECEDENCE,
            (
                "indeterminate",
                "cross_boundary_token",
                "target_eliminated",
                "partial_target_truncation",
                "boundary_drift",
                "aligned",
            ),
        )

    def test_report_module_imports_only_the_standard_library(self) -> None:
        path = REPOSITORY_ROOT / "token_boundary/report.py"
        tree = ast.parse(path.read_text("utf-8"), filename=str(path))
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.partition(".")[0] for alias in node.names
                )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module is not None
            ):
                imported_roots.add(node.module.partition(".")[0])

        self.assertLessEqual(imported_roots, sys.stdlib_module_names)
        self.assertNotIn("finetune", imported_roots)
        self.assertNotIn("tokenizers", imported_roots)


if __name__ == "__main__":
    unittest.main()
