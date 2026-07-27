from __future__ import annotations

import importlib
import importlib.util
import json
import unittest
from collections import Counter
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import ModuleType
from typing import ClassVar, cast
from unittest import mock

from token_boundary import analyze_boundary as public_analyze_boundary
from token_boundary.contract import BoundaryCase, parse_boundary_case
from token_boundary.errors import BoundaryEngineError
from token_boundary.report import (
    BoundaryReport,
    EncodingSnapshot,
    boundary_report_sha256,
    boundary_report_to_primitive,
    canonical_boundary_report_bytes,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPOSITORY_ROOT / "fixtures/token-boundary"
TOKENIZERS_AVAILABLE = importlib.util.find_spec("tokenizers") is not None
AnalyzeFunction = Callable[[BoundaryCase], BoundaryReport]

EXPECTED = {
    "aligned": {
        "status": "pass",
        "classifications": ("aligned",),
        "source_ids": (1, 5),
        "full_ids": (1, 5, 6, 7),
        "truncated_ids": (1, 5, 6, 7),
        "ownerships": ("injected_prefix", "source", "target", "target"),
        "cutoff": 2,
        "oracle": 2,
        "prompt_leakage": (),
        "masked_target": (),
        "cross": (),
        "counts": (2, 2, 2, 0),
        "causes": (),
        "reasons": (),
        "sha256": "49f851a6d89d25d5aee9d8b68548883dfbc2962a5610e9b55c2dfde176ac6d76",
    },
    "merge-cross-boundary": {
        "status": "fail",
        "classifications": ("cross_boundary_token",),
        "source_ids": (1, 3),
        "full_ids": (1, 14),
        "truncated_ids": (1, 14),
        "ownerships": ("injected_prefix", "cross_boundary"),
        "cutoff": 2,
        "oracle": None,
        "prompt_leakage": (),
        "masked_target": (),
        "cross": (1,),
        "counts": (0, 0, 0, 0),
        "causes": (),
        "reasons": (),
        "sha256": "b4c7364274c4c0f43c70db7b8b3e852cc4d4fbefeefd0d77ff60d7eef460721b",
    },
    "nfc-cross-boundary": {
        "status": "fail",
        "classifications": ("cross_boundary_token",),
        "source_ids": (1, 7),
        "full_ids": (1, 13, 12),
        "truncated_ids": (1, 13, 12),
        "ownerships": ("injected_prefix", "cross_boundary", "target"),
        "cutoff": 2,
        "oracle": 2,
        "prompt_leakage": (),
        "masked_target": (),
        "cross": (1,),
        "counts": (1, 1, 1, 0),
        "causes": (),
        "reasons": (),
        "sha256": "1aeec08efe815ca11b599b1b9427c94a951aebfa2c004e3733fa06b3a662344a",
    },
    "normalized-away": {
        "status": "indeterminate",
        "classifications": ("indeterminate",),
        "source_ids": (1, 5),
        "full_ids": (1, 5),
        "truncated_ids": (1, 5),
        "ownerships": ("injected_prefix", "source"),
        "cutoff": 2,
        "oracle": None,
        "prompt_leakage": (),
        "masked_target": (),
        "cross": (),
        "counts": (0, 0, 0, 0),
        "causes": (),
        "reasons": ("target.no_attributable_token",),
        "sha256": "ef5b1e1ced0609f09c5bf4a8a05c31406d0e0fe0a662287583eaed57bc5bf475",
    },
    "partial-truncation": {
        "status": "fail",
        "classifications": ("partial_target_truncation",),
        "source_ids": (1, 5, 6, 7),
        "full_ids": (1, 5, 6, 7, 8, 9, 10, 11),
        "truncated_ids": (1, 5, 6, 7, 8, 9),
        "ownerships": (
            "injected_prefix",
            "source",
            "source",
            "source",
            "target",
            "target",
            "target",
            "target",
        ),
        "cutoff": 4,
        "oracle": 4,
        "prompt_leakage": (),
        "masked_target": (),
        "cross": (),
        "counts": (4, 2, 2, 2),
        "causes": (),
        "reasons": (),
        "sha256": "35186327542ccaeb3d3f6ce1f6d1ff85b25c23bc02a36ace71cc9ab0cfcc0179",
    },
    "right-strip-drift": {
        "status": "fail",
        "classifications": ("boundary_drift",),
        "source_ids": (1, 5),
        "full_ids": (1, 5, 2, 6),
        "truncated_ids": (1, 5, 2, 6),
        "ownerships": ("injected_prefix", "source", "source", "target"),
        "cutoff": 2,
        "oracle": 3,
        "prompt_leakage": (2,),
        "masked_target": (),
        "cross": (),
        "counts": (1, 1, 1, 0),
        "causes": (),
        "reasons": (),
        "sha256": "ffe7b5194512f64c5d998acef0f3d7d026f29cb4f3fef26b28ca9b1baee95e01",
    },
    "target-eliminated": {
        "status": "fail",
        "classifications": ("target_eliminated",),
        "source_ids": (1, 5, 6, 7, 8),
        "full_ids": (1, 5, 6, 7, 8, 9),
        "truncated_ids": (1, 5, 6, 7, 8),
        "ownerships": (
            "injected_prefix",
            "source",
            "source",
            "source",
            "source",
            "target",
        ),
        "cutoff": 5,
        "oracle": None,
        "prompt_leakage": (),
        "masked_target": (),
        "cross": (),
        "counts": (1, 0, 0, 1),
        "causes": ("truncation",),
        "reasons": (),
        "sha256": "d5c2121a22967b0a187882d978a66a919d653f8997649402e9e3e1246a73e48f",
    },
}


def _analyze(case: BoundaryCase) -> BoundaryReport:
    module = importlib.import_module("token_boundary.engine")
    implementation = cast(AnalyzeFunction, module.analyze_boundary)
    return implementation(case)


@unittest.skipUnless(
    TOKENIZERS_AVAILABLE,
    "optional tokenizers==0.21.4 runtime is not installed",
)
class TokenBoundaryEngineTests(unittest.TestCase):
    engine: ClassVar[ModuleType]
    cases: ClassVar[dict[str, BoundaryCase]]
    reports: ClassVar[dict[str, BoundaryReport]]

    @classmethod
    def setUpClass(cls) -> None:
        module = importlib.import_module("token_boundary.engine")
        cls.engine = module
        cls.cases = {}
        cls.reports = {}
        for path in sorted(FIXTURE_ROOT.glob("*.json")):
            case = parse_boundary_case(path.read_bytes())
            cls.cases[case.case_id] = case
            cls.reports[case.case_id] = _analyze(case)

    def test_all_seven_cases_have_exact_runtime_observations(self) -> None:
        self.assertEqual(set(self.reports), set(EXPECTED))
        for case_id, expected in EXPECTED.items():
            with self.subTest(case_id=case_id):
                report = self.reports[case_id]
                diagnostics = report.diagnostics
                self.assertEqual(report.status, expected["status"])
                self.assertEqual(
                    report.classifications,
                    expected["classifications"],
                )
                self.assertEqual(
                    report.primary_classification,
                    report.classifications[0],
                )
                self.assertEqual(
                    report.source_truncated.ids,
                    expected["source_ids"],
                )
                self.assertEqual(report.combined_full.ids, expected["full_ids"])
                self.assertEqual(
                    report.combined_truncated.ids,
                    expected["truncated_ids"],
                )
                self.assertEqual(
                    tuple(item.ownership for item in report.full_attribution),
                    expected["ownerships"],
                )
                self.assertEqual(report.legacy_mask.cutoff, expected["cutoff"])
                self.assertEqual(diagnostics.oracle_cutoff, expected["oracle"])
                self.assertEqual(
                    diagnostics.prompt_leakage_positions,
                    expected["prompt_leakage"],
                )
                self.assertEqual(
                    diagnostics.masked_retained_target_positions,
                    expected["masked_target"],
                )
                self.assertEqual(
                    diagnostics.cross_boundary_positions,
                    expected["cross"],
                )
                self.assertEqual(
                    (
                        diagnostics.full_target_token_count,
                        diagnostics.retained_target_token_count,
                        diagnostics.supervised_target_token_count,
                        diagnostics.truncated_target_token_count,
                    ),
                    expected["counts"],
                )
                self.assertEqual(
                    diagnostics.elimination_causes,
                    expected["causes"],
                )
                self.assertEqual(
                    diagnostics.indeterminate_reasons,
                    expected["reasons"],
                )

    def test_valid_input_matrix_is_total_and_deterministic(self) -> None:
        texts = (
            "a",
            "b",
            "ab",
            "c ",
            "e",
            "e\u0301",
            "é",
            " ",
            "z",
            "\u1100",
            "\u1161",
        )
        observed: Counter[tuple[str, ...]] = Counter()

        for source in texts:
            for target in texts:
                for max_length in (2, 3, 4, 8):
                    report = _analyze(
                        BoundaryCase(
                            case_id="input-matrix",
                            source=source,
                            target=target,
                            max_length=max_length,
                        )
                    )
                    observed[report.classifications] += 1

        self.assertEqual(
            observed,
            Counter(
                {
                    ("aligned",): 123,
                    ("boundary_drift",): 35,
                    ("cross_boundary_token",): 4,
                    ("indeterminate",): 239,
                    ("indeterminate", "boundary_drift"): 21,
                    ("target_eliminated",): 48,
                    ("target_eliminated", "boundary_drift"): 14,
                }
            ),
        )

    def test_each_complete_report_has_a_frozen_canonical_hash(self) -> None:
        for case_id, expected in EXPECTED.items():
            with self.subTest(case_id=case_id):
                report = self.reports[case_id]
                canonical = canonical_boundary_report_bytes(report)
                self.assertTrue(canonical.endswith(b"\n"))
                self.assertEqual(canonical.count(b"\n"), 1)
                self.assertEqual(
                    boundary_report_sha256(report),
                    expected["sha256"],
                )
                self.assertEqual(
                    json.loads(canonical),
                    boundary_report_to_primitive(report),
                )
                self.assertEqual(_analyze(self.cases[case_id]), report)

    def test_nfc_oracle_recovers_origin_hidden_by_runtime_offset(self) -> None:
        report = self.reports["nfc-cross-boundary"]

        self.assertEqual(report.combined_full.pieces, ("<bos>", "é", "x"))
        self.assertEqual(report.combined_full.offsets, ((0, 0), (0, 1), (2, 3)))
        self.assertEqual(report.full_attribution[1].raw_origins, (0, 1))
        self.assertEqual(
            report.combined_normalization.cross_boundary_output_positions,
            (0,),
        )
        self.assertEqual(report.diagnostics.cross_boundary_positions, (1,))

    def test_bpe_merge_is_cross_boundary_without_inventing_target_loss(self) -> None:
        report = self.reports["merge-cross-boundary"]

        self.assertEqual(report.combined_full.pieces, ("<bos>", "ab"))
        self.assertEqual(report.full_attribution[1].raw_origins, (0, 1))
        self.assertEqual(report.diagnostics.full_target_token_count, 0)
        self.assertNotIn("target_eliminated", report.classifications)

    def test_right_strip_exposes_the_exact_source_owned_leak(self) -> None:
        report = self.reports["right-strip-drift"]

        self.assertEqual(
            report.source_normalization.trailing_normalized_codepoints_removed,
            1,
        )
        self.assertEqual(
            report.combined_normalization.trailing_normalized_codepoints_removed,
            0,
        )
        self.assertEqual(report.combined_full.pieces, ("<bos>", "c", " ", "d"))
        self.assertEqual(report.legacy_mask.supervised_positions, (2, 3))
        self.assertEqual(report.diagnostics.prompt_leakage_positions, (2,))

    def test_truncation_diagnostics_distinguish_partial_and_total_loss(
        self,
    ) -> None:
        partial = self.reports["partial-truncation"]
        eliminated = self.reports["target-eliminated"]

        self.assertEqual(
            partial.combined_truncated,
            replace(
                partial.combined_full,
                ids=partial.combined_full.ids[:6],
                pieces=partial.combined_full.pieces[:6],
                offsets=partial.combined_full.offsets[:6],
                type_ids=partial.combined_full.type_ids[:6],
                special_tokens_mask=partial.combined_full.special_tokens_mask[:6],
                attention_mask=partial.combined_full.attention_mask[:6],
            ),
        )
        self.assertEqual(
            partial.diagnostics.truncated_target_token_count,
            2,
        )
        self.assertEqual(eliminated.legacy_mask.supervised_positions, ())
        self.assertEqual(
            eliminated.diagnostics.elimination_causes,
            ("truncation",),
        )

    def test_reports_omit_raw_input_fields_and_make_scope_false(self) -> None:
        for report in self.reports.values():
            primitive = boundary_report_to_primitive(report)
            input_identity = cast(
                dict[str, object],
                primitive["input_identity"],
            )
            self.assertEqual(
                set(input_identity),
                {
                    "boundary_codepoint",
                    "canonical_sha256",
                    "case_id",
                    "max_length",
                    "source_codepoints",
                    "target_codepoints",
                },
            )
            self.assertNotIn("source", primitive)
            self.assertNotIn("target", primitive)
            scope = cast(dict[str, object], primitive["scope"])
            for key, value in scope.items():
                if key not in {"artifact_origin", "case_class"}:
                    self.assertIs(value, False)

    def test_report_values_are_immutable(self) -> None:
        report = self.reports["aligned"]

        self.assertFalse(hasattr(report, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            report.status = "fail"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            report.combined_full.ids[0] = 99  # type: ignore[index]

    def test_each_encoding_uses_a_fresh_tokenizer_instance(self) -> None:
        original = self.engine._new_tokenizer
        with mock.patch.object(
            self.engine,
            "_new_tokenizer",
            wraps=original,
        ) as constructor:
            report = _analyze(self.cases["aligned"])

        self.assertEqual(report.status, "pass")
        self.assertEqual(constructor.call_count, 3)

    def test_public_api_loads_the_optional_engine_lazily(self) -> None:
        report = public_analyze_boundary(self.cases["aligned"])

        self.assertEqual(report, self.reports["aligned"])
        self.assertEqual(report.status, "pass")

    def test_runtime_and_artifact_failures_are_stable(self) -> None:
        with (
            mock.patch.object(
                self.engine.metadata,
                "version",
                return_value="0.0.0",
            ),
            self.assertRaises(BoundaryEngineError) as version_error,
        ):
            _analyze(self.cases["aligned"])
        self.assertEqual(version_error.exception.code, "runtime.version")
        self.assertIsNone(version_error.exception.__cause__)
        self.assertIsNone(version_error.exception.__context__)

        artifact_error_type = self.engine.ArtifactVerificationError
        with (
            mock.patch.object(
                self.engine,
                "verify_local_artifact",
                side_effect=artifact_error_type("artifact.identity"),
            ),
            self.assertRaises(BoundaryEngineError) as artifact_error,
        ):
            _analyze(self.cases["aligned"])
        self.assertEqual(
            artifact_error.exception.code,
            "artifact.identity",
        )
        self.assertIsNone(artifact_error.exception.__cause__)
        self.assertIsNone(artifact_error.exception.__context__)

        invalid_encoding = mock.Mock(
            ids=[0],
            tokens=["<unk>"],
            offsets=[(0, 1)],
            type_ids=[0],
            special_tokens_mask=[0],
            attention_mask=[1],
        )
        with self.assertRaises(BoundaryEngineError) as encoding_error:
            self.engine._encoding_snapshot(invalid_encoding)
        self.assertEqual(encoding_error.exception.code, "runtime.encoding")
        self.assertIsNone(encoding_error.exception.__cause__)
        self.assertIsNone(encoding_error.exception.__context__)

        malformed_offsets = mock.Mock(
            ids=[1],
            tokens=["<bos>"],
            offsets=[(0,)],
            type_ids=[0],
            special_tokens_mask=[1],
            attention_mask=[1],
        )
        with self.assertRaises(BoundaryEngineError) as malformed_error:
            self.engine._encoding_snapshot(malformed_offsets)
        self.assertEqual(malformed_error.exception.code, "runtime.encoding")
        self.assertIsNone(malformed_error.exception.__cause__)
        self.assertIsNone(malformed_error.exception.__context__)

    def test_corrupted_offsets_become_indeterminate_not_guessed(self) -> None:
        case = self.cases["aligned"]
        artifact = self.engine._verified_artifact()
        source = self.engine._capture_encoding(
            artifact,
            case.source,
            case.max_length,
        )
        full = self.engine._capture_encoding(
            artifact,
            case.source + case.target,
            None,
        )
        bad_offsets = (*full.snapshot.offsets[:2], (0, 1), full.snapshot.offsets[3])
        corrupted_snapshot = replace(full.snapshot, offsets=bad_offsets)
        corrupted = replace(full, snapshot=corrupted_snapshot)

        with mock.patch.object(
            self.engine,
            "_capture_encoding",
            side_effect=(source, corrupted, corrupted),
        ):
            report = _analyze(case)

        self.assertEqual(report.status, "indeterminate")
        self.assertEqual(report.primary_classification, "indeterminate")
        self.assertEqual(report.classifications, ("indeterminate",))
        self.assertIsNone(report.diagnostics.oracle_cutoff)
        self.assertEqual(report.diagnostics.ambiguous_positions, (2,))
        self.assertIn(
            "encoding.offset_provenance",
            report.diagnostics.indeterminate_reasons,
        )

    def test_supervised_ambiguity_does_not_invent_total_elimination(self) -> None:
        case = BoundaryCase(
            case_id="ambiguous-supervision",
            source="c",
            target="de",
            max_length=3,
        )
        artifact = self.engine._verified_artifact()
        source = self.engine._capture_encoding(
            artifact,
            case.source,
            case.max_length,
        )
        full = self.engine._capture_encoding(
            artifact,
            case.source + case.target,
            None,
        )
        truncated = self.engine._capture_encoding(
            artifact,
            case.source + case.target,
            case.max_length,
        )
        bad_full_offsets = (
            *full.snapshot.offsets[:2],
            (0, 1),
            *full.snapshot.offsets[3:],
        )
        bad_truncated_offsets = (*truncated.snapshot.offsets[:2], (0, 1))
        corrupted_full = replace(
            full,
            snapshot=replace(
                full.snapshot,
                offsets=bad_full_offsets,
            ),
        )
        corrupted_truncated = replace(
            truncated,
            snapshot=replace(
                truncated.snapshot,
                offsets=bad_truncated_offsets,
            ),
        )

        with mock.patch.object(
            self.engine,
            "_capture_encoding",
            side_effect=(source, corrupted_full, corrupted_truncated),
        ):
            report = _analyze(case)

        self.assertEqual(report.classifications, ("indeterminate",))
        self.assertEqual(report.diagnostics.ambiguous_positions, (2,))
        self.assertEqual(report.diagnostics.elimination_causes, ())
        self.assertEqual(
            report.diagnostics.indeterminate_reasons,
            ("encoding.offset_provenance",),
        )

    def test_multi_label_failures_keep_public_precedence(self) -> None:
        cases = (
            (
                BoundaryCase(
                    case_id="eliminated-and-drifted",
                    source="c ",
                    target="d",
                    max_length=3,
                ),
                ("target_eliminated", "boundary_drift"),
                ("truncation",),
            ),
            (
                BoundaryCase(
                    case_id="partial-and-drifted",
                    source="c ",
                    target="de",
                    max_length=4,
                ),
                ("partial_target_truncation", "boundary_drift"),
                (),
            ),
        )

        for case, classifications, elimination_causes in cases:
            with self.subTest(case_id=case.case_id):
                report = _analyze(case)
                self.assertEqual(report.status, "fail")
                self.assertEqual(report.classifications, classifications)
                self.assertEqual(
                    report.primary_classification,
                    classifications[0],
                )
                self.assertEqual(
                    report.diagnostics.elimination_causes,
                    elimination_causes,
                )

    def test_full_source_window_is_truncation_not_alignment_failure(self) -> None:
        report = _analyze(
            BoundaryCase(
                case_id="source-window-full",
                source="aa",
                target="a",
                max_length=2,
            )
        )

        self.assertEqual(report.source_truncated.pieces, ("<bos>", "a"))
        self.assertEqual(report.status, "fail")
        self.assertEqual(report.classifications, ("target_eliminated",))
        self.assertEqual(report.diagnostics.indeterminate_reasons, ())
        self.assertEqual(
            report.diagnostics.elimination_causes,
            ("truncation",),
        )

    def test_unsaturated_source_capture_fails_indeterminate(self) -> None:
        case = self.cases["aligned"]
        artifact = self.engine._verified_artifact()
        source = self.engine._capture_encoding(
            artifact,
            case.source,
            case.max_length,
        )
        full = self.engine._capture_encoding(
            artifact,
            case.source + case.target,
            None,
        )
        premature_source = replace(
            source,
            snapshot=EncodingSnapshot(
                ids=(1,),
                pieces=("<bos>",),
                offsets=((0, 0),),
                type_ids=(0,),
                special_tokens_mask=(1,),
                attention_mask=(1,),
            ),
        )

        with mock.patch.object(
            self.engine,
            "_capture_encoding",
            side_effect=(premature_source, full, full),
        ):
            report = _analyze(case)

        self.assertEqual(
            report.classifications,
            ("indeterminate", "boundary_drift"),
        )
        self.assertEqual(
            report.diagnostics.indeterminate_reasons,
            ("encoding.piece_alignment",),
        )

    def test_nonprefix_runtime_anomaly_keeps_retained_ownership_unknown(
        self,
    ) -> None:
        case = self.cases["aligned"]
        artifact = self.engine._verified_artifact()
        source = self.engine._capture_encoding(
            artifact,
            case.source,
            case.max_length,
        )
        full = self.engine._capture_encoding(
            artifact,
            case.source + case.target,
            None,
        )
        anomalous_snapshot = EncodingSnapshot(
            ids=(1, 5, 6, 6),
            pieces=("<bos>", "c", "d", "d"),
            offsets=((0, 0), (0, 1), (1, 2), (2, 3)),
            type_ids=(0, 0, 0, 0),
            special_tokens_mask=(1, 0, 0, 0),
            attention_mask=(1, 1, 1, 1),
        )
        anomalous = replace(full, snapshot=anomalous_snapshot)

        with mock.patch.object(
            self.engine,
            "_capture_encoding",
            side_effect=(source, full, anomalous),
        ):
            report = _analyze(case)

        self.assertEqual(report.classifications, ("indeterminate",))
        self.assertEqual(
            report.diagnostics.indeterminate_reasons,
            ("truncation.not_exact_prefix",),
        )
        self.assertEqual(
            tuple(item.ownership for item in report.truncated_attribution),
            ("injected_prefix", "ambiguous", "ambiguous", "ambiguous"),
        )
        self.assertEqual(report.diagnostics.elimination_causes, ())

    def test_exact_case_type_is_required(self) -> None:
        with self.assertRaises(BoundaryEngineError) as caught:
            _analyze(object())  # type: ignore[arg-type]

        self.assertEqual(caught.exception.code, "case.type")


if __name__ == "__main__":
    unittest.main()
