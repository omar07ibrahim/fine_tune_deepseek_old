from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import unittest

from loss_topology import (
    ALL_TOKENS,
    ASSISTANT_ONLY,
    IGNORE_INDEX,
    TopologyError,
    audit_label_topology,
    audit_sha256,
    audit_trace,
    build_labels,
    canonical_audit_bytes,
    parse_synthetic_trace,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class LossTopologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.healthy = parse_synthetic_trace(
            (
                REPOSITORY_ROOT / "fixtures/synthetic/healthy.v1.json"
            ).read_bytes()
        )
        cls.empty = parse_synthetic_trace(
            (
                REPOSITORY_ROOT
                / "fixtures/synthetic/empty-assistant.v1.json"
            ).read_bytes()
        )

    def test_all_token_policy_selects_every_non_padding_position(self) -> None:
        labels = build_labels(self.healthy, ALL_TOKENS)

        self.assertEqual(labels[:31], self.healthy.trace.token_ids[:31])
        self.assertEqual(labels[31:], (IGNORE_INDEX, IGNORE_INDEX))
        audit = audit_label_topology(self.healthy, labels, ALL_TOKENS)
        self.assertEqual(audit.supervised_token_count, 31)
        self.assertEqual(audit.ignored_token_count, 2)
        self.assertEqual(
            [(run.start, run.end, run.length) for run in audit.supervised_runs],
            [(0, 31, 31)],
        )
        self.assertEqual(audit.boundary_supervised_token_count, 17)
        self.assertEqual(audit.off_policy_supervision_positions, ())
        self.assertEqual(audit.missing_eligible_positions, ())

    def test_assistant_policy_selects_only_assistant_content(self) -> None:
        labels = build_labels(self.healthy, ASSISTANT_ONLY)
        expected_positions = set(range(14, 18)) | set(range(26, 29))

        for position, (label, token_id) in enumerate(
            zip(labels, self.healthy.trace.token_ids, strict=True)
        ):
            self.assertEqual(
                label,
                token_id if position in expected_positions else IGNORE_INDEX,
            )
        audit = audit_label_topology(
            self.healthy,
            labels,
            ASSISTANT_ONLY,
        )
        self.assertEqual(audit.eligible_token_count, 7)
        self.assertEqual(audit.supervised_token_count, 7)
        self.assertEqual(
            [(run.start, run.end, run.length) for run in audit.supervised_runs],
            [(14, 18, 4), (26, 29, 3)],
        )
        self.assertEqual(audit.boundary_supervised_token_count, 0)
        self.assertEqual(audit.boundary_leakage_positions, ())
        self.assertEqual(audit.padding_leakage_positions, ())

    def test_policy_construction_is_deterministic(self) -> None:
        self.assertEqual(
            build_labels(self.healthy, ALL_TOKENS),
            build_labels(self.healthy, ALL_TOKENS),
        )
        self.assertEqual(
            build_labels(self.healthy, ASSISTANT_ONLY),
            build_labels(self.healthy, ASSISTANT_ONLY),
        )

    def test_unsupported_policy_is_rejected(self) -> None:
        with self.assertRaises(TopologyError) as caught:
            build_labels(self.healthy, "model_guess")
        self.assertEqual(caught.exception.code, "policy.unsupported")

    def test_auditor_detects_boundary_and_off_policy_leakage(self) -> None:
        labels = list(build_labels(self.healthy, ASSISTANT_ONLY))
        labels[12] = self.healthy.trace.token_ids[12]
        audit = audit_label_topology(
            self.healthy,
            tuple(labels),
            ASSISTANT_ONLY,
        )

        self.assertEqual(audit.boundary_leakage_positions, (12,))
        self.assertEqual(audit.off_policy_supervision_positions, (12,))
        self.assertEqual(audit.boundary_supervised_token_count, 1)

    def test_auditor_detects_user_content_and_padding_leakage(self) -> None:
        labels = list(build_labels(self.healthy, ASSISTANT_ONLY))
        labels[8] = self.healthy.trace.token_ids[8]
        labels[31] = self.healthy.trace.token_ids[31]
        audit = audit_label_topology(
            self.healthy,
            tuple(labels),
            ASSISTANT_ONLY,
        )

        self.assertEqual(audit.boundary_leakage_positions, ())
        self.assertEqual(audit.padding_leakage_positions, (31,))
        self.assertEqual(audit.off_policy_supervision_positions, (8, 31))

    def test_auditor_detects_missing_eligible_targets(self) -> None:
        labels = list(build_labels(self.healthy, ASSISTANT_ONLY))
        labels[14] = IGNORE_INDEX
        audit = audit_label_topology(
            self.healthy,
            tuple(labels),
            ASSISTANT_ONLY,
        )

        self.assertEqual(audit.missing_eligible_positions, (14,))
        self.assertEqual(audit.supervised_runs[0].start, 15)

    def test_auditor_rejects_bad_length_types_and_token_values(self) -> None:
        good = list(build_labels(self.healthy, ASSISTANT_ONLY))
        cases: tuple[tuple[tuple[object, ...], str], ...] = (
            (tuple(good[:-1]), "labels.length"),
            (tuple([True, *good[1:]]), "labels.type"),
            (tuple([999_999, *good[1:]]), "labels.token_mismatch"),
        )
        for labels, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(TopologyError) as caught:
                    audit_label_topology(  # type: ignore[arg-type]
                        self.healthy,
                        labels,
                        ASSISTANT_ONLY,
                    )
                self.assertEqual(caught.exception.code, code)

    def test_empty_assistant_target_is_a_failing_diagnostic(self) -> None:
        report = audit_trace(self.empty)

        self.assertEqual(report.status, "fail")
        self.assertEqual(report.empty_assistant_message_indices, (2,))
        self.assertEqual(report.issue_codes, ("assistant_target.empty",))
        self.assertEqual(report.assistant_only.supervised_token_count, 0)
        self.assertEqual(report.assistant_only.supervised_runs, ())
        self.assertEqual(report.assistant_only.boundary_leakage_positions, ())

    def test_healthy_report_is_pass_and_immutable(self) -> None:
        report = audit_trace(self.healthy)

        self.assertEqual(report.status, "pass")
        self.assertEqual(report.message_count, 5)
        self.assertEqual(report.assistant_message_count, 2)
        self.assertEqual(report.token_count, 33)
        self.assertEqual(report.non_padding_token_count, 31)
        self.assertEqual(report.padding_token_count, 2)
        with self.assertRaises(FrozenInstanceError):
            report.status = "changed"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            report.assistant_only.policy = "changed"  # type: ignore[misc]

    def test_canonical_report_is_stable_and_explicit_about_scope(self) -> None:
        first = canonical_audit_bytes(audit_trace(self.healthy))
        second = canonical_audit_bytes(audit_trace(self.healthy))
        decoded = json.loads(first)

        self.assertEqual(first, second)
        self.assertTrue(first.endswith(b"\n"))
        self.assertEqual(
            audit_sha256(audit_trace(self.healthy)),
            "8e562de5355e1ec7bf516119f2174613ac48b9db5da45bcf8f5492b821c29fab",
        )
        self.assertEqual(decoded["kind"], "loss-topology.audit")
        self.assertEqual(decoded["status"], "pass")
        self.assertFalse(decoded["scope"]["tokenizer_executed"])
        self.assertFalse(decoded["scope"]["tokenizer_mapping_attested"])
        self.assertFalse(decoded["scope"]["trainer_imported_or_executed"])
        self.assertFalse(decoded["scope"]["model_or_dataset_loaded"])
        self.assertFalse(decoded["scope"]["causal_shift_or_loss_computed"])
        self.assertIsNone(decoded["scope"]["quality_metric"])
        self.assertEqual(
            decoded["input_sha256"],
            "95e1dee0360f6dc45b89cbea63c781d4cc34e3fad02406a86b44145edd00a489",
        )
        self.assertEqual(
            decoded["policies"]["assistant_only"]["labels_sha256"],
            "183f9e4db25d0ddedc79534cf52e7c8f953bb4c61b38b653e94c6f68e4d6cc95",
        )


if __name__ == "__main__":
    unittest.main()
