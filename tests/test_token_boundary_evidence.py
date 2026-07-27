from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import mock

from token_boundary.contract import (
    canonical_boundary_case_bytes,
    parse_boundary_case,
)
from tools import render_token_boundary_evidence as evidence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERATED_ROOT = REPOSITORY_ROOT.joinpath(*evidence.OUTPUT_PARTS)
TOKENIZERS_AVAILABLE = importlib.util.find_spec("tokenizers") is not None


class TokenBoundaryEvidenceTests(unittest.TestCase):
    manifest_bytes: bytes
    manifest: dict[str, Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_bytes = (GENERATED_ROOT / evidence.MANIFEST_NAME).read_bytes()
        cls.manifest = json.loads(cls.manifest_bytes)

    @unittest.skipUnless(
        TOKENIZERS_AVAILABLE,
        "tokenizers==0.21.4 is required to reconstruct executed evidence",
    )
    def test_bundle_matches_fresh_cli_executions_byte_for_byte(self) -> None:
        expected = evidence._build_bundle(REPOSITORY_ROOT)

        evidence._check_bundle(REPOSITORY_ROOT, expected)
        self.assertEqual(tuple(expected), evidence.PUBLICATION_ORDER)

    def test_manifest_attests_exact_sources_and_artifacts(self) -> None:
        self.assertEqual(
            set(self.manifest),
            {
                "artifacts",
                "cli_executions",
                "kind",
                "legacy_algorithm_binding",
                "runtime",
                "schema_version",
                "scope",
                "sources",
                "truncation_sweep",
            },
        )
        self.assertEqual(self.manifest["schema_version"], 1)
        self.assertEqual(
            self.manifest["kind"],
            "token-boundary.visual-evidence-manifest",
        )
        self.assertEqual(
            [entry["path"] for entry in self.manifest["sources"]],
            list(evidence.SOURCE_PATHS),
        )
        for source in self.manifest["sources"]:
            payload = (REPOSITORY_ROOT / source["path"]).read_bytes()
            self.assertEqual(source["bytes"], len(payload))
            self.assertEqual(
                source["sha256"],
                hashlib.sha256(payload).hexdigest(),
            )

        expected_paths = [
            f"{'/'.join(evidence.OUTPUT_PARTS)}/{name}"
            for name in evidence.ARTIFACT_NAMES
        ]
        self.assertEqual(
            [entry["path"] for entry in self.manifest["artifacts"]],
            expected_paths,
        )
        for artifact in self.manifest["artifacts"]:
            payload = (REPOSITORY_ROOT / artifact["path"]).read_bytes()
            self.assertEqual(artifact["bytes"], len(payload))
            self.assertEqual(
                artifact["sha256"],
                hashlib.sha256(payload).hexdigest(),
            )

    def test_seven_cli_records_match_fixed_real_outputs(self) -> None:
        records = {record["id"]: record for record in self.manifest["cli_executions"]}

        self.assertEqual(set(records), set(evidence.EXPECTED_RUNS))
        for identifier, expected in evidence.EXPECTED_RUNS.items():
            record = records[identifier]
            report_name = evidence.REPORT_NAMES[identifier]
            report = (GENERATED_ROOT / report_name).read_bytes()

            self.assertEqual(record["argv"], ["python3", "-m", "token_boundary.cli"])
            self.assertEqual(record["expected_exit_code"], expected["exit"])
            self.assertEqual(record["observed_exit_code"], expected["exit"])
            self.assertEqual(record["expected_status"], expected["status"])
            self.assertEqual(record["observed_status"], expected["status"])
            self.assertEqual(
                record["expected_primary_classification"],
                expected["primary"],
            )
            self.assertEqual(
                record["observed_primary_classification"],
                expected["primary"],
            )
            self.assertEqual(record["report_bytes"], len(report))
            self.assertEqual(
                record["report_sha256"],
                hashlib.sha256(report).hexdigest(),
            )
            self.assertEqual(record["report_sha256"], expected["sha256"])
            self.assertEqual(record["stderr_bytes"], 0)
            self.assertEqual(record["stderr_sha256"], evidence.EMPTY_SHA256)
            self.assertTrue(record["stdout_is_canonical_report"])
            self.assertTrue(report.endswith(b"\n"))
            self.assertEqual(report.count(b"\n"), 1)

    def test_reports_preserve_exact_counterexample_facts(self) -> None:
        reports = {
            identifier: json.loads((GENERATED_ROOT / name).read_bytes())
            for identifier, name in evidence.REPORT_NAMES.items()
        }

        self.assertEqual(reports["aligned"]["status"], "pass")
        self.assertEqual(
            reports["merge-cross-boundary"]["diagnostics"]["cross_boundary_positions"],
            [1],
        )
        self.assertEqual(
            reports["nfc-cross-boundary"]["normalization"]["combined"][
                "cross_boundary_output_positions"
            ],
            [0],
        )
        self.assertEqual(
            reports["normalized-away"]["diagnostics"]["indeterminate_reasons"],
            ["target.no_attributable_token"],
        )
        self.assertEqual(
            reports["partial-truncation"]["diagnostics"][
                "truncated_target_token_count"
            ],
            2,
        )
        self.assertEqual(
            reports["right-strip-drift"]["diagnostics"]["prompt_leakage_positions"],
            [2],
        )
        self.assertEqual(
            reports["target-eliminated"]["diagnostics"]["elimination_causes"],
            ["truncation"],
        )
        for report in reports.values():
            self.assertNotIn("source", report["input_identity"])
            self.assertNotIn("target", report["input_identity"])
            self.assertFalse(report["scope"]["network_used"])
            self.assertFalse(report["scope"]["inherited_trainer_executed"])
            self.assertFalse(report["scope"]["training_or_evaluation_performed"])

    def test_runtime_and_scope_are_narrow_and_explicit(self) -> None:
        runtime = self.manifest["runtime"]

        self.assertEqual(runtime["engine"], "tokenizers")
        self.assertEqual(runtime["engine_version"], "0.21.4")
        self.assertEqual(runtime["artifact_id"], "local-boundary-bpe-v1")
        self.assertEqual(runtime["artifact_bytes"], 1533)
        self.assertEqual(
            runtime["artifact_sha256"],
            "29508edbb44ce9cbe77cdde972c0919fe3df8ee2ae1270e69545314f2e1f8358",
        )
        self.assertEqual(
            runtime["normalizers"],
            ["NFC", "Strip(left=false,right=true)"],
        )
        self.assertEqual(runtime["post_processor"], "prefix-bos-only")
        self.assertEqual(runtime["truncation"], "caller-bounded-right")

        scope = self.manifest["scope"]
        self.assertTrue(scope["local_synthetic_tokenizer_artifact_executed"])
        for field in (
            "contains_personal_data",
            "contains_secrets",
            "deepseek_model_executed",
            "deepseek_tokenizer_attested",
            "forward_pass_or_loss_computed",
            "gpu_used",
            "inherited_trainer_executed",
            "inherited_trainer_fixed",
            "model_or_dataset_loaded",
            "model_quality_measured",
            "model_training_or_evaluation_performed",
            "network_used",
            "prevalence_measured",
            "sweep_is_benchmark_or_sample",
            "universal_masking_policy_claimed",
        ):
            self.assertIs(scope[field], False)

    def test_legacy_cutoff_binding_is_attestation_only(self) -> None:
        binding = self.manifest["legacy_algorithm_binding"]

        self.assertEqual(
            binding,
            {
                "algorithm_id": "standalone-source-token-count",
                "attestation_path": "provenance/legacy-snapshot.v1.json",
                "attestation_sha256": (
                    "be86490f71ea597a441f26b5473156e7a444d6b974fa3eabedd4979ddb9123b3"
                ),
                "semantic_equivalence_claimed": False,
                "snapshot_source_git_commit": (
                    "6912653d881bedee71ef527bc5650db55f115779"
                ),
                "source_git_blob_sha1": ("d334caa2cec91ed97a1974cdf61e3ab0d3edf415"),
                "source_imported_or_executed": False,
                "source_path": "finetune.py",
                "source_sha256": (
                    "5de3316c8cf37edea97e83230fd90bf01092582dc25016c87fdc404aa1024e26"
                ),
            },
        )

    def test_legacy_binding_rejects_current_or_declared_identity_drift(
        self,
    ) -> None:
        manifest_bytes = (
            REPOSITORY_ROOT / "provenance/legacy-snapshot.v1.json"
        ).read_bytes()
        original = json.loads(manifest_bytes)
        trainer_bytes = (REPOSITORY_ROOT / "finetune.py").read_bytes()
        cases: list[tuple[str, dict[str, Any], bytes]] = [
            ("current-bytes", original, trainer_bytes + b"\n"),
        ]
        for field, replacement in (
            ("byte_length", len(trainer_bytes) + 1),
            ("sha256", "0" * 64),
            ("git_blob_sha1", "0" * 40),
        ):
            changed = json.loads(manifest_bytes)
            trainer = next(
                item for item in changed["files"] if item["path"] == "finetune.py"
            )
            trainer[field] = replacement
            cases.append((field, changed, trainer_bytes))

        for label, manifest, observed_bytes in cases:
            with (
                self.subTest(label=label),
                self.assertRaises(evidence.EvidenceError) as caught,
            ):
                evidence._legacy_binding(
                    manifest,
                    manifest_bytes,
                    observed_bytes,
                )
            self.assertEqual(caught.exception.code, "legacy.source_drift")

    def test_transcript_contains_verbatim_reports_and_no_host_path(self) -> None:
        transcript_bytes = (GENERATED_ROOT / evidence.TRANSCRIPT_NAME).read_bytes()
        transcript = transcript_bytes.decode("ascii")

        self.assertIn("REAL STDIN SUBPROCESS CAPTURE", transcript)
        self.assertEqual(
            transcript.count("--- BEGIN CANONICAL STDOUT"),
            len(evidence.EXPECTED_RUNS),
        )
        self.assertEqual(
            transcript.count("--- END CANONICAL STDOUT ---"),
            len(evidence.EXPECTED_RUNS),
        )
        for identifier, report_name in evidence.REPORT_NAMES.items():
            report = (GENERATED_ROOT / report_name).read_text("ascii")
            self.assertIn(report.rstrip("\n"), transcript)
            self.assertIn(
                f"input/{identifier}.v1.json",
                transcript,
            )
        self.assertNotIn(str(REPOSITORY_ROOT), transcript)
        self.assertNotIn(sys.executable, transcript)
        self.assertNotIn("Omar", transcript)
        self.assertNotIn("ubuntu", transcript.lower())

    def test_sweep_csv_has_all_56_executed_cells(self) -> None:
        with (GENERATED_ROOT / evidence.SWEEP_NAME).open(
            encoding="utf-8",
            newline="",
        ) as stream:
            rows = list(csv.DictReader(stream))

        self.assertEqual(len(rows), 56)
        self.assertEqual(
            {(row["case_id"], int(row["max_length"])) for row in rows},
            {
                (identifier, max_length)
                for identifier in evidence.EXPECTED_RUNS
                for max_length in evidence.SWEEP_LENGTHS
            },
        )
        partial = {
            int(row["max_length"]): row
            for row in rows
            if row["case_id"] == "partial-truncation"
        }
        self.assertEqual(
            [
                (
                    length,
                    partial[length]["primary_classification"],
                    partial[length]["retained_target_tokens"],
                    partial[length]["full_target_tokens"],
                )
                for length in evidence.SWEEP_LENGTHS
            ],
            [
                (2, "target_eliminated", "0", "4"),
                (3, "target_eliminated", "0", "4"),
                (4, "target_eliminated", "0", "4"),
                (5, "partial_target_truncation", "1", "4"),
                (6, "partial_target_truncation", "2", "4"),
                (7, "partial_target_truncation", "3", "4"),
                (8, "aligned", "4", "4"),
                (9, "aligned", "4", "4"),
            ],
        )
        target = {
            int(row["max_length"]): row
            for row in rows
            if row["case_id"] == "target-eliminated"
        }
        self.assertTrue(
            all(
                target[length]["retained_target_tokens"] == "0"
                for length in range(2, 6)
            )
        )
        self.assertTrue(
            all(
                target[length]["primary_classification"] == "aligned"
                for length in range(6, 10)
            )
        )

    def test_sweep_manifest_binds_csv_and_every_execution(self) -> None:
        sweep = self.manifest["truncation_sweep"]
        csv_bytes = (GENERATED_ROOT / evidence.SWEEP_NAME).read_bytes()

        self.assertEqual(sweep["run_count"], 56)
        self.assertEqual(sweep["max_lengths"], list(range(2, 10)))
        self.assertEqual(sweep["csv_bytes"], len(csv_bytes))
        self.assertEqual(
            sweep["csv_sha256"],
            hashlib.sha256(csv_bytes).hexdigest(),
        )
        self.assertEqual(len(sweep["executions"]), 56)
        self.assertEqual(
            sum(sweep["classification_counts"].values()),
            56,
        )
        self.assertEqual(
            sweep["execution_boundary"],
            "fresh public stdin CLI subprocess for every cell",
        )
        for execution in sweep["executions"]:
            self.assertEqual(len(execution["stdout_sha256"]), 64)
            self.assertIn(execution["exit_code"], (0, 1))
            fixture = parse_boundary_case(
                (REPOSITORY_ROOT / execution["base_fixture_path"]).read_bytes()
            )
            canonical_input = canonical_boundary_case_bytes(
                replace(fixture, max_length=execution["max_length"])
            )
            self.assertEqual(
                execution["canonical_input_sha256"],
                hashlib.sha256(canonical_input).hexdigest(),
            )

    def test_svg_artifacts_are_accessible_and_self_contained(self) -> None:
        for name in (
            evidence.ARCHITECTURE_SVG_NAME,
            evidence.MECHANISMS_SVG_NAME,
            evidence.TRANSCRIPT_SVG_NAME,
            evidence.LANES_SVG_NAME,
            evidence.MATRIX_SVG_NAME,
        ):
            with self.subTest(name=name):
                payload = (GENERATED_ROOT / name).read_bytes()
                root = ET.fromstring(payload)
                namespace = f"{{{evidence.SVG_NAMESPACE}}}"

                self.assertEqual(root.tag, f"{namespace}svg")
                self.assertEqual(root.attrib["role"], "img")
                self.assertLessEqual(int(root.attrib["width"]), 1000)
                self.assertIsNotNone(root.find(f"{namespace}title"))
                self.assertIsNotNone(root.find(f"{namespace}desc"))
                self.assertEqual(root.findall(f".//{namespace}script"), [])
                self.assertEqual(root.findall(f".//{namespace}image"), [])
                self.assertEqual(root.findall(f".//{namespace}use"), [])
                for element in root.iter():
                    self.assertFalse(
                        any(key.endswith("href") for key in element.attrib)
                    )
                text = payload.decode("utf-8")
                self.assertNotIn(str(REPOSITORY_ROOT), text)
                self.assertNotIn("Omar", text)
                self.assertNotIn("ubuntu", text.lower())

    def test_visuals_include_direct_factual_labels(self) -> None:
        architecture = (GENERATED_ROOT / evidence.ARCHITECTURE_SVG_NAME).read_text(
            "utf-8"
        )
        lanes = (GENERATED_ROOT / evidence.LANES_SVG_NAME).read_text("utf-8")
        mechanisms = (GENERATED_ROOT / evidence.MECHANISMS_SVG_NAME).read_text("utf-8")
        matrix = (GENERATED_ROOT / evidence.MATRIX_SVG_NAME).read_text("utf-8")
        cli = (GENERATED_ROOT / evidence.TRANSCRIPT_SVG_NAME).read_text("utf-8")

        self.assertIn("QUARANTINED SOURCE BINDING", architecture)
        self.assertIn("three fresh tokenizer instances", architecture)
        for identifier in evidence.EXPECTED_RUNS:
            self.assertIn(identifier, lanes)
            self.assertIn(identifier, matrix)
            self.assertIn(f"input/{identifier}.v1.json", cli)
        self.assertIn("BPE merge across boundary", mechanisms)
        self.assertIn("NFC composition across boundary", mechanisms)
        self.assertIn("Right-strip context drift", mechanisms)
        self.assertIn("Target normalized away", mechanisms)
        self.assertIn("retained/full", matrix)
        self.assertIn("Color is redundant", matrix)

    def test_generator_import_surface_is_closed(self) -> None:
        path = REPOSITORY_ROOT / "tools/render_token_boundary_evidence.py"
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

        allowed = set(sys.stdlib_module_names) | {"token_boundary"}
        self.assertLessEqual(imported_roots, allowed)
        self.assertNotIn("finetune", imported_roots)
        self.assertTrue(
            imported_roots.isdisjoint({"requests", "socket", "torch", "transformers"})
        )

    def test_invalid_generator_arguments_fail_before_execution(self) -> None:
        stderr = io.BytesIO()
        wrapper = io.TextIOWrapper(stderr, encoding="utf-8")
        with (
            mock.patch.object(sys, "stderr", wrapper),
            mock.patch.object(evidence, "_build_bundle") as builder,
        ):
            result = evidence.main(())
            wrapper.flush()

        self.assertEqual(result, 2)
        builder.assert_not_called()
        self.assertEqual(
            json.loads(stderr.getvalue()),
            {
                "error": {"code": "cli.arguments"},
                "kind": "token-boundary.evidence-error",
                "status": "error",
            },
        )

    def test_broken_error_channel_is_best_effort(self) -> None:
        stderr = mock.Mock()
        stderr.buffer.write.side_effect = OSError("synthetic broken stderr")

        with mock.patch.object(sys, "stderr", stderr):
            evidence._emit_error("internal.error")

        stderr.buffer.write.assert_called_once()

    def test_manifest_last_failure_restores_previous_bundle(self) -> None:
        build_root = REPOSITORY_ROOT / "build"
        build_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="token-evidence-rollback-",
            dir=build_root,
        ) as temporary_name:
            root = Path(temporary_name)
            old_bundle = {
                name: f"old:{name}\n".encode("ascii")
                for name in evidence.PUBLICATION_ORDER
            }
            new_bundle = {
                name: f"new:{name}\n".encode("ascii")
                for name in evidence.PUBLICATION_ORDER
            }
            evidence._publish_bundle(root, old_bundle)
            original_replace = os.replace
            failed = False

            def fail_manifest(
                source: os.PathLike[str], destination: os.PathLike[str]
            ) -> None:
                nonlocal failed
                source_path = Path(source)
                destination_path = Path(destination)
                if (
                    not failed
                    and destination_path.name == evidence.MANIFEST_NAME
                    and ".token-boundary-evidence-stage-" in source_path.parent.name
                ):
                    failed = True
                    raise OSError("synthetic manifest publication failure")
                original_replace(source, destination)

            with (
                mock.patch.object(os, "replace", side_effect=fail_manifest),
                self.assertRaises(evidence.EvidenceError) as caught,
            ):
                evidence._publish_bundle(root, new_bundle)
            self.assertEqual(caught.exception.code, "io.publish_failed")
            generated = root.joinpath(*evidence.OUTPUT_PARTS)
            for name, payload in old_bundle.items():
                self.assertEqual((generated / name).read_bytes(), payload)

    def test_post_publication_validation_failure_rolls_back(self) -> None:
        build_root = REPOSITORY_ROOT / "build"
        build_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="token-evidence-post-validate-",
            dir=build_root,
        ) as temporary_name:
            root = Path(temporary_name)
            old_bundle = {
                name: f"old:{name}\n".encode("ascii")
                for name in evidence.PUBLICATION_ORDER
            }
            new_bundle = {
                name: f"new:{name}\n".encode("ascii")
                for name in evidence.PUBLICATION_ORDER
            }
            evidence._publish_bundle(root, old_bundle)
            original_validate = evidence._validate_inventory

            def fail_complete_inventory(
                output_root: Path,
                *,
                allow_missing: bool,
            ) -> None:
                if not allow_missing:
                    raise evidence.EvidenceError("synthetic.validation")
                original_validate(
                    output_root,
                    allow_missing=allow_missing,
                )

            with (
                mock.patch.object(
                    evidence,
                    "_validate_inventory",
                    side_effect=fail_complete_inventory,
                ),
                self.assertRaises(evidence.EvidenceError) as caught,
            ):
                evidence._publish_bundle(root, new_bundle)
            self.assertEqual(caught.exception.code, "io.publish_failed")
            generated = root.joinpath(*evidence.OUTPUT_PARTS)
            for name, payload in old_bundle.items():
                self.assertEqual((generated / name).read_bytes(), payload)

    def test_partial_stage_creation_is_cleaned(self) -> None:
        build_root = REPOSITORY_ROOT / "build"
        build_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="token-evidence-partial-stage-",
            dir=build_root,
        ) as temporary_name:
            root = Path(temporary_name)
            bundle = {
                name: f"value:{name}\n".encode("ascii")
                for name in evidence.PUBLICATION_ORDER
            }
            original_mkdtemp = tempfile.mkdtemp
            created: list[Path] = []

            def fail_second_mkdtemp(
                suffix: str | None = None,
                prefix: str | None = None,
                dir: str | os.PathLike[str] | None = None,
            ) -> str:
                if created:
                    raise OSError("synthetic backup creation failure")
                path = Path(
                    original_mkdtemp(
                        suffix=suffix,
                        prefix=prefix,
                        dir=dir,
                    )
                )
                created.append(path)
                return str(path)

            with (
                mock.patch.object(
                    tempfile,
                    "mkdtemp",
                    side_effect=fail_second_mkdtemp,
                ),
                self.assertRaises(evidence.EvidenceError) as caught,
            ):
                evidence._publish_bundle(root, bundle)
            self.assertEqual(caught.exception.code, "io.stage_failed")
            self.assertEqual(len(created), 1)
            self.assertFalse(created[0].exists())

    def test_symlinked_output_parent_is_rejected(self) -> None:
        build_root = REPOSITORY_ROOT / "build"
        build_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="token-evidence-parent-link-",
            dir=build_root,
        ) as temporary_name:
            root = Path(temporary_name)
            outside = root / "outside"
            outside.mkdir()
            docs = root / "docs"
            docs.mkdir()
            (docs / "token-boundary").symlink_to(outside)
            bundle = {
                name: f"value:{name}\n".encode("ascii")
                for name in evidence.PUBLICATION_ORDER
            }

            with self.assertRaises(evidence.EvidenceError) as caught:
                evidence._publish_bundle(root, bundle)
            self.assertEqual(
                caught.exception.code,
                "io.output_directory_unavailable",
            )
            self.assertEqual(list(outside.iterdir()), [])

    def test_non_regular_destination_and_extra_inventory_fail_closed(self) -> None:
        build_root = REPOSITORY_ROOT / "build"
        build_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="token-evidence-safety-",
            dir=build_root,
        ) as temporary_name:
            root = Path(temporary_name)
            bundle = {
                name: f"value:{name}\n".encode("ascii")
                for name in evidence.PUBLICATION_ORDER
            }
            evidence._publish_bundle(root, bundle)
            generated = root.joinpath(*evidence.OUTPUT_PARTS)
            extra = generated / "unmanifested.svg"
            extra.write_bytes(b"extra\n")

            with self.assertRaises(evidence.EvidenceError) as extra_caught:
                evidence._check_bundle(root, bundle)
            self.assertEqual(extra_caught.exception.code, "bundle.inventory")
            extra.unlink()

            sentinel = root / "sentinel"
            sentinel.write_bytes(b"sentinel\n")
            destination = generated / evidence.TRANSCRIPT_NAME
            destination.unlink()
            destination.symlink_to(sentinel)
            with self.assertRaises(evidence.EvidenceError) as link_caught:
                evidence._publish_bundle(root, bundle)
            self.assertEqual(link_caught.exception.code, "io.file_not_regular")
            self.assertEqual(sentinel.read_bytes(), b"sentinel\n")


if __name__ == "__main__":
    unittest.main()
