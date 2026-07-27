from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

from tools import render_loss_topology_evidence as evidence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERATED_ROOT = REPOSITORY_ROOT.joinpath(*evidence.OUTPUT_PARTS)


class LossTopologyEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_bytes = (
            GENERATED_ROOT / evidence.MANIFEST_NAME
        ).read_bytes()
        cls.manifest = json.loads(cls.manifest_bytes)

    def test_generated_bundle_matches_a_fresh_real_cli_capture(self) -> None:
        expected = evidence._build_bundle(REPOSITORY_ROOT)

        evidence._check_bundle(REPOSITORY_ROOT, expected)
        self.assertEqual(
            tuple(expected),
            evidence.PUBLICATION_ORDER,
        )

    def test_manifest_is_exact_source_and_artifact_attestation(self) -> None:
        self.assertEqual(
            set(self.manifest),
            {
                "artifacts",
                "cli_executions",
                "fault_injections",
                "kind",
                "schema_version",
                "scope",
                "sources",
            },
        )
        self.assertEqual(self.manifest["schema_version"], 1)
        self.assertEqual(
            self.manifest["kind"],
            "loss-topology.visual-evidence-manifest",
        )
        self.assertEqual(
            [entry["path"] for entry in self.manifest["sources"]],
            list(evidence.SOURCE_PATHS),
        )
        forbidden = {
            "finetune.py",
            "configs/ds_config_zero3.json",
            "requirements.txt",
        }
        self.assertTrue(
            forbidden.isdisjoint(
                entry["path"] for entry in self.manifest["sources"]
            )
        )
        for source in self.manifest["sources"]:
            payload = (REPOSITORY_ROOT / source["path"]).read_bytes()
            self.assertEqual(source["bytes"], len(payload))
            self.assertEqual(
                source["sha256"],
                hashlib.sha256(payload).hexdigest(),
            )

        expected_artifact_paths = [
            f"{'/'.join(evidence.OUTPUT_PARTS)}/{name}"
            for name in evidence.ARTIFACT_NAMES
        ]
        self.assertEqual(
            [entry["path"] for entry in self.manifest["artifacts"]],
            expected_artifact_paths,
        )
        for artifact in self.manifest["artifacts"]:
            payload = (REPOSITORY_ROOT / artifact["path"]).read_bytes()
            self.assertEqual(artifact["bytes"], len(payload))
            self.assertEqual(
                artifact["sha256"],
                hashlib.sha256(payload).hexdigest(),
            )

    def test_manifest_scope_keeps_every_non_claim_explicit(self) -> None:
        scope = self.manifest["scope"]

        self.assertEqual(
            scope["fixture_class"],
            "committed synthetic pretokenized JSON",
        )
        for field in (
            "causal_shift_or_loss_computed",
            "contains_personal_data",
            "contains_secrets",
            "faults_are_cli_inputs_or_outputs",
            "model_or_dataset_loaded",
            "network_used",
            "tokenizer_executed",
            "tokenizer_mapping_attested",
            "trainer_imported_or_executed",
        ):
            self.assertIs(scope[field], False)

    def test_real_cli_records_bind_expected_statuses_and_hashes(self) -> None:
        records = {
            record["id"]: record
            for record in self.manifest["cli_executions"]
        }
        self.assertEqual(set(records), {"healthy", "empty-assistant"})
        self.assertEqual(records["healthy"]["expected_exit_code"], 0)
        self.assertEqual(
            records["healthy"]["output_sha256"],
            "8e562de5355e1ec7bf516119f2174613ac48b9db5da45bcf8f5492b821c29fab",
        )
        self.assertEqual(records["empty-assistant"]["expected_exit_code"], 1)
        self.assertEqual(
            records["empty-assistant"]["output_sha256"],
            "7c04b2309a661b1b0e748cb2ce5948817c65630b27e4ba3131670bd21d1c54d2",
        )
        for record in records.values():
            self.assertEqual(
                record["argv"][:3],
                ["python3", "-m", "loss_topology.cli"],
            )
            self.assertEqual(
                record["stderr_sha256"],
                hashlib.sha256(b"").hexdigest(),
            )

    def test_canonical_reports_preserve_policy_and_scope_facts(self) -> None:
        healthy_bytes = (GENERATED_ROOT / evidence.HEALTHY_NAME).read_bytes()
        empty_bytes = (GENERATED_ROOT / evidence.EMPTY_NAME).read_bytes()
        healthy = json.loads(healthy_bytes)
        empty = json.loads(empty_bytes)

        self.assertTrue(healthy_bytes.endswith(b"\n"))
        self.assertTrue(empty_bytes.endswith(b"\n"))
        self.assertEqual(healthy["status"], "pass")
        self.assertEqual(healthy["trace_summary"]["token_count"], 33)
        self.assertEqual(
            healthy["policies"]["all_tokens"]["supervised_token_count"],
            31,
        )
        self.assertEqual(
            healthy["policies"]["all_tokens"][
                "boundary_supervised_token_count"
            ],
            17,
        )
        self.assertEqual(
            healthy["policies"]["assistant_only"]["supervised_runs"],
            [
                {"end": 18, "length": 4, "start": 14},
                {"end": 29, "length": 3, "start": 26},
            ],
        )
        self.assertEqual(empty["status"], "fail")
        self.assertEqual(
            empty["diagnostics"]["issue_codes"],
            ["assistant_target.empty"],
        )
        for report in (healthy, empty):
            self.assertFalse(report["scope"]["tokenizer_executed"])
            self.assertFalse(
                report["scope"]["trainer_imported_or_executed"]
            )
            self.assertFalse(report["scope"]["model_or_dataset_loaded"])
            self.assertFalse(report["scope"]["causal_shift_or_loss_computed"])

    def test_fault_manifest_records_public_in_memory_audits(self) -> None:
        faults = {
            item["id"]: item for item in self.manifest["fault_injections"]
        }

        self.assertEqual(
            faults["boundary-leak"]["detected"],
            {
                "boundary_leakage_positions": [12],
                "missing_eligible_positions": [],
                "off_policy_supervision_positions": [12],
                "padding_leakage_positions": [],
                "supervised_runs": [
                    {"end": 13, "start": 12},
                    {"end": 18, "start": 14},
                    {"end": 29, "start": 26},
                ],
            },
        )
        self.assertEqual(
            faults["user-padding-leak"]["detected"][
                "off_policy_supervision_positions"
            ],
            [8, 31],
        )
        self.assertEqual(
            faults["user-padding-leak"]["detected"][
                "padding_leakage_positions"
            ],
            [31],
        )
        self.assertEqual(
            faults["missing-target"]["detected"][
                "missing_eligible_positions"
            ],
            [14],
        )
        for fault in faults.values():
            self.assertEqual(fault["policy"], "assistant_only")
            self.assertIn(
                "not CLI input or output",
                fault["execution_boundary"],
            )

    def test_transcript_is_real_relative_and_boundary_labeled(self) -> None:
        transcript = (
            GENERATED_ROOT / evidence.TRANSCRIPT_NAME
        ).read_text(encoding="utf-8")

        self.assertIn("REAL SUBPROCESS CAPTURE", transcript)
        self.assertIn(
            "$ python3 -m loss_topology.cli --input "
            "input/healthy.v1.json --output output/healthy.audit.json",
            transcript,
        )
        self.assertIn("[recorder] exit=0 stderr=empty", transcript)
        self.assertIn("[recorder] exit=1 stderr=empty", transcript)
        self.assertIn("not a CLI error", transcript)
        self.assertIn("it is not CLI output", transcript)
        self.assertNotIn(str(REPOSITORY_ROOT), transcript)
        self.assertNotIn("finetune.py", transcript)

    def test_svg_artifacts_are_accessible_and_self_contained(self) -> None:
        for name in (
            evidence.POLICY_SVG_NAME,
            evidence.FAULT_SVG_NAME,
            evidence.TRANSCRIPT_SVG_NAME,
        ):
            with self.subTest(name=name):
                payload = (GENERATED_ROOT / name).read_bytes()
                root = ET.fromstring(payload)
                namespace = "{http://www.w3.org/2000/svg}"
                self.assertEqual(root.tag, f"{namespace}svg")
                self.assertEqual(root.attrib["role"], "img")
                self.assertIsNotNone(root.find(f"{namespace}title"))
                self.assertIsNotNone(root.find(f"{namespace}desc"))
                self.assertEqual(root.findall(f".//{namespace}script"), [])
                self.assertEqual(root.findall(f".//{namespace}image"), [])
                for element in root.iter():
                    self.assertFalse(
                        any(key.endswith("href") for key in element.attrib)
                    )

        policy = (
            GENERATED_ROOT / evidence.POLICY_SVG_NAME
        ).read_text(encoding="utf-8")
        self.assertIn("boundary selection is intentional", policy.lower())
        self.assertIn("NO TOKENIZER", policy)
        fault = (
            GENERATED_ROOT / evidence.FAULT_SVG_NAME
        ).read_text(encoding="utf-8")
        self.assertIn("not CLI inputs or outputs", fault)
        self.assertIn("public audit_label_topology", fault)

    def test_generator_imports_only_stdlib_and_loss_topology(self) -> None:
        path = REPOSITORY_ROOT / "tools/render_loss_topology_evidence.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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

        allowed = set(sys.stdlib_module_names) | {"loss_topology"}
        self.assertLessEqual(imported_roots, allowed)
        self.assertNotIn("finetune", imported_roots)

    def test_manifest_last_failure_restores_previous_bundle(self) -> None:
        build_root = REPOSITORY_ROOT / "build"
        build_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="evidence-rollback-",
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
            original_replace = evidence._replace_entry

            def fail_manifest(
                source: str,
                destination: str,
                directory_fd: int,
            ) -> None:
                if (
                    destination == evidence.MANIFEST_NAME
                    and source.startswith(evidence.STAGE_PREFIX)
                ):
                    raise OSError("synthetic manifest publication failure")
                original_replace(source, destination, directory_fd)

            with mock.patch.object(
                evidence,
                "_replace_entry",
                side_effect=fail_manifest,
            ):
                with self.assertRaises(evidence.EvidenceError) as caught:
                    evidence._publish_bundle(root, new_bundle)
            self.assertEqual(caught.exception.code, "io.publish_failed")

            generated = root.joinpath(*evidence.OUTPUT_PARTS)
            for name, payload in old_bundle.items():
                self.assertEqual((generated / name).read_bytes(), payload)
            self.assertEqual(
                set(os.listdir(generated)),
                set(evidence.PUBLICATION_ORDER),
            )

    def test_non_regular_destination_fails_closed(self) -> None:
        build_root = REPOSITORY_ROOT / "build"
        build_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="evidence-symlink-",
            dir=build_root,
        ) as temporary_name:
            root = Path(temporary_name)
            generated = root.joinpath(*evidence.OUTPUT_PARTS)
            generated.mkdir(parents=True)
            target = root / "sentinel.txt"
            target.write_bytes(b"sentinel\n")
            (generated / evidence.HEALTHY_NAME).symlink_to(target)
            bundle = {
                name: f"new:{name}\n".encode("ascii")
                for name in evidence.PUBLICATION_ORDER
            }

            with self.assertRaises(evidence.EvidenceError) as caught:
                evidence._publish_bundle(root, bundle)
            self.assertEqual(
                caught.exception.code,
                "io.destination_not_regular",
            )
            self.assertEqual(target.read_bytes(), b"sentinel\n")

    def test_check_requires_an_exact_generated_inventory(self) -> None:
        build_root = REPOSITORY_ROOT / "build"
        build_root.mkdir(exist_ok=True)
        for extra_kind in ("regular", "symlink"):
            with (
                self.subTest(extra_kind=extra_kind),
                tempfile.TemporaryDirectory(
                    prefix=f"evidence-extra-{extra_kind}-",
                    dir=build_root,
                ) as temporary_name,
            ):
                root = Path(temporary_name)
                bundle = {
                    name: f"expected:{name}\n".encode("ascii")
                    for name in evidence.PUBLICATION_ORDER
                }
                evidence._publish_bundle(root, bundle)
                generated = root.joinpath(*evidence.OUTPUT_PARTS)
                extra = generated / "unmanifested-output.svg"
                if extra_kind == "regular":
                    extra.write_bytes(b"unmanifested\n")
                else:
                    extra.symlink_to(evidence.HEALTHY_NAME)

                with self.assertRaises(evidence.EvidenceError) as caught:
                    evidence._check_bundle(root, bundle)
                self.assertEqual(caught.exception.code, "bundle.inventory")


if __name__ == "__main__":
    unittest.main()
