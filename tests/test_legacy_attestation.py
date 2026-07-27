from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from tools.verify_legacy_snapshot import (
    AttestationError,
    semantic_json_sha256,
    verify_repository,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ATTESTED_PATHS = (
    Path("configs/ds_config_zero3.json"),
    Path("finetune.py"),
    Path("requirements.txt"),
    Path("provenance/legacy-snapshot.v1.json"),
    Path("third_party/deepseek-moe/LICENSE-CODE"),
)


class LegacyAttestationTests(unittest.TestCase):
    def _copy_attested_tree(self, destination: Path) -> None:
        for relative in ATTESTED_PATHS:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPOSITORY_ROOT / relative, target)

    def _temporary_snapshot(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        self._copy_attested_tree(root)
        return temporary, root

    def test_repository_snapshot_verifies(self) -> None:
        report = verify_repository(REPOSITORY_ROOT)

        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["legacy_state"], "quarantined")
        self.assertEqual(len(report["files_verified"]), 3)
        self.assertEqual(report["network_access"], "not-used")

    def test_manifest_records_the_audited_file_identities(self) -> None:
        manifest = json.loads(
            (REPOSITORY_ROOT / "provenance/legacy-snapshot.v1.json").read_text(
                encoding="utf-8"
            )
        )

        identities = {
            entry["path"]: (entry["byte_length"], entry["sha256"])
            for entry in manifest["files"]
        }
        self.assertEqual(
            identities,
            {
                "configs/ds_config_zero3.json": (
                    1299,
                    "c0efd31093f7d6e2ca8ea5fcd77e28a7aff6c3306893af56622850faa104d6a7",
                ),
                "finetune.py": (
                    8667,
                    "5de3316c8cf37edea97e83230fd90bf01092582dc25016c87fdc404aa1024e26",
                ),
                "requirements.txt": (
                    111,
                    "da3b0ecac531b2f304528f5778c136eef38b8deca8e7dea548c95e7a40e8ea77",
                ),
            },
        )

    def test_manifest_records_bounded_provenance_claims(self) -> None:
        manifest = json.loads(
            (REPOSITORY_ROOT / "provenance/legacy-snapshot.v1.json").read_text(
                encoding="utf-8"
            )
        )
        provenance = manifest["provenance"]
        observations = {item["id"]: item for item in provenance["observations"]}

        self.assertEqual(
            manifest["snapshot"]["source_git_commit"],
            "6912653d881bedee71ef527bc5650db55f115779",
        )
        self.assertEqual(
            provenance["upstream_repository"],
            "https://github.com/deepseek-ai/DeepSeek-MoE",
        )
        self.assertEqual(
            provenance["upstream_git_commit"],
            "66edeee5a4f75cbd76e0316229ad101805a90e01",
        )
        self.assertEqual(
            [
                (item["path"], item["git_blob_sha1"], item["byte_length"])
                for item in provenance["upstream_file_identities"]
            ],
            [
                (
                    "finetune/finetune.py",
                    "244ff4e11416ab4687df3a7343df36846ff45e81",
                    13274,
                ),
                (
                    "finetune/configs/ds_config_zero3.json",
                    "73f3b5f4c430d1ff5ab5ac9e82c11f436440e728",
                    1348,
                ),
                (
                    "LICENSE-CODE",
                    "d84f527e101b2cdd171e2b14253f84ea4fedabe9",
                    1065,
                ),
            ],
        )
        self.assertEqual(
            provenance["upstream_code_license"]["local_notice_sha256"],
            "6e4c38e1172f42fdbff13edf9a7a017679fb82b0fde415a3e8b3c31c6ed4a4e4",
        )
        self.assertEqual(
            (
                observations["trainer-line-lcs"]["matching_line_count"],
                observations["trainer-line-lcs"]["local_line_count"],
            ),
            (185, 230),
        )
        self.assertFalse(
            observations["trainer-line-lcs"][
                "reproducible_from_this_repository_alone"
            ]
        )
        self.assertEqual(
            observations["zero3-semantic-json"]["semantic_sha256"],
            "ac305ab8aba093eb0a29f94629baf3c89ca266077f1246ae89a42b3648aaf23e",
        )
        self.assertFalse(
            observations["zero3-semantic-json"][
                "reproducible_from_this_repository_alone"
            ]
        )

    def test_byte_drift_fails_closed(self) -> None:
        temporary, root = self._temporary_snapshot()
        self.addCleanup(temporary.cleanup)
        path = root / "finetune.py"
        payload = bytearray(path.read_bytes())
        payload[0] ^= 1
        path.write_bytes(payload)

        with self.assertRaisesRegex(AttestationError, "SHA-256 changed"):
            verify_repository(root)

    def test_manifest_cannot_self_bless_modified_legacy_bytes(self) -> None:
        temporary, root = self._temporary_snapshot()
        self.addCleanup(temporary.cleanup)
        legacy_path = root / "finetune.py"
        payload = bytearray(legacy_path.read_bytes())
        payload[0] ^= 1
        changed = bytes(payload)
        legacy_path.write_bytes(changed)

        manifest_path = root / "provenance/legacy-snapshot.v1.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = next(
            item for item in manifest["files"] if item["path"] == "finetune.py"
        )
        entry["byte_length"] = len(changed)
        entry["line_count"] = len(changed.splitlines())
        entry["sha256"] = hashlib.sha256(changed).hexdigest()
        header = f"blob {len(changed)}\0".encode()
        entry["git_blob_sha1"] = hashlib.sha1(
            header + changed,
            usedforsecurity=False,
        ).hexdigest()
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            AttestationError,
            "identity differs from the trusted snapshot",
        ):
            verify_repository(root)

    def test_format_only_json_drift_still_fails_byte_attestation(self) -> None:
        temporary, root = self._temporary_snapshot()
        self.addCleanup(temporary.cleanup)
        path = root / "configs/ds_config_zero3.json"
        original_semantic_hash = semantic_json_sha256(path.read_bytes())
        value = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(
            json.dumps(value, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )

        self.assertEqual(semantic_json_sha256(path.read_bytes()), original_semantic_hash)
        with self.assertRaisesRegex(AttestationError, "byte length changed"):
            verify_repository(root)

    def test_duplicate_manifest_key_is_rejected(self) -> None:
        temporary, root = self._temporary_snapshot()
        self.addCleanup(temporary.cleanup)
        path = root / "provenance/legacy-snapshot.v1.json"
        text = path.read_text(encoding="utf-8")
        path.write_text(
            text.replace(
                '"schema_version": 1,',
                '"schema_version": 1, "schema_version": 1,',
                1,
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(AttestationError, "duplicate JSON key"):
            verify_repository(root)

    def test_symlinked_legacy_file_is_rejected(self) -> None:
        temporary, root = self._temporary_snapshot()
        self.addCleanup(temporary.cleanup)
        path = root / "requirements.txt"
        copy = root / "requirements-copy.txt"
        shutil.copyfile(path, copy)
        path.unlink()
        path.symlink_to(copy.name)

        with self.assertRaisesRegex(AttestationError, "must not use symlinks"):
            verify_repository(root)

    def test_executable_mode_drift_is_rejected(self) -> None:
        temporary, root = self._temporary_snapshot()
        self.addCleanup(temporary.cleanup)
        path = root / "finetune.py"
        path.chmod(0o755)

        with self.assertRaisesRegex(AttestationError, "Git file mode changed"):
            verify_repository(root)

    def test_notice_drift_fails_closed(self) -> None:
        temporary, root = self._temporary_snapshot()
        self.addCleanup(temporary.cleanup)
        path = root / "third_party/deepseek-moe/LICENSE-CODE"
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)

        with self.assertRaisesRegex(AttestationError, "notice SHA-256 changed"):
            verify_repository(root)

    def test_oversized_legacy_file_is_rejected_before_content_read(self) -> None:
        temporary, root = self._temporary_snapshot()
        self.addCleanup(temporary.cleanup)
        path = root / "finetune.py"
        with path.open("wb") as stream:
            stream.truncate(8668)

        with self.assertRaisesRegex(AttestationError, "8667-byte read limit"):
            verify_repository(root)

    def test_oversized_notice_is_rejected_before_content_read(self) -> None:
        temporary, root = self._temporary_snapshot()
        self.addCleanup(temporary.cleanup)
        path = root / "third_party/deepseek-moe/LICENSE-CODE"
        with path.open("wb") as stream:
            stream.truncate(1066)

        with self.assertRaisesRegex(AttestationError, "1065-byte read limit"):
            verify_repository(root)
