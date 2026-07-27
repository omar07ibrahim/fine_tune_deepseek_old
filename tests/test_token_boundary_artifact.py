from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import ClassVar, cast
from unittest import mock

from token_boundary import artifact

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "token_boundary" / "artifacts"
BUILD_ROOT = REPOSITORY_ROOT / "build"
ARTIFACT_PATH = ARTIFACT_ROOT / artifact.ARTIFACT_FILENAME
MANIFEST_PATH = ARTIFACT_ROOT / artifact.MANIFEST_FILENAME
FORBIDDEN_IMPORT_ROOTS = {
    "datasets",
    "finetune",
    "huggingface_hub",
    "numpy",
    "peft",
    "requests",
    "socket",
    "tokenizers",
    "torch",
    "transformers",
    "urllib",
}


class TokenBoundaryArtifactTests(unittest.TestCase):
    artifact_payload: ClassVar[bytes]
    manifest_payload: ClassVar[bytes]
    artifact_document: ClassVar[dict[str, object]]
    manifest_document: ClassVar[dict[str, object]]

    @classmethod
    def setUpClass(cls) -> None:
        BUILD_ROOT.mkdir(exist_ok=True)
        cls.artifact_payload = ARTIFACT_PATH.read_bytes()
        cls.manifest_payload = MANIFEST_PATH.read_bytes()
        cls.artifact_document = json.loads(cls.artifact_payload)
        cls.manifest_document = json.loads(cls.manifest_payload)

    def _temporary_directory(self) -> tempfile.TemporaryDirectory[str]:
        return tempfile.TemporaryDirectory(
            prefix="token-boundary-artifact-test-",
            dir=BUILD_ROOT,
        )

    def _write_valid_bundle(self, directory: Path) -> None:
        directory.mkdir()
        (directory / artifact.ARTIFACT_FILENAME).write_bytes(self.artifact_payload)
        (directory / artifact.MANIFEST_FILENAME).write_bytes(self.manifest_payload)

    def _assert_rejected(
        self,
        expected_code: str,
        action: Callable[[], object],
    ) -> artifact.ArtifactVerificationError:
        with self.assertRaises(artifact.ArtifactVerificationError) as caught:
            action()
        self.assertEqual(caught.exception.code, expected_code)
        return caught.exception

    def _assert_exception_graph_excludes(
        self,
        error: BaseException,
        forbidden: tuple[str, ...],
    ) -> None:
        stack = [error]
        visited: set[int] = set()
        rendered: list[str] = []
        while stack:
            current = stack.pop()
            if id(current) in visited:
                continue
            visited.add(id(current))
            rendered.extend(
                (
                    type(current).__module__,
                    type(current).__qualname__,
                    str(current),
                    repr(current),
                    repr(current.args),
                )
            )
            if current.__cause__ is not None:
                stack.append(current.__cause__)
            if current.__context__ is not None:
                stack.append(current.__context__)
        graph = "\n".join(rendered)
        for value in forbidden:
            self.assertNotIn(value, graph)

    def test_committed_bytes_have_frozen_independent_identities(self) -> None:
        self.assertEqual(len(self.artifact_payload), 1533)
        self.assertEqual(
            hashlib.sha256(self.artifact_payload).hexdigest(),
            "29508edbb44ce9cbe77cdde972c0919fe3df8ee2ae1270e69545314f2e1f8358",
        )
        self.assertEqual(len(self.manifest_payload), 1092)
        self.assertEqual(
            hashlib.sha256(self.manifest_payload).hexdigest(),
            "2e52353b8dc21c00601c2f9ce925d1f2acb6b738e61a4d03d0732da10a98ef1d",
        )
        self.assertTrue(self.artifact_payload.endswith(b"\n"))
        self.assertTrue(self.manifest_payload.endswith(b"\n"))

    def test_manifest_has_exact_closed_semantics(self) -> None:
        expected = {
            "schema_version": "token-boundary-artifact-manifest-v1",
            "artifact": {
                "byte_count": 1533,
                "format": "huggingface-tokenizers-json-v1",
                "id": "local-boundary-bpe-v1",
                "origin": "locally-authored-synthetic",
                "path": "local-boundary-bpe.v1.json",
                "sha256": (
                    "29508edbb44ce9cbe77cdde972c0919fe3df8ee2ae1270e69545314f2e1f8358"
                ),
            },
            "intent": {
                "bpe_merges": ["a b"],
                "normalizers": [
                    "NFC",
                    "Strip(left=false,right=true)",
                ],
                "post_processor": "prefix-bos-only",
                "pre_tokenizer": "none",
                "truncation": "caller-bounded-right",
            },
            "runtime": {
                "audit_mode": "offline-after-provisioning",
                "package": "tokenizers",
                "provisioning_may_use_network": True,
                "version": "0.21.4",
            },
            "scope": {
                "dataset_loaded": False,
                "deepseek_model_executed": False,
                "deepseek_tokenizer_attested": False,
                "gpu_used": False,
                "inherited_trainer_executed": False,
                "model_or_tokenizer_downloaded_by_audit": False,
                "pretrained_artifact": False,
                "training_or_forward_pass_executed": False,
            },
        }

        self.assertEqual(self.manifest_document, expected)
        self.assertEqual(artifact.TRUSTED_MANIFEST, expected)

    def test_artifact_has_exact_authored_pipeline_semantics(self) -> None:
        expected_vocab = {
            "<unk>": 0,
            "<bos>": 1,
            " ": 2,
            "a": 3,
            "b": 4,
            "c": 5,
            "d": 6,
            "e": 7,
            "f": 8,
            "g": 9,
            "h": 10,
            "i": 11,
            "x": 12,
            "é": 13,
            "ab": 14,
        }
        document = self.artifact_document

        self.assertEqual(
            set(document),
            {
                "version",
                "truncation",
                "padding",
                "added_tokens",
                "normalizer",
                "pre_tokenizer",
                "post_processor",
                "decoder",
                "model",
            },
        )
        self.assertEqual(document["version"], "1.0")
        self.assertIsNone(document["truncation"])
        self.assertIsNone(document["padding"])
        self.assertEqual(document["added_tokens"], [])
        self.assertIsNone(document["pre_tokenizer"])
        self.assertIsNone(document["decoder"])
        self.assertEqual(
            document["normalizer"],
            {
                "type": "Sequence",
                "normalizers": [
                    {"type": "NFC"},
                    {
                        "type": "Strip",
                        "strip_left": False,
                        "strip_right": True,
                    },
                ],
            },
        )
        self.assertEqual(
            document["post_processor"],
            {
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                ],
                "pair": [
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"Sequence": {"id": "B", "type_id": 1}},
                ],
                "special_tokens": {
                    "<bos>": {
                        "id": "<bos>",
                        "ids": [1],
                        "tokens": ["<bos>"],
                    }
                },
            },
        )
        self.assertEqual(
            document["model"],
            {
                "type": "BPE",
                "dropout": None,
                "unk_token": "<unk>",
                "continuing_subword_prefix": None,
                "end_of_word_suffix": None,
                "fuse_unk": False,
                "byte_fallback": False,
                "ignore_merges": False,
                "vocab": expected_vocab,
                "merges": [["a", "b"]],
            },
        )
        artifact._validate_artifact_document(copy.deepcopy(document))

    def test_verifier_returns_only_the_frozen_local_artifact(self) -> None:
        verified = artifact.verify_local_artifact()

        self.assertEqual(
            verified,
            artifact.VerifiedArtifact(
                artifact_id="local-boundary-bpe-v1",
                filename="local-boundary-bpe.v1.json",
                artifact_format="huggingface-tokenizers-json-v1",
                runtime_package="tokenizers",
                runtime_version="0.21.4",
                sha256=(
                    "29508edbb44ce9cbe77cdde972c0919fe3df8ee2ae1270e69545314f2e1f8358"
                ),
                byte_count=1533,
                payload=self.artifact_payload,
            ),
        )
        self.assertIsInstance(verified.payload, bytes)

    def test_external_manifest_cannot_self_bless_mutated_artifact(self) -> None:
        mutated = self.artifact_payload.replace(
            b'"ab": 14',
            b'"ac": 14',
        )
        self.assertEqual(len(mutated), len(self.artifact_payload))
        self.assertNotEqual(mutated, self.artifact_payload)
        self_blessing = copy.deepcopy(self.manifest_document)
        manifest_artifact = cast(
            dict[str, object],
            self_blessing["artifact"],
        )
        manifest_artifact["sha256"] = hashlib.sha256(mutated).hexdigest()
        self_blessing_payload = (
            json.dumps(
                self_blessing,
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")

        with self._temporary_directory() as temporary_name:
            directory = Path(temporary_name) / "bundle"
            directory.mkdir()
            (directory / artifact.ARTIFACT_FILENAME).write_bytes(mutated)
            (directory / artifact.MANIFEST_FILENAME).write_bytes(self_blessing_payload)

            self._assert_rejected(
                "artifact.identity",
                lambda: artifact.verify_local_artifact(directory),
            )

    def test_manifest_and_artifact_byte_mutations_are_rejected(self) -> None:
        cases = (
            (
                artifact.ARTIFACT_FILENAME,
                b'"ab": 14',
                b'"ac": 14',
                "artifact.identity",
            ),
            (
                artifact.MANIFEST_FILENAME,
                b'"version": "0.21.4"',
                b'"version": "0.21.5"',
                "manifest.identity",
            ),
        )
        for filename, old, new, code in cases:
            with (
                self.subTest(filename=filename),
                self._temporary_directory() as temporary_name,
            ):
                directory = Path(temporary_name) / "bundle"
                self._write_valid_bundle(directory)
                path = directory / filename
                payload = path.read_bytes()
                self.assertIn(old, payload)
                path.write_bytes(payload.replace(old, new))

                self._assert_rejected(
                    code,
                    partial(artifact.verify_local_artifact, directory),
                )

    def test_semantic_validator_rejects_pipeline_mutations(self) -> None:
        cases: tuple[tuple[str, tuple[str, ...], object], ...] = (
            ("artifact.pipeline", ("truncation",), {"max_length": 8}),
            (
                "artifact.normalizer",
                ("normalizer", "normalizers"),
                [{"type": "NFC"}],
            ),
            ("artifact.model", ("model", "merges"), []),
            (
                "artifact.processor",
                ("post_processor", "single"),
                [{"Sequence": {"id": "A", "type_id": 0}}],
            ),
        )
        for code, keys, value in cases:
            with self.subTest(code=code):
                mutated = copy.deepcopy(self.artifact_document)
                target: dict[str, object] = mutated
                for key in keys[:-1]:
                    target = cast(dict[str, object], target[key])
                target[keys[-1]] = value

                self._assert_rejected(
                    code,
                    partial(artifact._validate_artifact_document, mutated),
                )

    def test_private_decoder_rejects_duplicate_and_invalid_json(self) -> None:
        payloads = (
            b'{"same":1,"same":2}',
            b'{"value":NaN}',
            b'{"unterminated":',
            b"\xff",
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                error = self._assert_rejected(
                    "fixture.invalid",
                    partial(
                        artifact._decode_json,
                        payload,
                        "fixture.invalid",
                    ),
                )
                self._assert_exception_graph_excludes(
                    error,
                    ("same", "NaN", "unterminated"),
                )

    def test_directory_argument_is_absolute_exact_path_only(self) -> None:
        invalid_directories: tuple[object, ...] = (
            Path("relative"),
            str(ARTIFACT_ROOT),
            None,
        )
        for value in invalid_directories:
            with self.subTest(value=value):
                self._assert_rejected(
                    "directory.value",
                    partial(
                        artifact.verify_local_artifact,
                        value,  # type: ignore[arg-type]
                    ),
                )

    def test_directory_and_file_symlinks_are_rejected(self) -> None:
        with self._temporary_directory() as temporary_name:
            root = Path(temporary_name)
            real_directory = root / "real"
            self._write_valid_bundle(real_directory)
            directory_link = root / "directory-link"
            directory_link.symlink_to(real_directory, target_is_directory=True)

            self._assert_rejected(
                "directory.shape",
                lambda: artifact.verify_local_artifact(directory_link),
            )

        for filename in (
            artifact.ARTIFACT_FILENAME,
            artifact.MANIFEST_FILENAME,
        ):
            with (
                self.subTest(filename=filename),
                self._temporary_directory() as temporary_name,
            ):
                root = Path(temporary_name)
                directory = root / "bundle"
                self._write_valid_bundle(directory)
                path = directory / filename
                target = root / f"target-{filename}"
                target.write_bytes(path.read_bytes())
                path.unlink()
                path.symlink_to(target)

                self._assert_rejected(
                    "file.io",
                    partial(artifact.verify_local_artifact, directory),
                )

    def test_hardlinked_fixed_files_are_rejected(self) -> None:
        for filename, payload in (
            (artifact.ARTIFACT_FILENAME, self.artifact_payload),
            (artifact.MANIFEST_FILENAME, self.manifest_payload),
        ):
            with (
                self.subTest(filename=filename),
                self._temporary_directory() as temporary_name,
            ):
                root = Path(temporary_name)
                directory = root / "bundle"
                self._write_valid_bundle(directory)
                path = directory / filename
                seed = root / f"seed-{filename}"
                seed.write_bytes(payload)
                path.unlink()
                os.link(seed, path)
                self.assertEqual(path.stat().st_nlink, 2)

                self._assert_rejected(
                    "file.shape",
                    partial(artifact.verify_local_artifact, directory),
                )

    def test_directory_and_fifo_in_fixed_file_slots_are_rejected(self) -> None:
        with self._temporary_directory() as temporary_name:
            directory = Path(temporary_name) / "bundle"
            self._write_valid_bundle(directory)
            path = directory / artifact.ARTIFACT_FILENAME
            path.unlink()
            path.mkdir()

            self._assert_rejected(
                "file.shape",
                lambda: artifact.verify_local_artifact(directory),
            )

        with self._temporary_directory() as temporary_name:
            directory = Path(temporary_name) / "bundle"
            self._write_valid_bundle(directory)
            path = directory / artifact.ARTIFACT_FILENAME
            path.unlink()
            os.mkfifo(path, mode=0o600)

            program = (
                "from pathlib import Path\n"
                "import sys\n"
                "from token_boundary.artifact import "
                "ArtifactVerificationError, verify_local_artifact\n"
                "try:\n"
                "    verify_local_artifact(Path(sys.argv[1]))\n"
                "except ArtifactVerificationError as error:\n"
                "    print(error.code)\n"
                "    raise SystemExit(0)\n"
                "raise SystemExit(1)\n"
            )
            completed = subprocess.run(
                [sys.executable, "-c", program, str(directory)],
                cwd=REPOSITORY_ROOT,
                env={
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONHASHSEED": "0",
                    "PYTHONPATH": str(REPOSITORY_ROOT),
                },
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )

            self.assertEqual(completed.returncode, 0)
            self.assertEqual(completed.stdout, "file.shape\n")
            self.assertEqual(completed.stderr, "")

    def test_empty_and_oversized_files_are_rejected_before_parsing(self) -> None:
        cases = (
            (artifact.ARTIFACT_FILENAME, b""),
            (
                artifact.ARTIFACT_FILENAME,
                b"x" * (artifact.MAX_ARTIFACT_BYTES + 1),
            ),
            (artifact.MANIFEST_FILENAME, b""),
            (
                artifact.MANIFEST_FILENAME,
                b"x" * (artifact.MAX_MANIFEST_BYTES + 1),
            ),
        )
        for filename, payload in cases:
            with (
                self.subTest(filename=filename, size=len(payload)),
                self._temporary_directory() as temporary_name,
            ):
                directory = Path(temporary_name) / "bundle"
                self._write_valid_bundle(directory)
                (directory / filename).write_bytes(payload)

                self._assert_rejected(
                    "file.size",
                    partial(artifact.verify_local_artifact, directory),
                )

    def test_short_read_is_rejected_without_using_partial_content(self) -> None:
        with self._temporary_directory() as temporary_name:
            directory = Path(temporary_name) / "bundle"
            self._write_valid_bundle(directory)
            original_read = os.read
            call_count = 0

            def stop_first_read(descriptor: int, count: int) -> bytes:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return b""
                return original_read(descriptor, count)

            with mock.patch(
                "token_boundary.artifact.os.read",
                side_effect=stop_first_read,
            ):
                self._assert_rejected(
                    "file.read",
                    lambda: artifact.verify_local_artifact(directory),
                )
            self.assertEqual(call_count, 1)

    def test_exception_graphs_do_not_expose_paths_or_rejected_bytes(self) -> None:
        sentinel = "PRIVATE-TOKEN-BOUNDARY-CONTENT-91D72"
        with self._temporary_directory() as temporary_name:
            root = Path(temporary_name)
            directory = root / f"bundle-{sentinel}"
            self._write_valid_bundle(directory)
            rejected = directory / artifact.ARTIFACT_FILENAME
            rejected.write_bytes(sentinel.encode("ascii"))

            error = self._assert_rejected(
                "artifact.identity",
                lambda: artifact.verify_local_artifact(directory),
            )
            self._assert_exception_graph_excludes(
                error,
                (sentinel, str(root), str(directory), str(rejected)),
            )
            self.assertEqual(
                str(error),
                "tokenizer artifact rejected: artifact.identity",
            )

    def test_verification_imports_no_runtime_ml_or_network_modules(self) -> None:
        before = set(sys.modules)
        artifact.verify_local_artifact()
        newly_imported_roots = {
            name.partition(".")[0] for name in set(sys.modules) - before
        }

        self.assertTrue(
            FORBIDDEN_IMPORT_ROOTS.isdisjoint(newly_imported_roots),
            newly_imported_roots,
        )

        source_path = REPOSITORY_ROOT / "token_boundary" / "artifact.py"
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=str(source_path),
        )
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
        self.assertTrue(FORBIDDEN_IMPORT_ROOTS.isdisjoint(imported_roots))

    def test_pyproject_keeps_runtime_optional_and_packages_fixed_data(
        self,
    ) -> None:
        metadata = tomllib.loads(
            (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )

        self.assertEqual(metadata["project"]["dependencies"], [])
        self.assertEqual(
            metadata["project"]["optional-dependencies"],
            {"tokenizer-lab": ["tokenizers==0.21.4"]},
        )
        self.assertEqual(
            metadata["tool"]["setuptools"]["packages"]["find"]["include"],
            ["loss_topology*", "token_boundary*"],
        )
        self.assertEqual(
            metadata["tool"]["setuptools"]["package-data"],
            {"token_boundary": ["artifacts/*.json"]},
        )
        package_files = {
            path.name
            for path in ARTIFACT_ROOT.iterdir()
            if path.is_file() and path.suffix == ".json"
        }
        self.assertEqual(
            package_files,
            {
                artifact.ARTIFACT_FILENAME,
                artifact.MANIFEST_FILENAME,
            },
        )

    def test_temporary_test_directories_are_removed(self) -> None:
        temporary = self._temporary_directory()
        path = Path(temporary.name)
        self.assertTrue(path.is_relative_to(REPOSITORY_ROOT))
        self.assertTrue(path.is_dir())

        temporary.cleanup()

        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
