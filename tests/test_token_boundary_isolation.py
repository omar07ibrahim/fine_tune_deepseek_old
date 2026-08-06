from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "token_boundary"
LEGACY_PATH = REPOSITORY_ROOT / "finetune.py"
LEGACY_SHA256 = "5de3316c8cf37edea97e83230fd90bf01092582dc25016c87fdc404aa1024e26"
FORBIDDEN_IMPORT_ROOTS = {
    "datasets",
    "finetune",
    "huggingface_hub",
    "numpy",
    "peft",
    "requests",
    "socket",
    "torch",
    "transformers",
    "urllib",
}


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module is not None
        ):
            roots.add(node.module.partition(".")[0])
    return roots


def _assigned_name(node: ast.Assign) -> str | None:
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        return None
    return node.targets[0].id


class TokenBoundaryIsolationTests(unittest.TestCase):
    def test_only_engine_imports_the_optional_tokenizer_runtime(self) -> None:
        external_by_module = {
            path.name: _import_roots(path) - sys.stdlib_module_names
            for path in sorted(PACKAGE_ROOT.glob("*.py"))
        }

        for module, roots in external_by_module.items():
            with self.subTest(module=module):
                self.assertTrue(FORBIDDEN_IMPORT_ROOTS.isdisjoint(roots))
                if module == "engine.py":
                    self.assertEqual(roots, {"tokenizers"})
                else:
                    self.assertEqual(roots, set())

    def test_package_import_does_not_load_optional_or_ml_modules(self) -> None:
        program = (
            "import json, sys\n"
            "before=set(sys.modules)\n"
            "import token_boundary\n"
            "after=set(sys.modules)-before\n"
            "print(json.dumps(sorted({name.split('.')[0] for name in after})))\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", program],
            cwd=REPOSITORY_ROOT,
            env={
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "PYTHONPATH": str(REPOSITORY_ROOT),
            },
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        imported = set(json.loads(completed.stdout))
        self.assertNotIn("tokenizers", imported)
        self.assertTrue(FORBIDDEN_IMPORT_ROOTS.isdisjoint(imported))

    @unittest.skipIf(
        importlib.util.find_spec("tokenizers") is not None,
        "base-install behavior requires the optional runtime to be absent",
    )
    def test_public_audit_api_fails_cleanly_without_optional_runtime(self) -> None:
        program = (
            "import json, sys\n"
            "from token_boundary import BoundaryCase, BoundaryEngineError, "
            "analyze_boundary\n"
            "try:\n"
            "    analyze_boundary(BoundaryCase('base-install','c','d',8))\n"
            "except BoundaryEngineError as error:\n"
            "    print(json.dumps({'code':error.code,"
            "'cause':error.__cause__ is None,"
            "'context':error.__context__ is None,"
            "'runtime_loaded':'tokenizers' in sys.modules},sort_keys=True))\n"
            "    raise SystemExit(0)\n"
            "raise SystemExit(1)\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", program],
            cwd=REPOSITORY_ROOT,
            env={
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "PYTHONPATH": str(REPOSITORY_ROOT),
            },
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            json.loads(completed.stdout),
            {
                "cause": True,
                "code": "runtime.unavailable",
                "context": True,
                "runtime_loaded": False,
            },
        )

    def test_boundary_lab_contains_no_remote_artifact_surface(self) -> None:
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(PACKAGE_ROOT.glob("*.py"))
        )
        for forbidden in (
            "AutoTokenizer",
            "from_pretrained",
            "hf_hub_download",
            "http://",
            "https://",
            "deepseek-ai/",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_legacy_snapshot_identity_is_still_the_attested_source(self) -> None:
        payload = LEGACY_PATH.read_bytes()
        manifest = json.loads(
            (REPOSITORY_ROOT / "provenance/legacy-snapshot.v1.json").read_bytes()
        )
        entry = next(
            item for item in manifest["files"] if item["path"] == "finetune.py"
        )

        self.assertEqual(hashlib.sha256(payload).hexdigest(), LEGACY_SHA256)
        self.assertEqual(entry["sha256"], LEGACY_SHA256)
        self.assertEqual(entry["byte_length"], len(payload))

    def test_attested_preprocess_builds_concatenated_and_source_encodings(
        self,
    ) -> None:
        tree = ast.parse(
            LEGACY_PATH.read_text(encoding="utf-8"),
            filename=str(LEGACY_PATH),
        )
        preprocess = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "preprocess"
        )
        examples_assignment = next(
            node
            for node in preprocess.body
            if isinstance(node, ast.Assign) and _assigned_name(node) == "examples"
        )
        self.assertIsInstance(examples_assignment.value, ast.ListComp)
        assert isinstance(examples_assignment.value, ast.ListComp)
        self.assertIsInstance(examples_assignment.value.elt, ast.BinOp)
        assert isinstance(examples_assignment.value.elt, ast.BinOp)
        self.assertIsInstance(examples_assignment.value.elt.op, ast.Add)
        left = examples_assignment.value.elt.left
        right = examples_assignment.value.elt.right
        self.assertIsInstance(left, ast.Name)
        self.assertIsInstance(right, ast.Name)
        assert isinstance(left, ast.Name)
        assert isinstance(right, ast.Name)
        self.assertEqual(
            {
                left.id,
                right.id,
            },
            {"s", "t"},
        )

        paired_assignment = next(
            node
            for node in preprocess.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Tuple)
        )
        paired_target = paired_assignment.targets[0]
        assert isinstance(paired_target, ast.Tuple)
        target_names = tuple(
            item.id for item in paired_target.elts if isinstance(item, ast.Name)
        )
        self.assertEqual(
            target_names,
            ("examples_tokenized", "sources_tokenized"),
        )
        self.assertIsInstance(paired_assignment.value, ast.ListComp)
        assert isinstance(paired_assignment.value, ast.ListComp)
        generator = paired_assignment.value.generators[0]
        self.assertIsInstance(generator.iter, ast.Tuple)
        assert isinstance(generator.iter, ast.Tuple)
        self.assertEqual(
            tuple(
                item.id for item in generator.iter.elts if isinstance(item, ast.Name)
            ),
            ("examples", "sources"),
        )

    def test_attested_preprocess_masks_by_standalone_source_length(self) -> None:
        tree = ast.parse(
            LEGACY_PATH.read_text(encoding="utf-8"),
            filename=str(LEGACY_PATH),
        )
        preprocess = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "preprocess"
        )
        loop = next(node for node in preprocess.body if isinstance(node, ast.For))
        self.assertIsInstance(loop.target, ast.Tuple)
        assert isinstance(loop.target, ast.Tuple)
        self.assertEqual(
            tuple(item.id for item in loop.target.elts if isinstance(item, ast.Name)),
            ("label", "source_len"),
        )
        self.assertIsInstance(loop.iter, ast.Call)
        assert isinstance(loop.iter, ast.Call)
        self.assertIsInstance(loop.iter.func, ast.Name)
        assert isinstance(loop.iter.func, ast.Name)
        self.assertEqual(loop.iter.func.id, "zip")
        self.assertEqual(len(loop.body), 1)

        masking = loop.body[0]
        self.assertIsInstance(masking, ast.Assign)
        assert isinstance(masking, ast.Assign)
        self.assertIsInstance(masking.targets[0], ast.Subscript)
        assert isinstance(masking.targets[0], ast.Subscript)
        self.assertIsInstance(masking.targets[0].slice, ast.Slice)
        assert isinstance(masking.targets[0].slice, ast.Slice)
        self.assertIsInstance(masking.targets[0].slice.upper, ast.Name)
        assert isinstance(masking.targets[0].slice.upper, ast.Name)
        self.assertEqual(masking.targets[0].slice.upper.id, "source_len")
        self.assertIsInstance(masking.value, ast.Name)
        assert isinstance(masking.value, ast.Name)
        self.assertEqual(masking.value.id, "IGNORE_INDEX")

    def test_engine_reproduces_the_same_numeric_cutoff_without_trainer_import(
        self,
    ) -> None:
        path = PACKAGE_ROOT / "engine.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
        cutoff = next(
            node for node in assignments if _assigned_name(node) == "legacy_cutoff"
        )

        self.assertIsInstance(cutoff.value, ast.Call)
        assert isinstance(cutoff.value, ast.Call)
        self.assertIsInstance(cutoff.value.func, ast.Name)
        assert isinstance(cutoff.value.func, ast.Name)
        self.assertEqual(cutoff.value.func.id, "len")
        self.assertEqual(cutoff.value.keywords, [])
        self.assertEqual(len(cutoff.value.args), 1)

        ids = cutoff.value.args[0]
        self.assertIsInstance(ids, ast.Attribute)
        assert isinstance(ids, ast.Attribute)
        self.assertEqual(ids.attr, "ids")
        self.assertIsInstance(ids.ctx, ast.Load)
        self.assertIsInstance(ids.value, ast.Attribute)
        assert isinstance(ids.value, ast.Attribute)
        self.assertEqual(ids.value.attr, "snapshot")
        self.assertIsInstance(ids.value.ctx, ast.Load)
        self.assertIsInstance(ids.value.value, ast.Name)
        assert isinstance(ids.value.value, ast.Name)
        self.assertEqual(ids.value.value.id, "source_capture")
        self.assertIsInstance(ids.value.value.ctx, ast.Load)
        self.assertNotIn("finetune", _import_roots(path))


if __name__ == "__main__":
    unittest.main()
