from __future__ import annotations

import ast
from pathlib import Path
import sys
import tomllib
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = REPOSITORY_ROOT / "loss_topology"


class LabIsolationTests(unittest.TestCase):
    def test_original_lab_imports_only_stdlib_or_its_own_package(self) -> None:
        imported_roots: set[str] = set()
        for path in sorted(LAB_ROOT.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_roots.update(
                        alias.name.partition(".")[0] for alias in node.names
                    )
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    if node.module is not None:
                        imported_roots.add(node.module.partition(".")[0])

        self.assertLessEqual(imported_roots, sys.stdlib_module_names)
        self.assertNotIn("finetune", imported_roots)

    def test_package_metadata_declares_no_runtime_dependencies(self) -> None:
        metadata = tomllib.loads(
            (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )

        self.assertEqual(metadata["project"]["dependencies"], [])
        self.assertEqual(metadata["project"]["requires-python"], ">=3.11")

    def test_inherited_sources_are_not_package_modules(self) -> None:
        package_files = {
            path.relative_to(REPOSITORY_ROOT).as_posix()
            for path in LAB_ROOT.glob("*.py")
        }

        self.assertNotIn("finetune.py", package_files)
        self.assertNotIn("configs/ds_config_zero3.json", package_files)


if __name__ == "__main__":
    unittest.main()
