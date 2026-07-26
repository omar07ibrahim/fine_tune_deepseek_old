from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from loss_topology.cli import IOBoundaryError, _atomic_write, _read_input


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HEALTHY_PATH = REPOSITORY_ROOT / "fixtures/synthetic/healthy.v1.json"
EMPTY_PATH = REPOSITORY_ROOT / "fixtures/synthetic/empty-assistant.v1.json"


class LossTopologyCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "input").mkdir()
        (self.root / "output").mkdir()
        shutil.copyfile(HEALTHY_PATH, self.root / "input/healthy.json")
        shutil.copyfile(EMPTY_PATH, self.root / "input/empty.json")

    def _run(
        self,
        *,
        input_path: str = "input/healthy.json",
        output_path: str = "output/audit.json",
    ) -> subprocess.CompletedProcess[bytes]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "loss_topology.cli",
                "--input",
                input_path,
                "--output",
                output_path,
            ],
            cwd=self.root,
            env=environment,
            capture_output=True,
            check=False,
            timeout=10,
        )

    def _stderr_code(
        self,
        result: subprocess.CompletedProcess[bytes],
    ) -> str:
        payload = json.loads(result.stderr)
        self.assertEqual(payload["kind"], "loss-topology.cli-error")
        self.assertEqual(payload["status"], "error")
        return payload["error"]["code"]

    def test_healthy_cli_writes_canonical_atomic_report(self) -> None:
        result = self._run()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, b"")
        output = (self.root / "output/audit.json").read_bytes()
        decoded = json.loads(output)
        summary = json.loads(result.stdout)
        self.assertEqual(decoded["status"], "pass")
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(
            summary["output_sha256"],
            hashlib.sha256(output).hexdigest(),
        )
        self.assertTrue(output.endswith(b"\n"))
        self.assertEqual(
            stat.S_IMODE((self.root / "output/audit.json").stat().st_mode),
            0o644,
        )
        self.assertEqual(
            list((self.root / "output").glob(".loss-topology-*.tmp")),
            [],
        )

    def test_valid_empty_target_writes_fail_report_and_returns_one(self) -> None:
        result = self._run(input_path="input/empty.json")

        self.assertEqual(result.returncode, 1, result.stderr)
        decoded = json.loads((self.root / "output/audit.json").read_bytes())
        self.assertEqual(decoded["status"], "fail")
        self.assertEqual(
            decoded["diagnostics"]["issue_codes"],
            ["assistant_target.empty"],
        )
        self.assertEqual(json.loads(result.stdout)["status"], "fail")

    def test_contract_error_does_not_create_output(self) -> None:
        source = (self.root / "input/healthy.json").read_text(encoding="utf-8")
        (self.root / "input/healthy.json").write_text(
            source.replace(
                '"schema_version": 1,',
                '"schema_version": 1, "schema_version": 1,',
                1,
            ),
            encoding="utf-8",
        )
        result = self._run()

        self.assertEqual(result.returncode, 2)
        self.assertEqual(self._stderr_code(result), "json.duplicate_key")
        self.assertFalse((self.root / "output/audit.json").exists())
        self.assertEqual(result.stdout, b"")

    def test_argument_errors_are_stable_and_do_not_echo_values(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "loss_topology.cli",
                "--unknown-sensitive-value",
            ],
            cwd=self.root,
            env=environment,
            capture_output=True,
            check=False,
            timeout=10,
        )

        self.assertEqual(result.returncode, 2)
        self.assertEqual(self._stderr_code(result), "cli.arguments")
        self.assertNotIn(b"unknown-sensitive-value", result.stderr)

    def test_invalid_input_preserves_existing_output_byte_for_byte(self) -> None:
        sentinel = b"do-not-replace\n"
        (self.root / "output/audit.json").write_bytes(sentinel)
        (self.root / "input/healthy.json").write_bytes(b"{")
        result = self._run()

        self.assertEqual(result.returncode, 2)
        self.assertEqual(self._stderr_code(result), "json.invalid")
        self.assertEqual(
            (self.root / "output/audit.json").read_bytes(),
            sentinel,
        )

    def test_absolute_parent_and_alias_paths_are_rejected_and_redacted(self) -> None:
        cases = (
            ("/private/sensitive-name.json", "output/audit.json", "io.input_path"),
            ("../sensitive-name.json", "output/audit.json", "io.input_path"),
            ("input/healthy.json", "../sensitive-name.json", "io.output_path"),
            ("input/healthy.json", "input/healthy.json", "io.paths_alias"),
        )
        original = (self.root / "input/healthy.json").read_bytes()
        for input_path, output_path, code in cases:
            with self.subTest(code=code):
                result = self._run(
                    input_path=input_path,
                    output_path=output_path,
                )
                self.assertEqual(result.returncode, 2)
                self.assertEqual(self._stderr_code(result), code)
                self.assertNotIn(b"sensitive-name", result.stderr)
        self.assertEqual((self.root / "input/healthy.json").read_bytes(), original)

    def test_symlinked_input_and_input_parent_are_rejected(self) -> None:
        (self.root / "input/link.json").symlink_to("healthy.json")
        result = self._run(input_path="input/link.json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self._stderr_code(result), "io.input_unavailable")

        (self.root / "linked-input").symlink_to("input", target_is_directory=True)
        result = self._run(input_path="linked-input/healthy.json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self._stderr_code(result), "io.input_unavailable")

    def test_symlinked_output_is_rejected_without_touching_target(self) -> None:
        target = self.root / "target.txt"
        target.write_bytes(b"sentinel\n")
        (self.root / "output/audit.json").symlink_to("../target.txt")
        result = self._run()

        self.assertEqual(result.returncode, 2)
        self.assertEqual(self._stderr_code(result), "io.output_not_regular")
        self.assertEqual(target.read_bytes(), b"sentinel\n")
        self.assertTrue((self.root / "output/audit.json").is_symlink())

    def test_symlinked_or_missing_output_parent_is_rejected(self) -> None:
        (self.root / "linked-output").symlink_to(
            "output",
            target_is_directory=True,
        )
        result = self._run(output_path="linked-output/audit.json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(
            self._stderr_code(result),
            "io.output_parent_unavailable",
        )

        result = self._run(output_path="missing/audit.json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(
            self._stderr_code(result),
            "io.output_parent_unavailable",
        )

    def test_output_directory_is_not_replaced(self) -> None:
        (self.root / "output/audit.json").mkdir()
        result = self._run()

        self.assertEqual(result.returncode, 2)
        self.assertEqual(self._stderr_code(result), "io.output_not_regular")
        self.assertTrue((self.root / "output/audit.json").is_dir())

    def test_oversized_file_is_rejected_before_json_decode(self) -> None:
        with (self.root / "input/healthy.json").open("wb") as stream:
            stream.truncate((128 * 1024) + 1)
        result = self._run()

        self.assertEqual(result.returncode, 2)
        self.assertEqual(self._stderr_code(result), "input.byte_limit")
        self.assertFalse((self.root / "output/audit.json").exists())

    def test_failed_atomic_replace_removes_temporary_file(self) -> None:
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        with mock.patch(
            "loss_topology.cli.os.replace",
            side_effect=OSError("synthetic failure"),
        ):
            with self.assertRaises(IOBoundaryError) as caught:
                _atomic_write(
                    Path("output/audit.json"),  # type: ignore[arg-type]
                    b"payload\n",
                )

        self.assertEqual(caught.exception.code, "io.output_unavailable")
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        self.assertEqual(
            list((self.root / "output").glob(".loss-topology-*.tmp")),
            [],
        )
        self.assertFalse((self.root / "output/audit.json").exists())

    def test_library_io_failure_retains_no_private_path_context(self) -> None:
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        private_path = PurePosixPath(
            "private-parent-marker/private-input-marker.json"
        )

        with self.assertRaises(IOBoundaryError) as caught:
            _read_input(private_path)

        self.assertEqual(caught.exception.code, "io.input_unavailable")
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        self.assertNotIn("private", repr(caught.exception))


if __name__ == "__main__":
    unittest.main()
