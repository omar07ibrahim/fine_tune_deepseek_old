from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from token_boundary import analyze_boundary
from token_boundary.cli import (
    ERROR_KIND,
    ERROR_SCHEMA_VERSION,
    BoundaryCLIError,
    _emit_error,
    _error_payload,
    _read_stdin,
    _stdin_is_fifo,
    _write_all,
    main,
)
from token_boundary.contract import (
    MAX_DOCUMENT_BYTES,
    BoundaryContractError,
    parse_boundary_case,
)
from token_boundary.errors import BoundaryEngineError
from token_boundary.report import (
    BoundaryReportError,
    canonical_boundary_report_bytes,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPOSITORY_ROOT / "fixtures/token-boundary"
CLI_PATH = REPOSITORY_ROOT / "token_boundary/cli.py"
TOKENIZERS_AVAILABLE = importlib.util.find_spec("tokenizers") is not None


class TokenBoundaryCLIProcessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)

    def _run(
        self,
        payload: bytes,
        *arguments: str,
    ) -> subprocess.CompletedProcess[bytes]:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONHASHSEED"] = "0"
        environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
        return subprocess.run(
            [sys.executable, "-m", "token_boundary.cli", *arguments],
            cwd=self.temporary.name,
            env=environment,
            input=payload,
            capture_output=True,
            check=False,
            timeout=10,
        )

    def _error_code(self, completed: subprocess.CompletedProcess[bytes]) -> str:
        document = json.loads(completed.stderr)
        self.assertEqual(
            set(document),
            {"error", "kind", "schema_version", "status"},
        )
        self.assertEqual(document["kind"], ERROR_KIND)
        self.assertEqual(document["schema_version"], ERROR_SCHEMA_VERSION)
        self.assertEqual(document["status"], "error")
        self.assertEqual(set(document["error"]), {"code"})
        self.assertTrue(completed.stderr.endswith(b"\n"))
        self.assertEqual(completed.stderr.count(b"\n"), 1)
        code = document["error"]["code"]
        self.assertIs(type(code), str)
        assert isinstance(code, str)
        return code

    @unittest.skipUnless(
        TOKENIZERS_AVAILABLE,
        "optional tokenizers==0.21.4 runtime is not installed",
    )
    def test_all_fixtures_emit_the_exact_canonical_report(self) -> None:
        for path in sorted(FIXTURE_ROOT.glob("*.json")):
            with self.subTest(path=path.name):
                payload = path.read_bytes()
                case = parse_boundary_case(payload)
                expected = canonical_boundary_report_bytes(analyze_boundary(case))
                completed = self._run(payload)
                report = json.loads(completed.stdout)

                self.assertEqual(completed.stderr, b"")
                self.assertEqual(completed.stdout, expected)
                self.assertEqual(
                    completed.returncode,
                    0 if report["status"] == "pass" else 1,
                )
                self.assertEqual(
                    report["input_identity"]["canonical_sha256"],
                    analyze_boundary(case).input_identity.canonical_sha256,
                )
                self.assertNotIn("source", report["input_identity"])
                self.assertNotIn("target", report["input_identity"])
                self.assertEqual(completed.stdout.count(b"\n"), 1)

    @unittest.skipUnless(
        TOKENIZERS_AVAILABLE,
        "optional tokenizers==0.21.4 runtime is not installed",
    )
    def test_exact_input_limit_is_accepted(self) -> None:
        payload = (FIXTURE_ROOT / "aligned.v1.json").read_bytes()
        padded = payload.rstrip() + b" " * (MAX_DOCUMENT_BYTES - len(payload.rstrip()))

        completed = self._run(padded)

        self.assertEqual(len(padded), MAX_DOCUMENT_BYTES)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, b"")
        self.assertEqual(json.loads(completed.stdout)["status"], "pass")

    def test_arguments_are_rejected_without_echo_or_stdin_processing(self) -> None:
        arguments = (
            "--help",
            "private-path-marker.json",
            "https://private-url-marker.invalid/case.json",
        )

        for argument in arguments:
            with self.subTest(argument=argument):
                completed = self._run(
                    (FIXTURE_ROOT / "aligned.v1.json").read_bytes(),
                    argument,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertEqual(completed.stdout, b"")
                self.assertEqual(self._error_code(completed), "cli.arguments")
                self.assertNotIn(argument.encode("ascii"), completed.stderr)

    def test_invalid_stdin_is_rejected_with_closed_error_codes(self) -> None:
        cases = (
            (b"", "document.size"),
            (b"{", "document.json"),
            (b"\xff", "document.utf8"),
            (b" " * (MAX_DOCUMENT_BYTES + 1), "document.size"),
        )

        for payload, code in cases:
            with self.subTest(code=code, size=len(payload)):
                completed = self._run(payload)
                self.assertEqual(completed.returncode, 2)
                self.assertEqual(completed.stdout, b"")
                self.assertEqual(self._error_code(completed), code)

    def test_rejected_private_content_is_not_echoed(self) -> None:
        marker = b"private-input-marker"
        payload = (
            b'{"case_id":"private-case","max_length":8,'
            b'"schema_version":"token-boundary-synthetic-case-v1",'
            b'"source":"c","source":"' + marker + b'","target":"de"}'
        )

        completed = self._run(payload)

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, b"")
        self.assertEqual(self._error_code(completed), "document.json")
        self.assertNotIn(marker, completed.stderr)

    @unittest.skipIf(
        TOKENIZERS_AVAILABLE,
        "base-install behavior requires the optional runtime to be absent",
    )
    def test_base_install_reports_the_missing_optional_runtime(self) -> None:
        payload = (FIXTURE_ROOT / "aligned.v1.json").read_bytes()

        completed = self._run(payload)

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, b"")
        self.assertEqual(
            self._error_code(completed),
            "runtime.unavailable",
        )


class TokenBoundaryCLIIOTests(unittest.TestCase):
    def test_partial_stdin_reads_are_joined_without_an_extra_probe(self) -> None:
        with (
            mock.patch(
                "token_boundary.cli._stdin_is_fifo",
                return_value=False,
            ),
            mock.patch(
                "token_boundary.cli.os.read",
                side_effect=(b"first-", b"\xc3", b"\xa9", b"-second", b""),
            ) as reader,
        ):
            payload = _read_stdin(17)

        self.assertEqual(payload, "first-é-second".encode())
        self.assertEqual(reader.call_count, 5)
        self.assertTrue(
            all(call.args[1] <= 16 * 1024 for call in reader.call_args_list)
        )

    def test_tty_is_rejected_before_reading(self) -> None:
        with (
            mock.patch("token_boundary.cli.os.isatty", return_value=True),
            mock.patch("token_boundary.cli.os.read") as reader,
            self.assertRaises(BoundaryCLIError) as caught,
        ):
            _read_stdin()

        self.assertEqual(caught.exception.code, "io.stdin_tty")
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        reader.assert_not_called()

    def test_read_failure_does_not_retain_private_os_error(self) -> None:
        marker = "private-read-marker"
        with (
            mock.patch(
                "token_boundary.cli._stdin_is_fifo",
                return_value=False,
            ),
            mock.patch(
                "token_boundary.cli.os.read",
                side_effect=OSError(marker),
            ),
            self.assertRaises(BoundaryCLIError) as caught,
        ):
            _read_stdin()

        self.assertEqual(caught.exception.code, "io.stdin_unavailable")
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        self.assertNotIn(marker, repr(caught.exception))

    def test_interrupted_read_is_retried(self) -> None:
        with (
            mock.patch(
                "token_boundary.cli._stdin_is_fifo",
                return_value=False,
            ),
            mock.patch(
                "token_boundary.cli.os.read",
                side_effect=(InterruptedError(), b"payload", b""),
            ) as reader,
        ):
            payload = _read_stdin(17)

        self.assertEqual(payload, b"payload")
        self.assertEqual(reader.call_count, 3)

    def test_stdin_reader_stops_at_the_single_sentinel_byte(self) -> None:
        chunks = (
            b"x" * (16 * 1024),
            b"x" * (16 * 1024),
            b"x",
        )
        with (
            mock.patch(
                "token_boundary.cli._stdin_is_fifo",
                return_value=False,
            ),
            mock.patch(
                "token_boundary.cli.os.read",
                side_effect=chunks,
            ) as reader,
        ):
            payload = _read_stdin(17)

        self.assertEqual(len(payload), MAX_DOCUMENT_BYTES + 1)
        self.assertEqual(reader.call_count, 3)
        self.assertEqual(
            [call.args[1] for call in reader.call_args_list],
            [16 * 1024, 16 * 1024, 1],
        )

    def test_interrupted_fifo_wait_keeps_the_same_read(self) -> None:
        with (
            mock.patch(
                "token_boundary.cli._stdin_is_fifo",
                return_value=True,
            ),
            mock.patch(
                "token_boundary.cli.select.select",
                side_effect=(
                    InterruptedError(),
                    ([17], [], []),
                    ([17], [], []),
                ),
            ) as waiter,
            mock.patch(
                "token_boundary.cli.os.read",
                side_effect=(b"payload", b""),
            ),
        ):
            payload = _read_stdin(17)

        self.assertEqual(payload, b"payload")
        self.assertEqual(waiter.call_count, 3)

    def test_closed_pipe_is_read_and_stalled_pipe_times_out(self) -> None:
        read_descriptor, write_descriptor = os.pipe()
        self.addCleanup(os.close, read_descriptor)
        os.write(write_descriptor, b"first-")
        os.write(write_descriptor, b"second")
        os.close(write_descriptor)

        self.assertEqual(
            _read_stdin(read_descriptor),
            b"first-second",
        )

        stalled_read, stalled_write = os.pipe()
        self.addCleanup(os.close, stalled_read)
        self.addCleanup(os.close, stalled_write)
        with (
            mock.patch(
                "token_boundary.cli.READ_TIMEOUT_SECONDS",
                0.02,
            ),
            self.assertRaises(BoundaryCLIError) as caught,
        ):
            _read_stdin(stalled_read)
        self.assertEqual(caught.exception.code, "io.stdin_timeout")

    def test_directory_stdin_shape_is_rejected(self) -> None:
        descriptor = os.open(
            REPOSITORY_ROOT,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        self.addCleanup(os.close, descriptor)

        with self.assertRaises(BoundaryCLIError) as caught:
            _stdin_is_fifo(descriptor)

        self.assertEqual(caught.exception.code, "io.stdin_shape")

    def test_write_all_retries_short_writes(self) -> None:
        observed: list[bytes] = []

        def short_write(_: int, view: memoryview) -> int:
            current = bytes(view)
            observed.append(current)
            return min(2, len(current))

        with mock.patch(
            "token_boundary.cli.os.write",
            side_effect=short_write,
        ):
            _write_all(19, b"abcdef", "io.synthetic")

        self.assertEqual(observed, [b"abcdef", b"cdef", b"ef"])

    def test_broken_pipe_is_redacted(self) -> None:
        marker = "private-broken-pipe-marker"
        with (
            mock.patch(
                "token_boundary.cli.os.write",
                side_effect=BrokenPipeError(marker),
            ),
            self.assertRaises(BoundaryCLIError) as caught,
        ):
            _write_all(1, b"report\n", "io.stdout_unavailable")

        self.assertEqual(caught.exception.code, "io.stdout_unavailable")
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        self.assertNotIn(marker, repr(caught.exception))

        with (
            mock.patch("token_boundary.cli.os.write", return_value=0),
            self.assertRaises(BoundaryCLIError) as zero_write,
        ):
            _write_all(1, b"report\n", "io.stdout_unavailable")
        self.assertEqual(zero_write.exception.code, "io.stdout_unavailable")

    def test_error_payload_is_canonical_ascii(self) -> None:
        payload = _error_payload("runtime.unavailable")

        self.assertEqual(
            payload,
            (
                b'{"error":{"code":"runtime.unavailable"},'
                b'"kind":"token-boundary.cli-error",'
                b'"schema_version":"token-boundary-cli-error-v1",'
                b'"status":"error"}\n'
            ),
        )
        self.assertTrue(payload.isascii())

    def test_unexpected_failure_maps_to_internal_error(self) -> None:
        marker = "private-internal-marker"
        with (
            mock.patch(
                "token_boundary.cli._read_stdin",
                side_effect=RuntimeError(marker),
            ),
            mock.patch("token_boundary.cli._emit_error") as emit,
        ):
            return_code = main(())

        self.assertEqual(return_code, 2)
        emit.assert_called_once_with("internal.error")
        self.assertNotIn(marker, repr(emit.call_args))

    def test_only_allowlisted_upstream_codes_reach_stderr(self) -> None:
        cases = (
            (
                BoundaryContractError("document.json"),
                "document.json",
            ),
            (
                BoundaryEngineError("runtime.version"),
                "runtime.version",
            ),
            (
                BoundaryEngineError("private-engine-marker"),
                "internal.error",
            ),
            (
                BoundaryReportError("private-report-marker"),
                "report.invariant",
            ),
            (
                KeyboardInterrupt(),
                "cli.interrupted",
            ),
        )

        for error, expected in cases:
            with (
                self.subTest(expected=expected),
                mock.patch(
                    "token_boundary.cli._read_stdin",
                    side_effect=error,
                ),
                mock.patch("token_boundary.cli._emit_error") as emit,
            ):
                return_code = main(())
                self.assertEqual(return_code, 2)
                emit.assert_called_once_with(expected)

    def test_broken_stderr_is_silent(self) -> None:
        failures = (
            BoundaryCLIError("io.stderr_unavailable"),
            RuntimeError("private-stderr-marker"),
        )
        for failure in failures:
            with (
                self.subTest(type=type(failure).__name__),
                mock.patch(
                    "token_boundary.cli._write_all",
                    side_effect=failure,
                ),
            ):
                _emit_error("internal.error")

    def test_stdout_failure_maps_to_stable_process_error(self) -> None:
        report = mock.Mock(status="pass")
        failures = (
            (
                BoundaryCLIError("io.stdout_unavailable"),
                "io.stdout_unavailable",
            ),
            (
                RuntimeError("private-stdout-marker"),
                "internal.error",
            ),
        )
        for failure, expected in failures:
            with (
                self.subTest(expected=expected),
                mock.patch("token_boundary.cli._read_stdin", return_value=b"{}"),
                mock.patch(
                    "token_boundary.cli.parse_boundary_case",
                    return_value=object(),
                ),
                mock.patch(
                    "token_boundary.cli.analyze_boundary",
                    return_value=report,
                ),
                mock.patch(
                    "token_boundary.cli.canonical_boundary_report_bytes",
                    return_value=b"canonical-report\n",
                ),
                mock.patch(
                    "token_boundary.cli._write_all",
                    side_effect=failure,
                ),
                mock.patch("token_boundary.cli._emit_error") as emit,
            ):
                return_code = main(())

                self.assertEqual(return_code, 2)
                emit.assert_called_once_with(expected)


class TokenBoundaryCLISurfaceTests(unittest.TestCase):
    def test_cli_has_no_path_url_or_network_surface(self) -> None:
        tree = ast.parse(CLI_PATH.read_text("utf-8"), filename=str(CLI_PATH))
        imports: set[str] = set()
        forbidden_calls: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.partition(".")[0] for alias in node.names)
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module is not None
            ):
                imports.add(node.module.partition(".")[0])
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "open":
                    forbidden_calls.append("open")
                elif (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                    and node.func.attr
                    in {
                        "listdir",
                        "open",
                        "readlink",
                        "scandir",
                        "stat",
                        "walk",
                    }
                ):
                    forbidden_calls.append(f"os.{node.func.attr}")

        self.assertLessEqual(imports, sys.stdlib_module_names)
        self.assertTrue(
            imports.isdisjoint(
                {
                    "argparse",
                    "pathlib",
                    "requests",
                    "socket",
                    "urllib",
                }
            )
        )
        self.assertEqual(forbidden_calls, [])

    def test_console_script_is_the_only_new_cli_entrypoint(self) -> None:
        pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text("utf-8")

        self.assertIn(
            'token-boundary-audit = "token_boundary.cli:main"',
            pyproject,
        )
        self.assertEqual(pyproject.count("token-boundary-audit"), 1)


if __name__ == "__main__":
    unittest.main()
