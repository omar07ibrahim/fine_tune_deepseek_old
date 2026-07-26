from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import unittest

from loss_topology.contract import (
    ContractError,
    MAX_INPUT_BYTES,
    canonical_trace_bytes,
    parse_synthetic_trace,
    trace_sha256,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HEALTHY_PATH = REPOSITORY_ROOT / "fixtures/synthetic/healthy.v1.json"


def _healthy_value() -> dict[str, object]:
    return json.loads(HEALTHY_PATH.read_text(encoding="utf-8"))


def _encode(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")


class SyntheticTraceContractTests(unittest.TestCase):
    def _assert_value_error(
        self,
        value: object,
        expected_code: str,
    ) -> None:
        with self.assertRaises(ContractError) as caught:
            parse_synthetic_trace(_encode(value))
        self.assertEqual(caught.exception.code, expected_code)
        self.assertEqual(str(caught.exception), expected_code)
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)

    def test_healthy_fixture_parses_to_immutable_tuples(self) -> None:
        trace = parse_synthetic_trace(HEALTHY_PATH.read_bytes())

        self.assertEqual(trace.schema_version, 1)
        self.assertEqual(trace.example_id, "healthy-multiturn")
        self.assertEqual(len(trace.messages), 5)
        self.assertIsInstance(trace.messages, tuple)
        self.assertIsInstance(trace.trace.token_ids, tuple)
        self.assertIsInstance(trace.trace.segments, tuple)
        with self.assertRaises(FrozenInstanceError):
            trace.example_id = "changed"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            trace.messages[0].role = "user"  # type: ignore[misc]

    def test_canonical_bytes_normalize_whitespace_and_key_order(self) -> None:
        source = _healthy_value()
        reordered = {
            "trace": source["trace"],
            "conversation": source["conversation"],
            "example_id": source["example_id"],
            "kind": source["kind"],
            "schema_version": source["schema_version"],
        }
        first = parse_synthetic_trace(HEALTHY_PATH.read_bytes())
        second = parse_synthetic_trace(
            json.dumps(reordered, separators=(",", ":")).encode()
        )

        self.assertEqual(canonical_trace_bytes(first), canonical_trace_bytes(second))
        self.assertEqual(trace_sha256(first), trace_sha256(second))
        self.assertTrue(canonical_trace_bytes(first).endswith(b"\n"))
        self.assertEqual(
            trace_sha256(first),
            "95e1dee0360f6dc45b89cbea63c781d4cc34e3fad02406a86b44145edd00a489",
        )

    def test_duplicate_key_is_rejected_without_echoing_it(self) -> None:
        payload = HEALTHY_PATH.read_text(encoding="utf-8").replace(
            '"schema_version": 1,',
            '"schema_version": 1, "schema_version": 1,',
            1,
        )
        with self.assertRaises(ContractError) as caught:
            parse_synthetic_trace(payload.encode())

        self.assertEqual(caught.exception.code, "json.duplicate_key")
        self.assertNotIn("schema_version", str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)

    def test_parser_failures_retain_no_input_in_exception_context(self) -> None:
        private_json = b'{"private_marker":"never-retain-this-value"'
        cases = (
            (b"\xffprivate-binary-marker", "input.utf8"),
            (private_json, "json.invalid"),
        )

        for payload, expected_code in cases:
            with self.subTest(code=expected_code):
                with self.assertRaises(ContractError) as caught:
                    parse_synthetic_trace(payload)
                self.assertEqual(caught.exception.code, expected_code)
                self.assertIsNone(caught.exception.__cause__)
                self.assertIsNone(caught.exception.__context__)
                self.assertNotIn("private", repr(caught.exception))

    def test_unknown_and_missing_root_fields_fail_closed(self) -> None:
        unknown = _healthy_value()
        unknown["surprise"] = "secret-value"
        self._assert_value_error(unknown, "root.fields")

        missing = _healthy_value()
        del missing["example_id"]
        self._assert_value_error(missing, "root.fields")

    def test_unknown_nested_fields_fail_closed(self) -> None:
        value = _healthy_value()
        value["conversation"]["messages"][0]["name"] = "not-allowed"  # type: ignore[index]
        self._assert_value_error(value, "message.fields")

        value = _healthy_value()
        value["trace"]["segments"][0]["token"] = "not-allowed"  # type: ignore[index]
        self._assert_value_error(value, "segment.fields")

    def test_exact_integer_types_reject_booleans(self) -> None:
        value = _healthy_value()
        value["schema_version"] = True
        self._assert_value_error(value, "schema.version")

        value = _healthy_value()
        value["trace"]["token_ids"][0] = False  # type: ignore[index]
        self._assert_value_error(value, "token_ids.item")

        value = _healthy_value()
        value["trace"]["segments"][0]["start"] = True  # type: ignore[index]
        self._assert_value_error(value, "segment.start")

    def test_floating_non_finite_and_huge_numbers_are_rejected(self) -> None:
        original = HEALTHY_PATH.read_text(encoding="utf-8")
        for replacement, code in (
            ("1.0", "json.floating_point_forbidden"),
            ("NaN", "json.non_finite_number"),
            ("999999999999999999999", "json.integer_out_of_range"),
        ):
            payload = original.replace('"schema_version": 1', f'"schema_version": {replacement}', 1)
            with self.subTest(replacement=replacement):
                with self.assertRaises(ContractError) as caught:
                    parse_synthetic_trace(payload.encode())
                self.assertEqual(caught.exception.code, code)

    def test_invalid_utf8_and_oversized_input_are_rejected(self) -> None:
        class BytesSubclass(bytes):
            pass

        with self.assertRaises(ContractError) as caught:
            parse_synthetic_trace("{}")  # type: ignore[arg-type]
        self.assertEqual(caught.exception.code, "input.type")

        with self.assertRaises(ContractError) as caught:
            parse_synthetic_trace(BytesSubclass(b"{}"))
        self.assertEqual(caught.exception.code, "input.type")

        with self.assertRaises(ContractError) as caught:
            parse_synthetic_trace(b"\xff")
        self.assertEqual(caught.exception.code, "input.utf8")

        with self.assertRaises(ContractError) as caught:
            parse_synthetic_trace(b" " * (MAX_INPUT_BYTES + 1))
        self.assertEqual(caught.exception.code, "input.byte_limit")

    def test_depth_and_node_limits_are_enforced(self) -> None:
        with self.assertRaises(ContractError) as caught:
            parse_synthetic_trace((b"[" * 9) + b"0" + (b"]" * 9))
        self.assertEqual(caught.exception.code, "json.depth_limit")

        payload = _encode([0] * 20_001)
        with self.assertRaises(ContractError) as caught:
            parse_synthetic_trace(payload)
        self.assertEqual(caught.exception.code, "json.node_limit")

    def test_schema_identity_and_example_identifier_are_exact(self) -> None:
        value = _healthy_value()
        value["schema_version"] = 2
        self._assert_value_error(value, "schema.version")

        value = _healthy_value()
        value["kind"] = "real-tokenizer-output"
        self._assert_value_error(value, "schema.kind")

        for example_id in ("Uppercase", "../escape", "", "x" * 65):
            value = _healthy_value()
            value["example_id"] = example_id
            with self.subTest(example_id=example_id):
                self._assert_value_error(value, "example_id")

    def test_message_count_and_turn_order_are_bounded(self) -> None:
        value = _healthy_value()
        value["conversation"]["messages"] = []  # type: ignore[index]
        self._assert_value_error(value, "messages.count")

        value = _healthy_value()
        value["conversation"]["messages"] = [  # type: ignore[index]
            {"role": "user" if index % 2 == 0 else "assistant", "content": "x"}
            for index in range(66)
        ]
        self._assert_value_error(value, "messages.count")

        invalid_sequences = (
            [{"role": "system", "content": "alone"}],
            [
                {"role": "assistant", "content": "first"},
                {"role": "user", "content": "last"},
            ],
            [
                {"role": "user", "content": "one"},
                {"role": "user", "content": "two"},
                {"role": "assistant", "content": "three"},
            ],
            [
                {"role": "user", "content": "one"},
                {"role": "system", "content": "late"},
                {"role": "assistant", "content": "three"},
            ],
        )
        for messages in invalid_sequences:
            value = _healthy_value()
            value["conversation"]["messages"] = messages  # type: ignore[index]
            with self.subTest(messages=messages):
                self._assert_value_error(value, "messages.turn_order")

    def test_message_role_content_and_content_limits_are_strict(self) -> None:
        value = _healthy_value()
        value["conversation"]["messages"][0]["role"] = "tool"  # type: ignore[index]
        self._assert_value_error(value, "message.role")

        value = _healthy_value()
        value["conversation"]["messages"][0]["content"] = 3  # type: ignore[index]
        self._assert_value_error(value, "message.content_type")

        value = _healthy_value()
        value["conversation"]["messages"][0]["content"] = "safe\x00unsafe"  # type: ignore[index]
        self._assert_value_error(value, "message.content_nul")

        payload = HEALTHY_PATH.read_text(encoding="utf-8").replace(
            "Answer with one synthetic color.",
            r"\ud800",
            1,
        )
        with self.assertRaises(ContractError) as caught:
            parse_synthetic_trace(payload.encode())
        self.assertEqual(caught.exception.code, "message.content_unicode")
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)

        value = _healthy_value()
        value["conversation"]["messages"][0]["content"] = "é" * 4_097  # type: ignore[index]
        self._assert_value_error(value, "message.content_limit")

        value = _healthy_value()
        value["conversation"]["messages"] = [  # type: ignore[index]
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": "x" * 1_025,
            }
            for index in range(64)
        ]
        self._assert_value_error(value, "messages.content_limit")

    def test_token_ids_are_nonempty_bounded_exact_integers(self) -> None:
        value = _healthy_value()
        value["trace"]["token_ids"] = []  # type: ignore[index]
        self._assert_value_error(value, "token_ids.count")

        value = _healthy_value()
        value["trace"]["token_ids"] = list(range(4_097))  # type: ignore[index]
        self._assert_value_error(value, "token_ids.count")

        value = _healthy_value()
        value["trace"]["token_ids"][0] = -1  # type: ignore[index]
        self._assert_value_error(value, "token_ids.item")

        value = _healthy_value()
        value["trace"]["padding_token_id"] = -1  # type: ignore[index]
        self._assert_value_error(value, "padding_token_id")

    def test_segment_count_sequence_and_message_indices_are_exact(self) -> None:
        value = _healthy_value()
        value["trace"]["segments"].pop()  # type: ignore[index]
        self._assert_value_error(value, "segments.count")

        value = _healthy_value()
        value["trace"]["segments"][1]["kind"] = "message_content"  # type: ignore[index]
        self._assert_value_error(value, "segments.sequence")

        value = _healthy_value()
        value["trace"]["segments"][1]["message_index"] = 1  # type: ignore[index]
        self._assert_value_error(value, "segments.sequence")

        value = _healthy_value()
        value["trace"]["segments"][0]["message_index"] = 0  # type: ignore[index]
        self._assert_value_error(value, "segments.sequence")

    def test_segments_must_be_contiguous_in_range_and_cover_tokens(self) -> None:
        value = _healthy_value()
        value["trace"]["segments"][1]["start"] = 2  # type: ignore[index]
        self._assert_value_error(value, "segments.coverage")

        value = _healthy_value()
        value["trace"]["segments"][0]["end"] = 34  # type: ignore[index]
        self._assert_value_error(value, "segments.range")

        value = _healthy_value()
        value["trace"]["segments"][-1]["end"] = 32  # type: ignore[index]
        self._assert_value_error(value, "segments.coverage")

    def test_message_boundaries_must_have_tokens_but_content_may_be_empty(self) -> None:
        value = _healthy_value()
        value["trace"]["segments"][1]["end"] = 1  # type: ignore[index]
        value["trace"]["segments"][2]["start"] = 1  # type: ignore[index]
        self._assert_value_error(value, "segments.boundary_empty")

        empty = parse_synthetic_trace(
            (
                REPOSITORY_ROOT
                / "fixtures/synthetic/empty-assistant.v1.json"
            ).read_bytes()
        )
        assistant_content = next(
            segment
            for segment in empty.trace.segments
            if segment.kind == "message_content" and segment.message_index == 2
        )
        self.assertEqual(assistant_content.length, 0)

    def test_padding_span_requires_the_declared_id(self) -> None:
        value = _healthy_value()
        value["trace"]["token_ids"][-1] = 7  # type: ignore[index]
        self._assert_value_error(value, "trace.padding_mismatch")

        value = _healthy_value()
        value["trace"]["token_ids"][3] = 0  # type: ignore[index]
        trace = parse_synthetic_trace(_encode(value))
        self.assertEqual(trace.trace.token_ids[3], 0)


if __name__ == "__main__":
    unittest.main()
