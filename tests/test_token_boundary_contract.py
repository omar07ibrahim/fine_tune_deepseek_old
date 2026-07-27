from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from token_boundary.contract import (
    MAX_DOCUMENT_BYTES,
    MAX_JSON_DEPTH,
    MAX_JSON_NODES,
    MAX_MODEL_LENGTH,
    MAX_TEXT_BYTES,
    MIN_MODEL_LENGTH,
    SCHEMA_VERSION,
    BoundaryCase,
    BoundaryContractError,
    boundary_case_sha256,
    canonical_boundary_case_bytes,
    parse_boundary_case,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPOSITORY_ROOT / "fixtures/token-boundary"
EXPECTED_FIXTURES = {
    "aligned.v1.json": (
        BoundaryCase("aligned", "c", "de", 8),
        "0420859da2f88bf5cef15f33bef51dfef928d4870fde1fdd6ab353703e6aa103",
    ),
    "merge-cross-boundary.v1.json": (
        BoundaryCase("merge-cross-boundary", "a", "b", 8),
        "9d045174341cdb5836d41590b4733e7844e1977f065a8d9f5a93a1283c888cd2",
    ),
    "nfc-cross-boundary.v1.json": (
        BoundaryCase("nfc-cross-boundary", "e", "\u0301x", 8),
        "c1d9ab41a7e3b0ce5cd6f14fd0b119dd9f6ea47d43ce3477cdd3bbab38ae934c",
    ),
    "normalized-away.v1.json": (
        BoundaryCase("normalized-away", "c", " ", 8),
        "ad0b507c46de21390bd3cf0cf6445b6ca0fa88159055d7f6c4f87adf0807351f",
    ),
    "partial-truncation.v1.json": (
        BoundaryCase("partial-truncation", "cde", "fghi", 6),
        "01d7d2e859e7645ac4ce9d6e9ee65bbe3c1ccebd5f6edab89993965e4a3e8fd7",
    ),
    "right-strip-drift.v1.json": (
        BoundaryCase("right-strip-drift", "c ", "d", 8),
        "d13b14596e7e8f7bfdb968ff7808100a7d2cd4fe4f05dba55f1854c5edf74648",
    ),
    "target-eliminated.v1.json": (
        BoundaryCase("target-eliminated", "cdef", "g", 5),
        "035465571acbbf6647ac7eb15489eac06282e1e2ee09754450a6c2e256dcb50c",
    ),
}


def _document(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "case_id": "contract-test",
        "source": "c",
        "target": "d",
        "max_length": 8,
    }
    value.update(changes)
    return value


def _encode(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("ascii")


class TokenBoundaryContractTests(unittest.TestCase):
    def _assert_rejected(
        self,
        payload: object,
        expected_code: str,
        *,
        forbidden_content: tuple[str, ...] = (),
    ) -> BoundaryContractError:
        with self.assertRaises(BoundaryContractError) as caught:
            parse_boundary_case(payload)  # type: ignore[arg-type]

        exception = caught.exception
        expected_message = f"boundary case rejected: {expected_code}"
        self.assertIs(type(exception), BoundaryContractError)
        self.assertEqual(exception.code, expected_code)
        self.assertEqual(exception.args, (expected_message,))
        self.assertEqual(str(exception), expected_message)
        self.assertEqual(
            repr(exception),
            f"BoundaryContractError({expected_message!r})",
        )
        self.assertIsNone(exception.__cause__)
        self.assertIsNone(exception.__context__)
        for content in forbidden_content:
            self.assertNotIn(content, str(exception))
            self.assertNotIn(content, repr(exception))
            self.assertNotIn(content, exception.code)
        return exception

    def _assert_case_rejected(
        self,
        expected_code: str,
        *args: object,
    ) -> None:
        with self.assertRaises(BoundaryContractError) as caught:
            BoundaryCase(*args)  # type: ignore[arg-type]

        self.assertEqual(caught.exception.code, expected_code)
        self.assertEqual(
            str(caught.exception),
            f"boundary case rejected: {expected_code}",
        )
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)

    def test_contract_limits_are_explicit_and_stable(self) -> None:
        self.assertEqual(SCHEMA_VERSION, "token-boundary-synthetic-case-v1")
        self.assertEqual(MAX_DOCUMENT_BYTES, 32 * 1024)
        self.assertEqual(MAX_TEXT_BYTES, 8 * 1024)
        self.assertEqual(MAX_JSON_DEPTH, 4)
        self.assertEqual(MAX_JSON_NODES, 64)
        self.assertEqual((MIN_MODEL_LENGTH, MAX_MODEL_LENGTH), (2, 512))

    def test_all_seven_authored_fixtures_parse_with_expected_values(self) -> None:
        fixture_names = {path.name for path in FIXTURE_ROOT.glob("*.json")}
        self.assertEqual(fixture_names, set(EXPECTED_FIXTURES))

        for name, (expected_case, _) in EXPECTED_FIXTURES.items():
            with self.subTest(name=name):
                parsed = parse_boundary_case((FIXTURE_ROOT / name).read_bytes())
                self.assertIs(type(parsed), BoundaryCase)
                self.assertEqual(parsed, expected_case)

    def test_every_fixture_has_a_fixed_canonical_identity(self) -> None:
        for name, (expected_case, expected_sha256) in EXPECTED_FIXTURES.items():
            with self.subTest(name=name):
                parsed = parse_boundary_case((FIXTURE_ROOT / name).read_bytes())
                canonical = canonical_boundary_case_bytes(parsed)
                self.assertEqual(
                    canonical,
                    _encode(
                        {
                            "case_id": expected_case.case_id,
                            "max_length": expected_case.max_length,
                            "schema_version": SCHEMA_VERSION,
                            "source": expected_case.source,
                            "target": expected_case.target,
                        }
                    )
                    + b"\n",
                )
                self.assertEqual(boundary_case_sha256(parsed), expected_sha256)
                self.assertEqual(
                    hashlib.sha256(canonical).hexdigest(),
                    expected_sha256,
                )

    def test_canonical_form_is_sorted_ascii_compact_and_newline_terminated(
        self,
    ) -> None:
        case = parse_boundary_case(
            b'{"target":"\\u0301x","source":"e","max_length":8,'
            b'"case_id":"nfc-cross-boundary",'
            b'"schema_version":"token-boundary-synthetic-case-v1"}'
        )

        self.assertEqual(
            canonical_boundary_case_bytes(case),
            (
                b'{"case_id":"nfc-cross-boundary","max_length":8,'
                b'"schema_version":"token-boundary-synthetic-case-v1",'
                b'"source":"e","target":"\\u0301x"}\n'
            ),
        )

    def test_whitespace_and_input_key_order_do_not_change_identity(self) -> None:
        fixture = (FIXTURE_ROOT / "aligned.v1.json").read_bytes()
        reordered = (
            b'{ "target" : "de", "source" : "c", "max_length" : 8, '
            b'"case_id" : "aligned", "schema_version" : '
            b'"token-boundary-synthetic-case-v1" }'
        )

        first = parse_boundary_case(fixture)
        second = parse_boundary_case(reordered)
        self.assertEqual(
            canonical_boundary_case_bytes(first),
            canonical_boundary_case_bytes(second),
        )
        self.assertEqual(boundary_case_sha256(first), boundary_case_sha256(second))

    def test_boundary_case_is_frozen_slotted_and_hashable(self) -> None:
        case = BoundaryCase("immutable", "c", "d", 8)

        self.assertFalse(hasattr(case, "__dict__"))
        self.assertEqual(hash(case), hash(BoundaryCase("immutable", "c", "d", 8)))
        with self.assertRaises(FrozenInstanceError):
            case.source = "changed"  # type: ignore[misc]

    def test_parser_requires_exact_bytes_type(self) -> None:
        payload = _encode(_document())

        class BytesSubclass(bytes):
            pass

        for invalid in (
            payload.decode("ascii"),
            bytearray(payload),
            memoryview(payload),
            BytesSubclass(payload),
            None,
        ):
            with self.subTest(type=type(invalid).__name__):
                self._assert_rejected(invalid, "document.type")

    def test_canonical_and_hash_apis_require_exact_boundary_case_type(
        self,
    ) -> None:
        class BoundaryCaseSubclass(BoundaryCase):
            pass

        subclass = BoundaryCaseSubclass("subclass", "c", "d", 8)
        for invalid in (object(), _document(), subclass):
            with self.subTest(type=type(invalid).__name__):
                with self.assertRaises(BoundaryContractError) as canonical_error:
                    canonical_boundary_case_bytes(invalid)  # type: ignore[arg-type]
                self.assertEqual(canonical_error.exception.code, "case.type")
                self.assertIsNone(canonical_error.exception.__cause__)
                self.assertIsNone(canonical_error.exception.__context__)

                with self.assertRaises(BoundaryContractError) as hash_error:
                    boundary_case_sha256(invalid)  # type: ignore[arg-type]
                self.assertEqual(hash_error.exception.code, "case.type")
                self.assertIsNone(hash_error.exception.__cause__)
                self.assertIsNone(hash_error.exception.__context__)

    def test_empty_and_oversized_documents_are_rejected(self) -> None:
        self._assert_rejected(b"", "document.size")
        self._assert_rejected(b" " * (MAX_DOCUMENT_BYTES + 1), "document.size")

    def test_document_size_limit_is_inclusive(self) -> None:
        compact = _encode(_document())
        exact_limit = compact + b" " * (MAX_DOCUMENT_BYTES - len(compact))

        self.assertEqual(len(exact_limit), MAX_DOCUMENT_BYTES)
        self.assertEqual(
            parse_boundary_case(exact_limit),
            BoundaryCase("contract-test", "c", "d", 8),
        )
        self._assert_rejected(exact_limit + b" ", "document.size")

    def test_invalid_utf8_is_rejected_without_echoing_payload(self) -> None:
        for payload in (b"\xffprivate-marker", b'{"source":"\xe2\x82"}'):
            with self.subTest(payload=payload):
                self._assert_rejected(
                    payload,
                    "document.utf8",
                    forbidden_content=("private-marker", "source"),
                )

    def test_invalid_json_is_rejected_with_one_stable_error(self) -> None:
        for payload in (
            b"{",
            b"[] trailing-private-marker",
            b"\xef\xbb\xbf{}",
            b'{"schema_version":}',
        ):
            with self.subTest(payload=payload):
                self._assert_rejected(
                    payload,
                    "document.json",
                    forbidden_content=("trailing-private-marker",),
                )

    def test_duplicate_keys_are_rejected_before_field_validation(self) -> None:
        payload = (
            b'{"schema_version":"token-boundary-synthetic-case-v1",'
            b'"case_id":"private-first","case_id":"private-second",'
            b'"source":"c","target":"d","max_length":8}'
        )

        self._assert_rejected(
            payload,
            "document.json",
            forbidden_content=("private-first", "private-second", "case_id"),
        )

    def test_floats_and_nonfinite_constants_are_rejected_as_json(self) -> None:
        for raw_value in ("8.0", "8e0", "NaN", "Infinity", "-Infinity"):
            payload = _encode(_document()).replace(b'"max_length":8', b"")
            payload = payload[:-1] + f'"max_length":{raw_value}}}'.encode("ascii")
            with self.subTest(raw_value=raw_value):
                self._assert_rejected(payload, "document.json")

    def test_integer_parser_rejects_more_than_six_digits(self) -> None:
        for raw_value in ("1000000", "-1000000", "999999999999999999999999"):
            payload = _encode(_document(max_length=8)).replace(
                b'"max_length":8',
                f'"max_length":{raw_value}'.encode("ascii"),
            )
            with self.subTest(raw_value=raw_value):
                self._assert_rejected(payload, "document.json")

    def test_root_must_be_an_exact_json_object(self) -> None:
        invalid_roots: tuple[object, ...] = ([], "private-root", None, 8, True)
        for value in invalid_roots:
            with self.subTest(type=type(value).__name__):
                self._assert_rejected(
                    _encode(value),
                    "document.root",
                    forbidden_content=("private-root",),
                )

    def test_root_field_set_is_closed_and_complete(self) -> None:
        missing = _document()
        del missing["target"]
        extra = _document(private_extra="private-value")

        self._assert_rejected(_encode(missing), "document.fields")
        self._assert_rejected(
            _encode(extra),
            "document.fields",
            forbidden_content=("private_extra", "private-value"),
        )

    def test_schema_version_requires_the_exact_string(self) -> None:
        for value in (
            "token-boundary-synthetic-case-v2",
            "",
            1,
            None,
            True,
        ):
            with self.subTest(value=value):
                self._assert_rejected(
                    _encode(_document(schema_version=value)),
                    "schema_version",
                )

    def test_case_id_syntax_and_type_are_closed(self) -> None:
        invalid_values: tuple[object, ...] = (
            "",
            "Uppercase",
            ".leading-dot",
            "slash/not-allowed",
            "a" * 65,
            7,
            True,
            None,
        )
        for value in invalid_values:
            with self.subTest(value=value):
                self._assert_rejected(
                    _encode(_document(case_id=value)),
                    "case_id",
                )

        valid = "a" + "z" * 63
        self.assertEqual(
            parse_boundary_case(_encode(_document(case_id=valid))).case_id,
            valid,
        )

    def test_source_and_target_require_nonempty_exact_strings(self) -> None:
        invalid_texts: tuple[object, ...] = ("", 7, True, None, [], {})
        for field in ("source", "target"):
            for value in invalid_texts:
                with self.subTest(field=field, value=value):
                    self._assert_rejected(
                        _encode(_document(**{field: value})),
                        "text.value",
                    )

    def test_nul_and_surrogate_code_points_are_rejected(self) -> None:
        for field in ("source", "target"):
            for value in ("before\x00after", "\ud800", "a\udfffb"):
                with self.subTest(field=field, value=ascii(value)):
                    self._assert_rejected(
                        _encode(_document(**{field: value})),
                        "text.value",
                        forbidden_content=("before", "after"),
                    )

    def test_text_limit_is_measured_in_utf8_bytes(self) -> None:
        exact_limit = "\u00e9" * (MAX_TEXT_BYTES // 2)
        too_large = exact_limit + "\u00e9"

        for field in ("source", "target"):
            with self.subTest(field=field):
                parsed = parse_boundary_case(_encode(_document(**{field: exact_limit})))
                self.assertEqual(
                    len(getattr(parsed, field).encode("utf-8")),
                    MAX_TEXT_BYTES,
                )
                self._assert_rejected(
                    _encode(_document(**{field: too_large})),
                    "text.size",
                )

    def test_max_length_requires_exact_integer_not_bool(self) -> None:
        invalid_lengths: tuple[object, ...] = (
            True,
            False,
            "8",
            8.0,
            None,
            [],
            {},
        )
        for value in invalid_lengths:
            with self.subTest(value=value):
                payload = (
                    _encode(_document(max_length=value))
                    if not isinstance(value, float)
                    else (
                        b'{"schema_version":"token-boundary-synthetic-case-v1",'
                        b'"case_id":"contract-test","source":"c","target":"d",'
                        b'"max_length":8.0}'
                    )
                )
                expected = (
                    "document.json" if isinstance(value, float) else "max_length.type"
                )
                self._assert_rejected(payload, expected)

    def test_max_length_range_is_inclusive(self) -> None:
        for value in (MIN_MODEL_LENGTH, MAX_MODEL_LENGTH):
            with self.subTest(value=value):
                self.assertEqual(
                    parse_boundary_case(
                        _encode(_document(max_length=value))
                    ).max_length,
                    value,
                )
        for value in (
            MIN_MODEL_LENGTH - 1,
            MAX_MODEL_LENGTH + 1,
            999999,
            -999999,
        ):
            with self.subTest(value=value):
                self._assert_rejected(
                    _encode(_document(max_length=value)),
                    "max_length.range",
                )

    def test_json_depth_limit_is_enforced_before_field_types(self) -> None:
        too_deep: object = [[[[0]]]]

        self._assert_rejected(
            _encode(_document(source=too_deep)),
            "document.depth",
        )

    def test_json_node_limit_is_enforced_before_field_types(self) -> None:
        too_many_nodes = list(range(MAX_JSON_NODES))

        self._assert_rejected(
            _encode(_document(source=too_many_nodes)),
            "document.nodes",
        )

    def test_direct_dataclass_construction_enforces_the_same_field_rules(
        self,
    ) -> None:
        self._assert_case_rejected("case_id", "Uppercase", "c", "d", 8)
        self._assert_case_rejected("text.value", "direct", "", "d", 8)
        self._assert_case_rejected(
            "text.size",
            "direct",
            "a" * (MAX_TEXT_BYTES + 1),
            "d",
            8,
        )
        self._assert_case_rejected(
            "max_length.type",
            "direct",
            "c",
            "d",
            True,
        )
        self._assert_case_rejected(
            "max_length.range",
            "direct",
            "c",
            "d",
            MAX_MODEL_LENGTH + 1,
        )

    def test_rejection_messages_are_content_free_and_code_stable(self) -> None:
        first = self._assert_rejected(
            _encode(_document(case_id="PRIVATE-A")),
            "case_id",
            forbidden_content=("PRIVATE-A",),
        )
        second = self._assert_rejected(
            _encode(_document(case_id="PRIVATE-B")),
            "case_id",
            forbidden_content=("PRIVATE-B",),
        )

        self.assertEqual(first.args, second.args)
        self.assertEqual(first.code, second.code)


if __name__ == "__main__":
    unittest.main()
