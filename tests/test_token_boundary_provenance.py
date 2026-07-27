from __future__ import annotations

import ast
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from token_boundary.provenance import (
    AMBIGUOUS_PROVENANCE_REASON,
    CLUSTER_REPLAY_REASON,
    MAX_TEXT_BYTES,
    NormalizationTrace,
    ProvenanceError,
    trace_normalization,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class TokenBoundaryProvenanceTests(unittest.TestCase):
    def _assert_rejected(
        self,
        text: object,
        boundary: object,
        expected_code: str,
        *,
        forbidden_content: tuple[str, ...] = (),
    ) -> ProvenanceError:
        with self.assertRaises(ProvenanceError) as caught:
            trace_normalization(
                text,  # type: ignore[arg-type]
                boundary,  # type: ignore[arg-type]
            )

        error = caught.exception
        expected_message = f"normalization input rejected: {expected_code}"
        self.assertIs(type(error), ProvenanceError)
        self.assertEqual(error.code, expected_code)
        self.assertEqual(error.args, (expected_message,))
        self.assertEqual(str(error), expected_message)
        self.assertEqual(
            repr(error),
            f"ProvenanceError({expected_message!r})",
        )
        self.assertIsNone(error.__cause__)
        self.assertIsNone(error.__context__)
        for content in forbidden_content:
            self.assertNotIn(content, str(error))
            self.assertNotIn(content, repr(error))
            self.assertNotIn(content, error.code)
        return error

    def test_ascii_codepoints_retain_exact_individual_origins(self) -> None:
        trace = trace_normalization("abcd", 2)

        self.assertEqual(trace.normalized, "abcd")
        self.assertEqual(trace.origins, ((0,), (1,), (2,), (3,)))
        self.assertEqual(trace.raw_codepoint_count, 4)
        self.assertEqual(trace.nfc_codepoint_count, 4)
        self.assertEqual(trace.trailing_normalized_codepoints_removed, 0)
        self.assertEqual(trace.cross_boundary_output_positions, ())
        self.assertTrue(trace.complete)
        self.assertEqual(trace.indeterminate_reasons, ())

    def test_python_right_strip_removes_only_trailing_whitespace(self) -> None:
        trace = trace_normalization(" a \t\n", 2)

        self.assertEqual(trace.normalized, " a")
        self.assertEqual(trace.origins, ((0,), (1,)))
        self.assertEqual(trace.raw_codepoint_count, 5)
        self.assertEqual(trace.nfc_codepoint_count, 5)
        self.assertEqual(trace.trailing_normalized_codepoints_removed, 3)
        self.assertTrue(trace.complete)

    def test_unicode_whitespace_uses_python_rstrip_semantics(self) -> None:
        trace = trace_normalization("a\u00a0\u2003", 1)

        self.assertEqual(trace.normalized, "a")
        self.assertEqual(trace.origins, ((0,),))
        self.assertEqual(trace.trailing_normalized_codepoints_removed, 2)
        self.assertTrue(trace.complete)

    def test_cross_boundary_nfc_composition_unions_raw_origins(self) -> None:
        trace = trace_normalization("e\u0301x", 1)

        self.assertEqual(trace.normalized, "\u00e9x")
        self.assertEqual(trace.origins, ((0, 1), (2,)))
        self.assertEqual(trace.raw_codepoint_count, 3)
        self.assertEqual(trace.nfc_codepoint_count, 2)
        self.assertEqual(trace.cross_boundary_output_positions, (0,))
        self.assertTrue(trace.complete)
        self.assertEqual(trace.indeterminate_reasons, ())

    def test_composition_on_one_side_is_not_cross_boundary(self) -> None:
        left = trace_normalization("e\u0301x", 2)
        right = trace_normalization("xe\u0301", 1)

        self.assertEqual(left.origins, ((0, 1), (2,)))
        self.assertEqual(right.origins, ((0,), (1, 2)))
        self.assertEqual(left.cross_boundary_output_positions, ())
        self.assertEqual(right.cross_boundary_output_positions, ())

    def test_unchanged_multichar_cluster_maps_individual_characters(self) -> None:
        trace = trace_normalization("a\u0315x", 1)

        self.assertEqual(trace.normalized, "a\u0315x")
        self.assertEqual(trace.origins, ((0,), (1,), (2,)))
        self.assertEqual(trace.cross_boundary_output_positions, ())
        self.assertTrue(trace.complete)

    def test_reordered_or_composed_multioutput_cluster_is_indeterminate(
        self,
    ) -> None:
        trace = trace_normalization("a\u0315\u0300", 1)

        self.assertEqual(trace.normalized, "\u00e0\u0315")
        self.assertEqual(trace.origins, (None, None))
        self.assertFalse(trace.complete)
        self.assertEqual(
            trace.indeterminate_reasons,
            (AMBIGUOUS_PROVENANCE_REASON,),
        )
        self.assertEqual(trace.cross_boundary_output_positions, ())

    def test_single_codepoint_multioutput_normalization_is_indeterminate(
        self,
    ) -> None:
        trace = trace_normalization("\u0344x", 1)

        self.assertEqual(trace.normalized, "\u0308\u0301x")
        self.assertEqual(trace.origins, (None, None, (1,)))
        self.assertFalse(trace.complete)
        self.assertEqual(
            trace.indeterminate_reasons,
            (AMBIGUOUS_PROVENANCE_REASON,),
        )

    def test_hangul_cross_cluster_composition_fails_replay_closed(self) -> None:
        trace = trace_normalization("\u1100\u1161x", 1)

        self.assertEqual(trace.normalized, "\uac00x")
        self.assertEqual(trace.origins, (None, None))
        self.assertEqual(trace.raw_codepoint_count, 3)
        self.assertEqual(trace.nfc_codepoint_count, 2)
        self.assertFalse(trace.complete)
        self.assertEqual(
            trace.indeterminate_reasons,
            (CLUSTER_REPLAY_REASON,),
        )
        self.assertEqual(trace.cross_boundary_output_positions, ())

    def test_leading_combining_mark_is_replayed_without_guessing(self) -> None:
        trace = trace_normalization("\u0301a", 1)

        self.assertEqual(trace.normalized, "\u0301a")
        self.assertEqual(trace.origins, ((0,), (1,)))
        self.assertTrue(trace.complete)
        self.assertEqual(trace.indeterminate_reasons, ())

    def test_boundary_endpoints_never_create_cross_positions(self) -> None:
        for boundary in (0, 2):
            with self.subTest(boundary=boundary):
                trace = trace_normalization("e\u0301", boundary)
                self.assertEqual(trace.cross_boundary_output_positions, ())
                self.assertEqual(trace.origins, ((0, 1),))

    def test_multiple_compositions_report_only_the_spanning_output(self) -> None:
        trace = trace_normalization("e\u0301e\u0301", 3)

        self.assertEqual(trace.normalized, "\u00e9\u00e9")
        self.assertEqual(trace.origins, ((0, 1), (2, 3)))
        self.assertEqual(trace.cross_boundary_output_positions, (1,))

    def test_text_requires_exact_builtin_string_type(self) -> None:
        class StringSubclass(str):
            pass

        for value in (b"abc", bytearray(b"abc"), StringSubclass("abc"), None):
            with self.subTest(type=type(value).__name__):
                self._assert_rejected(value, 1, "text.type")

    def test_text_rejects_empty_nul_and_surrogates(self) -> None:
        cases = (
            ("", ()),
            ("private-before\x00private-after", ("private-before", "private-after")),
            ("\ud800", ()),
            ("a\udfffb", ()),
        )
        for text, forbidden in cases:
            with self.subTest(text=ascii(text)):
                self._assert_rejected(
                    text,
                    0,
                    "text.value",
                    forbidden_content=forbidden,
                )

    def test_utf8_text_size_limit_is_inclusive(self) -> None:
        exact_limit = "\u00e9" * (MAX_TEXT_BYTES // 2)
        too_large = exact_limit + "\u00e9"

        trace = trace_normalization(exact_limit, len(exact_limit))
        self.assertEqual(
            len(trace.normalized.encode("utf-8")),
            MAX_TEXT_BYTES,
        )
        self.assertTrue(trace.complete)
        self._assert_rejected(too_large, 0, "text.size")

    def test_boundary_requires_exact_integer_type(self) -> None:
        for value in (True, False, 1.0, "1", None):
            with self.subTest(value=value):
                self._assert_rejected("ab", value, "boundary.type")

    def test_boundary_range_is_closed_and_inclusive(self) -> None:
        for value in (0, 2):
            with self.subTest(value=value):
                self.assertIsInstance(
                    trace_normalization("ab", value),
                    NormalizationTrace,
                )
        for value in (-1, 3):
            with self.subTest(value=value):
                self._assert_rejected("ab", value, "boundary.range")

    def test_rejection_messages_are_stable_and_content_free(self) -> None:
        first = self._assert_rejected(
            "PRIVATE-A\x00",
            0,
            "text.value",
            forbidden_content=("PRIVATE-A",),
        )
        second = self._assert_rejected(
            "PRIVATE-B\x00",
            0,
            "text.value",
            forbidden_content=("PRIVATE-B",),
        )

        self.assertEqual(first.code, second.code)
        self.assertEqual(first.args, second.args)

    def test_trace_is_frozen_slotted_and_has_exact_builtin_fields(self) -> None:
        trace = trace_normalization("e\u0301x ", 1)

        self.assertIs(type(trace), NormalizationTrace)
        self.assertFalse(hasattr(trace, "__dict__"))
        self.assertIs(type(trace.normalized), str)
        self.assertIs(type(trace.origins), tuple)
        self.assertTrue(
            all(
                origin is None
                or (
                    type(origin) is tuple
                    and all(type(position) is int for position in origin)
                )
                for origin in trace.origins
            )
        )
        self.assertIs(type(trace.raw_codepoint_count), int)
        self.assertIs(type(trace.nfc_codepoint_count), int)
        self.assertIs(
            type(trace.trailing_normalized_codepoints_removed),
            int,
        )
        self.assertIs(type(trace.cross_boundary_output_positions), tuple)
        self.assertIs(type(trace.complete), bool)
        self.assertIs(type(trace.indeterminate_reasons), tuple)
        with self.assertRaises(FrozenInstanceError):
            trace.normalized = "changed"  # type: ignore[misc]

    def test_reason_codes_are_deduplicated_and_sorted(self) -> None:
        trace = trace_normalization("a\u0315\u0300a\u0315\u0300", 3)

        self.assertEqual(
            trace.indeterminate_reasons,
            tuple(sorted(set(trace.indeterminate_reasons))),
        )
        self.assertEqual(
            trace.indeterminate_reasons,
            (AMBIGUOUS_PROVENANCE_REASON,),
        )

    def test_provenance_module_imports_only_standard_library(self) -> None:
        path = REPOSITORY_ROOT / "token_boundary/provenance.py"
        tree = ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
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
        self.assertNotIn("finetune", imported_roots)
        self.assertNotIn("tokenizers", imported_roots)


if __name__ == "__main__":
    unittest.main()
