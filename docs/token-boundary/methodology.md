# Token-boundary counterexample methodology

The version-1 suite contains seven deliberately small authored cases. Each
isolates one property of the fixed local tokenizer or one interaction with the
legacy cutoff.

| Case | Authored boundary | Declared mechanism | Expected primary result |
|---|---|---|---|
| `aligned` | `c|de` | stable baseline with prefix BOS | `aligned` |
| `right-strip-drift` | `c |d` | trailing source space is internal only after concatenation | `boundary_drift` |
| `merge-cross-boundary` | `a|b` | the only BPE merge forms `ab` across the boundary | `cross_boundary_token` |
| `nfc-cross-boundary` | `e|\u0301x` | NFC composes base and target combining mark | `cross_boundary_token` |
| `partial-truncation` | `cde|fghi` | window retains two of four target tokens | `partial_target_truncation` |
| `target-eliminated` | `cdef|g` | BOS plus source fills the complete window | `target_eliminated` |
| `normalized-away` | `c| ` | right-strip removes all target material | `indeterminate` |

The suite is not sampled data and has no denominator beyond these seven
authored counterexamples. It cannot estimate frequency, severity in a
production corpus, model impact, or DeepSeek-specific behavior.

## Three encodings per case

The engine creates a fresh tokenizer instance for each encoding mode:

1. truncated `source`;
2. untruncated `source + target`;
3. right-truncated `source + target`.

Fresh instances prevent mutable truncation state from leaking between calls.
The legacy cutoff is exactly the length of encoding 1 and is applied to
encoding 3. Encoding 3 must be an exact prefix of encoding 2 by ID, piece,
offset, type ID, and special-token mask; otherwise analysis fails
indeterminate.

## Diagnostics

The report preserves:

- the three exact encoding vectors;
- the inherited cutoff and supervised positions;
- attributable source, target, cross-boundary, injected, and ambiguous
  positions;
- source leakage and masked retained target positions;
- full, retained, and supervised target-token counts;
- truncated target-token count;
- elimination causes;
- all classification codes plus primary precedence;
- artifact, runtime, and canonical input identities;
- the exact standalone-source cutoff algorithm identifier; the evidence
  manifest binds that algorithm to the attested inherited source identity;
- explicit false capability and outcome claims.

The NFC example uses the reference provenance replay described in the contract,
not a claim that raw runtime offsets alone expose every combining code point.

## Executed evidence

The evidence generator executes the public CLI for all seven fixtures, binds
each canonical report and the absolute/private-path-free transcript, and
renders:

1. token lanes with the true authored boundary, inherited cutoff, ownership,
   mask, and exact leak/cross positions;
2. a normalization/merge mechanism view built from the executed encodings;
3. a 56-cell truncation matrix recomputed over `max_length` 2 through 9;
4. an accessible SVG rendering of the genuine CLI session.

All artifacts are synthetic, self-contained, checksum-bound, reproducible, and
covered by exact-inventory tests. A GIF is intentionally omitted because the
workflow is deterministic and non-interactive; exact lanes, CSV data, canonical
reports, and the full transcript are more reviewable evidence. See
`docs/token-boundary/evidence.md` for the source allowlist and claim boundary.
