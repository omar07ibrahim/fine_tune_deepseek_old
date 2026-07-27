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

The engine will create a fresh tokenizer instance for each encoding mode:

1. truncated `source`;
2. untruncated `source + target`;
3. right-truncated `source + target`.

Fresh instances prevent mutable truncation state from leaking between calls.
The legacy cutoff is exactly the length of encoding 1 and is applied to
encoding 3. Encoding 3 must be an exact prefix of encoding 2 by ID, piece,
offset, type ID, and special-token mask; otherwise analysis fails
indeterminate.

## Diagnostics

The report will preserve:

- the three exact encoding vectors;
- the inherited cutoff and supervised positions;
- attributable source, target, cross-boundary, injected, and ambiguous
  positions;
- source leakage and masked retained target positions;
- full, retained, and supervised target-token counts;
- truncated target-token count;
- elimination causes;
- all classification codes plus primary precedence;
- artifact, runtime, input, and attested-source identities;
- explicit false capability and outcome claims.

The NFC example uses the reference provenance replay described in the contract,
not a claim that raw runtime offsets alone expose every combining code point.

## Evidence plan

The later evidence generator will execute the public CLI for all seven
fixtures, bind each canonical report and raw path-free transcript, and render:

1. token lanes with the true authored boundary, inherited cutoff, ownership,
   mask, and exact leak/cross positions;
2. a normalization/merge mechanism view built from the executed encodings;
3. a truncation matrix recomputed over bounded maximum lengths;
4. an accessible SVG rendering of the genuine CLI session.

All artifacts will be synthetic, self-contained, checksum-bound, reproducible,
and inspected at full and narrow browser widths. A GIF is not planned because
the workflow is deterministic and non-interactive; exact lanes and a raw
transcript are more reviewable evidence.
