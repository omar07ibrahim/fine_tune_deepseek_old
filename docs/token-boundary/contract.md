# Token-boundary differential contract

Status: the version-1 input and local tokenizer artifacts are specified here.
The executable engine, CLI, reports, and evidence bundle are separate commits.

## Audited heuristic

The quarantined `finetune.py` tokenizes `source + target` and `source`
independently, then masks the first `len(tokenize(source))` positions of the
combined encoding. The exact inherited file is already attested as SHA-256
`5de3316c8cf37edea97e83230fd90bf01092582dc25016c87fdc404aa1024e26`.

That procedure assumes:

```text
T(source + target)[:len(T(source))] == T(source)
```

Prefix invariance is not a general tokenizer guarantee. Normalization,
cross-boundary subword merges, injected special tokens, and independent
truncation can change the relationship between the two encodings.

The lab will reproduce that exact length-based cutoff over a locally authored
synthetic tokenizer. It will not import or execute the inherited trainer.

## Input document

A case is one UTF-8 JSON object with exactly five fields:

```json
{
  "schema_version": "token-boundary-synthetic-case-v1",
  "case_id": "merge-cross-boundary",
  "source": "a",
  "target": "b",
  "max_length": 8
}
```

The parser rejects duplicate, missing, and unknown fields; invalid UTF-8;
floats; booleans in integer positions; non-finite or oversized numbers; NULs;
lone surrogates; excessive depth or node counts; and inputs larger than
32 KiB. `source` and `target` are non-empty and individually limited to
8 KiB of UTF-8. `max_length` is an exact integer from 2 through 512. `case_id`
matches `[a-z0-9][a-z0-9._-]{0,63}`.

Error codes, messages, and chained exceptions do not echo submitted text,
parser excerpts, or private paths. As with ordinary Python exceptions,
tracebacks and their frame locals are sensitive diagnostic state and should
not be published. Valid cases are immutable and have canonical sorted JSON
bytes and a semantic SHA-256 identity.

## Fixed tokenizer boundary

The only permitted artifact is
`token_boundary/artifacts/local-boundary-bpe.v1.json`:

- origin: locally authored synthetic data;
- artifact SHA-256:
  `29508edbb44ce9cbe77cdde972c0919fe3df8ee2ae1270e69545314f2e1f8358`;
- runtime: exactly `tokenizers==0.21.4`;
- normalizers: NFC followed by right-only strip;
- pre-tokenizer: none;
- model: a 15-entry BPE vocabulary with one declared `a b -> ab` merge;
- post-processor: one injected prefix `<bos>` token;
- truncation: disabled in the artifact and enabled per case in the engine,
  always from the right.

The adjacent manifest is pinned independently in trusted source. Neither a
caller-provided artifact nor a model/tokenizer identifier is accepted.
Serialization is not regenerated during normal verification because library
serialization can vary across runtime versions.

Installing the optional compiled dependency may contact a package index.
Runtime and evidence claims begin only after provisioning and require offline
execution.

## Output classifications

A valid report carries an ordered set of observed classifications and one
primary classification. Primary precedence is:

```text
indeterminate
cross_boundary_token
target_eliminated
partial_target_truncation
boundary_drift
aligned
```

- `aligned`: the inherited cutoff matches the first attributable target token;
  there is no source leakage, masked retained target, cross-boundary token, or
  target truncation.
- `boundary_drift`: the numeric cutoff disagrees with the attributable
  boundary, supervises source-owned positions, or masks retained target-owned
  positions.
- `cross_boundary_token`: one non-special token contains normalized material
  originating on both sides of `source|target`.
- `partial_target_truncation`: at least one but not every attributable target
  token remains after right truncation.
- `target_eliminated`: the full encoding has attributable target tokens but
  none remain supervised after truncation and the inherited cutoff.
- `indeterminate`: offsets, pieces, normalization provenance, prefix
  truncation, unknown tokens, or zero attributable target material do not
  support a safe conclusion.

The full ordered issue list remains visible when more than one mechanism is
present. A primary label never erases secondary diagnostics.

## Ownership oracle

The runtime provides the actual IDs, pieces, special-token mask, and offsets.
The lab additionally replays the artifact's declared NFC and right-strip
normalization with raw-code-point provenance, then aligns non-special token
pieces against that normalized stream.

This reference path exists because a composed character can consume code points
from both sides while a single runtime offset covers only the base character.
If the runtime pieces cannot be reconstructed exactly, an unknown token is
present, truncation is not an exact prefix, or the two views disagree, the
result is `indeterminate`; the lab does not guess ownership.

Reports omit the raw `source` and `target`. They contain only the case identity
and hash, bounded counts, exact synthetic token facts, classifications, and
explicit scope flags.

## Scope

The positive claim is narrow: one executable lab demonstrates concrete failure
mechanisms of the attested length-based heuristic on one hash-pinned, locally
authored tokenizer artifact.

The contract does not claim that:

- a DeepSeek tokenizer or model was used or attested;
- the inherited trainer was executed or fixed;
- a pretrained artifact or dataset was downloaded by the audit;
- a network request, GPU, forward pass, loss, perplexity, training, or
  evaluation was used;
- any mechanism has measured prevalence in real data;
- assistant-only masking is universally correct.
