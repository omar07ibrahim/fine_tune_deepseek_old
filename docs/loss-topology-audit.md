# Loss topology audit v1

The lab deterministically constructs two position-aligned label arrays from a
validated synthetic trace. `-100` is the ignore index.

| Policy | Supervised positions | Boundary behavior |
|---|---|---|
| `all_tokens` | every non-padding token | template and message-boundary tokens are intentionally selected |
| `assistant_only` | only `message_content` spans owned by assistant messages | every boundary, user/system content token, suffix, prefix, and padding token is ignored |

These are selection topologies, not loss values. The lab does not apply a
causal shift, run a forward pass, compute perplexity, or claim that either
policy is universally correct for a model template.

For each policy the report contains:

- exact labels and a canonical label-array SHA-256;
- eligible, supervised, and ignored token counts;
- half-open contiguous supervised runs;
- selected boundary-token count;
- exact positions for boundary leakage, padding leakage, off-policy
  supervision, and missing eligible targets.

The internally generated policies must have no off-policy or missing
positions. The public auditor also accepts a supplied in-memory label tuple so
tests and adapters can prove that leaked boundaries and omitted targets are
detected.

An assistant message whose content span has zero tokens produces
`assistant_target.empty` and a failing audit. Structural input failures write
no report. A valid but diagnostically failing trace writes a complete canonical
report and returns exit status 1.

## CLI boundary

Run from the repository root with safe repository-relative paths:

```bash
python3 -m loss_topology.cli \
  --input fixtures/synthetic/healthy.v1.json \
  --output build/healthy.audit.json
```

The output parent must already exist. Input and output path components cannot
be absolute, empty, `.`, `..`, backslash-based, or symlinks. The input must be
a bounded regular file. Output uses a same-directory temporary regular file,
`fsync`, and atomic replacement. Invalid input leaves an existing output
untouched. Error responses contain only a stable code—never input content,
field values, parser excerpts, or filesystem paths.

The CLI reads no model, dataset, tokenizer, checkpoint, configuration, or
inherited trainer input. Python necessarily imports the lab's own source; the
only data payload it opens is the explicitly supplied synthetic JSON.
