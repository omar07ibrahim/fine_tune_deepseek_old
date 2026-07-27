# Reproducible token-boundary evidence

This bundle is executable evidence for the local synthetic tokenizer lab. It
is not decoration, a benchmark, or evidence about a DeepSeek or pretrained
tokenizer.

## Closed inputs

`tools/render_token_boundary_evidence.py` is hardcoded to seven committed
fixtures:

- `aligned.v1.json`;
- `merge-cross-boundary.v1.json`;
- `nfc-cross-boundary.v1.json`;
- `normalized-away.v1.json`;
- `partial-truncation.v1.json`;
- `right-strip-drift.v1.json`;
- `target-eliminated.v1.json`.

Their raw SHA-256 identities are embedded in the generator. The source
allowlist also binds the complete `token_boundary` implementation, both local
artifact documents, `pyproject.toml`, the generator itself, the quarantined
trainer bytes, and the legacy attestation that identifies those trainer bytes.

The generator accepts only one of `--write` or `--check`. It accepts no
fixture, model, artifact, module, command, output-directory, URL, or network
argument.

## Executed runtime

Every original fixture is copied byte-for-byte into a repository-local
ephemeral sandbox and sent to a fresh subprocess through standard input:

```text
python3 -m token_boundary.cli < input/<fixture>
```

The process runs on exactly `tokenizers==0.21.4` and loads only the committed
1,533-byte local BPE artifact with SHA-256
`29508edbb44ce9cbe77cdde972c0919fe3df8ee2ae1270e69545314f2e1f8358`.
The generator independently rebuilds each report through the public analysis
API and requires byte equality with CLI stdout.

For each execution, the manifest records:

- the relative command and fixture path;
- raw fixture and canonical input identities;
- expected and observed exit, status, and primary classification;
- exact report path, byte count, and SHA-256;
- stderr byte count and SHA-256.

Exit `0` means a valid aligned report. Exit `1` means a valid complete fail or
indeterminate report. An input, runtime, report, or process-boundary error
would use exit `2` and is not accepted as evidence.

## Bounded truncation sweep

The generator also creates 56 new canonical inputs: all seven authored
source/target configurations at `max_length` values 2 through 9. Every cell is
another fresh stdin CLI subprocess, not a hand-edited report.

`truncation-sweep.csv` preserves, for every execution:

- the base fixture and raw hash;
- the max-length override and canonical input hash;
- exit, report status, primary classification, and stdout hash;
- legacy and oracle cutoffs;
- full, retained, supervised, and truncated target-token counts;
- prompt-leakage and cross-boundary positions;
- elimination and indeterminate reasons.

This sweep demonstrates deterministic transitions within a bounded authored
grid. It has no sampling denominator and cannot estimate prevalence, severity,
or model impact.

## Generated inventory

The dedicated `docs/token-boundary/generated` directory contains exactly:

| Artifact | Meaning |
|---|---|
| Seven `*.boundary-report.json` files | Canonical stdout from the seven original fixture executions |
| `cli-session.txt` | Recorder transcript with full canonical stdout between explicit markers |
| `cli-session.svg` | Readable index of the same real commands, exits, classifications, sizes, and report hashes |
| `token-boundary-architecture.svg` | Executable workflow and separate attestation-only trainer boundary |
| `token-lanes.svg` | Full and retained token lanes for all seven cases, including ownership and mask state |
| `boundary-mechanisms.svg` | Executed BPE, NFC, right-strip, and normalized-away mechanisms |
| `truncation-sweep.csv` | Exact 56-row sweep data |
| `truncation-matrix.svg` | Directly labeled visualization of all 56 sweep cells |
| `token-boundary-evidence.v1.json` | Runtime, source, execution, scope, sweep, and artifact manifest |

Every SVG is self-contained. It has an accessible title and description, uses
text and marks in addition to color, and contains no script, linked image,
external font, or remote resource. `SPACE`, `U+0020`, and `U+0301` labels are
display escapes; they do not mutate a report token piece.

The workflow is deterministic and non-interactive, so a GIF would make exact
states harder to inspect. Token lanes, the matrix, the complete transcript,
and canonical JSON reports are the more reviewable evidence.

## Legacy algorithm binding

The report algorithm identifier is `standalone-source-token-count`. The
evidence manifest binds that identifier to the attested historical
`finetune.py` identity:

- source snapshot commit
  `6912653d881bedee71ef527bc5650db55f115779`;
- Git blob `d334caa2cec91ed97a1974cdf61e3ab0d3edf415`;
- SHA-256
  `5de3316c8cf37edea97e83230fd90bf01092582dc25016c87fdc404aa1024e26`.

The generator reads those bytes only to hash-bind the source inventory. It
never imports or executes `finetune.py`, and it does not claim the new lab is
semantically equivalent to the full inherited trainer.

## Rebuild and verify

Create an isolated environment and install the exact optional runtime:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[tokenizer-lab]'
```

Provisioning may contact a package index. Evidence execution itself has no
network path and downloads no model, tokenizer, dataset, or checkpoint.

Regenerate or verify every committed byte:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python tools/render_token_boundary_evidence.py --write
PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python tools/render_token_boundary_evidence.py --check
```

Publication uses a fixed output directory and fixed inventory. Existing
destinations must be regular files. New files are staged and synchronized
before replacement, prior files are backed up, the manifest is replaced last,
and a failed publication restores the previous bundle. `--check` executes all
63 subprocesses again, reconstructs every output byte, rejects unmanifested
entries, and requires exact equality.

## Claim boundary

The positive claim is narrow: one real pinned tokenizer runtime executes a
small, local, synthetic artifact and produces deterministic counterexamples to
independent source/combined tokenization and length-based masking.

The bundle contains no secret or personal data and performs no network access,
GPU work, forward pass, loss, training, or evaluation. It loads no model or
dataset, does not attest a DeepSeek tokenizer, does not fix or execute the
inherited trainer, and makes no model-quality, prevalence, benchmark, or
universally correct masking-policy claim.
