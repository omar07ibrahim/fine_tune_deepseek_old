# Reproducible evidence contract

The visual bundle is executable evidence for the original LossTopology Lab,
not decoration and not evidence that the inherited trainer works.

## Closed inputs

`tools/render_loss_topology_evidence.py` is hardcoded to:

- `fixtures/synthetic/healthy.v1.json`;
- `fixtures/synthetic/empty-assistant.v1.json`;
- the standard-library-only `loss_topology` package.

The generator accepts no fixture, module, command, output-directory, model, or
network argument. Its manifest binds the exact bytes of those sources and the
generator itself. It never imports or executes `finetune.py`, and it has no
tokenizer, model, dataset, checkpoint, training runtime, network, or GPU path.

The fixture token IDs and segment boundaries were authored by hand. A
tokenizer-to-message mapping is not attested.

## Executed evidence

The generator copies the two committed fixture byte strings into a temporary
repository-local sandbox and starts exactly this module through `python3`:

```text
python3 -m loss_topology.cli
```

The complete argument vectors, raw fixture SHA-256 values, canonical input
identities, stdout hashes, stderr hashes, exit codes, and report hashes are
recorded in `docs/evidence/generated/loss-topology-evidence.v1.json`.

The healthy trace returns exit 0 and a passing structural audit. The valid
empty-assistant trace writes a complete failing report and returns exit 1.
Exit 1 is a diagnostic result. Contract or I/O errors instead return exit 2
and are not represented as successful evidence runs.

Three additional rows exercise `audit_label_topology` directly:

| Controlled mutation | Exact expected detection |
|---|---|
| supervise boundary position 12 | boundary and off-policy position 12 |
| supervise user position 8 and padding position 31 | off-policy positions 8 and 31; padding position 31 |
| ignore eligible assistant position 14 | missing eligible position 14 |

These are synthetic in-memory label mutations. They are not accepted by or
reported from the CLI.

## Generated bundle

The generated directory contains:

| Artifact | Meaning |
|---|---|
| `policy-topology.svg` | exact selection lanes for both supported policies |
| `assistant-only-fault-diagnostics.svg` | executed public-API fault mutations and detections |
| `cli-session.txt` | raw actual subprocess transcript |
| `cli-session.svg` | deterministic rendering of that transcript |
| `healthy.audit.json` | canonical CLI report for the healthy trace |
| `empty-assistant.audit.json` | canonical CLI report for the empty assistant target |
| `loss-topology-evidence.v1.json` | exact source, execution, fault, scope, and artifact manifest |

SVG files contain no script, linked image, external font, or remote resource.
They include accessible titles and descriptions. No timestamp, hostname,
absolute filesystem path, secret, or personal data enters the bundle.

## Rebuild and verify

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python3 tools/render_loss_topology_evidence.py --write
PYTHONDONTWRITEBYTECODE=1 \
  python3 tools/render_loss_topology_evidence.py --check
PYTHONDONTWRITEBYTECODE=1 \
  python3 -m unittest discover -s tests -v
```

All output files are staged as descriptor-relative regular files. Existing
destinations are backed up before publication. Artifacts are atomically
replaced in a fixed order, the manifest is replaced last, and a publication
failure restores the previous bundle before temporary files are removed.
`--check` reconstructs the evidence through fresh CLI subprocesses and rejects
any byte drift.

## Claim boundary

The evidence demonstrates deterministic parsing, label selection, structural
diagnostics, canonical reporting, and fail-closed publication for two
synthetic traces. It does not apply a causal shift, compute a loss or
perplexity, run a forward pass, train or evaluate a model, attest a tokenizer
mapping, or make a model-quality claim.

Under `all_tokens`, selecting template and message-boundary positions is the
defined policy behavior and is not described as assistant-only leakage.
Leakage diagnostics are evaluated relative to `assistant_only`.
