# Synthetic trace contract v1

LossTopology Lab accepts one deliberately narrow input class:
caller-supplied, synthetic, pretokenized JSON. The contract exists to study
label selection and message-boundary topology without downloading or executing
a tokenizer, model, trainer, dataset, checkpoint, or GPU runtime.

It does **not** prove that the supplied token IDs correspond to the supplied
message text. That mapping is an assertion by the fixture author. A future
real-tokenizer adapter would require its own pinned artifact identities,
licenses, checksums, and reproducibility evidence.

## Envelope

Every input object has exactly these fields:

| Field | Contract |
|---|---|
| `schema_version` | exact integer `1`; booleans are rejected |
| `kind` | exact string `loss-topology.synthetic-pretokenized-trace` |
| `example_id` | 1–64 lowercase ASCII identifier characters |
| `conversation` | object containing only `messages` |
| `trace` | object containing only `token_ids`, `padding_token_id`, and `segments` |

JSON objects reject duplicate keys. Unknown and missing fields fail closed at
every level. Floating-point and non-finite numbers are forbidden. Input is
UTF-8 and limited to 128 KiB, depth 8, and 20,000 parsed nodes.

## Conversation

Messages are immutable after parsing and contain exactly `role` and `content`.
V1 supports `system`, `user`, and `assistant`. An optional single system
message may appear first; user and assistant turns then alternate and the
conversation ends with an assistant turn.

The limits are 64 messages, 8 KiB UTF-8 per content string, and 64 KiB total
content. NUL is rejected. Empty content remains structurally valid so the
auditor can surface an empty assistant target rather than silently discarding
the example.

## Pretokenized trace

`token_ids` contains 1–4,096 exact integers in `[0, 2^31-1]`.
`padding_token_id` uses the same range. Segments are ordered, half-open spans
that cover every token exactly:

1. one `prefix_special` span;
2. `message_start`, `message_content`, and `message_end` for each message in
   exact message-index order;
3. one `suffix_special` span;
4. one terminal `padding` span.

Message-start and message-end spans must contain at least one token. Content,
prefix, suffix, and padding spans may be empty. Every token in the padding span
must equal `padding_token_id`; use of the same numeric ID outside padding is
not interpreted as padding.

This explicit segmentation distinguishes template boundaries from message
content. It avoids guessing role ownership from numeric token IDs.

## Canonical identity

After validation, the immutable model is serialized as sorted, compact,
newline-terminated UTF-8 JSON. SHA-256 over those canonical bytes is the
semantic input identity recorded in each audit. Whitespace and object-key
order in the source do not change this identity; message text, IDs, spans, or
roles do.

The healthy and deliberately failing examples are
`fixtures/synthetic/healthy.v1.json` and
`fixtures/synthetic/empty-assistant.v1.json`.
