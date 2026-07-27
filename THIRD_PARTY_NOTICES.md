# Third-party notices

## Inherited DeepSeek-MoE code

The repository began as a three-file training snapshot:

- `finetune.py`
- `configs/ds_config_zero3.json`
- `requirements.txt`

A provenance review found substantial overlap between `finetune.py` and the
official [DeepSeek-MoE](https://github.com/deepseek-ai/DeepSeek-MoE) trainer,
and a semantic match between the local and upstream ZeRO-3 configurations.
`requirements.txt` is locked with the same inherited snapshot, but no separate
line-level upstream relationship is asserted for it.

The DeepSeek-MoE code repository publishes its code under
[`LICENSE-CODE`](https://github.com/deepseek-ai/DeepSeek-MoE/blob/66edeee5a4f75cbd76e0316229ad101805a90e01/LICENSE-CODE),
an MIT license with a 2023 DeepSeek copyright notice. A local copy is retained
at [`third_party/deepseek-moe/LICENSE-CODE`](third_party/deepseek-moe/LICENSE-CODE).
That notice applies to the inherited DeepSeek-derived portions; it is not a
repository-wide license declaration and does not relicense model assets.

The exact local file identities and the recorded comparison observations live
in [`provenance/legacy-snapshot.v1.json`](provenance/legacy-snapshot.v1.json).
The comparison is pinned to upstream revision
[`66edeee5`](https://github.com/deepseek-ai/DeepSeek-MoE/tree/66edeee5a4f75cbd76e0316229ad101805a90e01)
and exact Git blob identities. Because those upstream bytes are not vendored,
the local verifier recomputes only the local side of the evidence. It does not
claim to reproduce the cross-repository comparison offline.

## Optional Tokenizers runtime

The tokenizer-boundary lab declares
[`tokenizers==0.21.4`](https://pypi.org/project/tokenizers/0.21.4/) as an exact
optional runtime. The package is published under the Apache Software License
and provides the Rust-backed tokenizer implementation exercised by the local
synthetic artifact. No Tokenizers source or binary is vendored in this
repository.

Installing the optional extra may resolve transitive packages and may contact a
package index. Those packages retain their own terms. The lab's offline claim
starts only after provisioning: its audit API accepts no Hub identifier, URL,
or caller-selected tokenizer artifact.

## Separate model and data terms

The upstream code license is not a model license. Anyone who later supplies a
model, tokenizer, dataset, or checkpoint must review and satisfy the terms for
those assets independently. This repository currently vendors and attests none
of them.
