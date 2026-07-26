"""Strict, standard-library-only SFT loss-topology auditing.

This package deliberately operates on caller-supplied synthetic token IDs. It
does not import the inherited trainer, a tokenizer, a model, or a dataset.
"""

from .contract import (
    ContractError,
    Message,
    PretokenizedTrace,
    SyntheticTrace,
    TokenSegment,
    canonical_trace_bytes,
    parse_synthetic_trace,
    trace_sha256,
)
from .topology import (
    ALL_TOKENS,
    ASSISTANT_ONLY,
    IGNORE_INDEX,
    AuditReport,
    PolicyAudit,
    SupervisedRun,
    TopologyError,
    audit_label_topology,
    audit_sha256,
    audit_trace,
    build_labels,
    canonical_audit_bytes,
)

__all__ = [
    "ALL_TOKENS",
    "ASSISTANT_ONLY",
    "IGNORE_INDEX",
    "AuditReport",
    "ContractError",
    "Message",
    "PolicyAudit",
    "PretokenizedTrace",
    "SupervisedRun",
    "SyntheticTrace",
    "TokenSegment",
    "TopologyError",
    "audit_label_topology",
    "audit_sha256",
    "audit_trace",
    "build_labels",
    "canonical_audit_bytes",
    "canonical_trace_bytes",
    "parse_synthetic_trace",
    "trace_sha256",
]
