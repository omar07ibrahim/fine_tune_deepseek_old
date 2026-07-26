"""Deterministic label construction and loss-topology auditing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .contract import SyntheticTrace, trace_sha256


IGNORE_INDEX = -100
ALL_TOKENS = "all_tokens"
ASSISTANT_ONLY = "assistant_only"
POLICIES = frozenset({ALL_TOKENS, ASSISTANT_ONLY})
AUDIT_SCHEMA_VERSION = 1
AUDIT_KIND = "loss-topology.audit"
BOUNDARY_KINDS = frozenset(
    {
        "prefix_special",
        "message_start",
        "message_end",
        "suffix_special",
    }
)


class TopologyError(RuntimeError):
    """A stable, redacted label-topology failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class SupervisedRun:
    """One half-open contiguous run of supervised label positions."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class PolicyAudit:
    """Immutable audit result for one deterministic label policy."""

    policy: str
    labels: tuple[int, ...]
    labels_sha256: str
    eligible_token_count: int
    supervised_token_count: int
    ignored_token_count: int
    supervised_runs: tuple[SupervisedRun, ...]
    boundary_supervised_token_count: int
    boundary_leakage_positions: tuple[int, ...]
    padding_leakage_positions: tuple[int, ...]
    off_policy_supervision_positions: tuple[int, ...]
    missing_eligible_positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Immutable audit over both supported v1 policies."""

    status: str
    input_sha256: str
    example_id: str
    message_count: int
    assistant_message_count: int
    token_count: int
    non_padding_token_count: int
    padding_token_count: int
    empty_assistant_message_indices: tuple[int, ...]
    issue_codes: tuple[str, ...]
    all_tokens: PolicyAudit
    assistant_only: PolicyAudit


def _eligible_positions(trace: SyntheticTrace, policy: str) -> frozenset[int]:
    if policy not in POLICIES:
        raise TopologyError("policy.unsupported")
    if policy == ALL_TOKENS:
        return frozenset(range(trace.trace.padding_start))

    eligible: set[int] = set()
    for segment in trace.trace.segments:
        if (
            segment.kind == "message_content"
            and segment.message_index is not None
            and trace.messages[segment.message_index].role == "assistant"
        ):
            eligible.update(range(segment.start, segment.end))
    return frozenset(eligible)


def _positions_for_kinds(
    trace: SyntheticTrace,
    kinds: frozenset[str],
) -> frozenset[int]:
    positions: set[int] = set()
    for segment in trace.trace.segments:
        if segment.kind in kinds:
            positions.update(range(segment.start, segment.end))
    return frozenset(positions)


def _supervised_runs(labels: tuple[int, ...]) -> tuple[SupervisedRun, ...]:
    runs: list[SupervisedRun] = []
    start: int | None = None
    for position, label in enumerate((*labels, IGNORE_INDEX)):
        if label != IGNORE_INDEX and start is None:
            start = position
        elif label == IGNORE_INDEX and start is not None:
            runs.append(SupervisedRun(start=start, end=position))
            start = None
    return tuple(runs)


def _labels_digest(labels: tuple[int, ...]) -> str:
    payload = (
        json.dumps(
            list(labels),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_labels(trace: SyntheticTrace, policy: str) -> tuple[int, ...]:
    """Build position-aligned labels for one v1 policy.

    The arrays describe selection topology only. No causal shift, forward
    pass, loss value, or optimization step is computed here.
    """

    eligible = _eligible_positions(trace, policy)
    return tuple(
        token_id if position in eligible else IGNORE_INDEX
        for position, token_id in enumerate(trace.trace.token_ids)
    )


def audit_label_topology(
    trace: SyntheticTrace,
    labels: tuple[int, ...],
    policy: str,
) -> PolicyAudit:
    """Audit an arbitrary label array against a deterministic v1 policy."""

    eligible = _eligible_positions(trace, policy)
    if len(labels) != len(trace.trace.token_ids):
        raise TopologyError("labels.length")
    for label, token_id in zip(labels, trace.trace.token_ids, strict=True):
        if isinstance(label, bool) or not isinstance(label, int):
            raise TopologyError("labels.type")
        if label != IGNORE_INDEX and label != token_id:
            raise TopologyError("labels.token_mismatch")

    supervised = frozenset(
        position for position, label in enumerate(labels) if label != IGNORE_INDEX
    )
    boundary = _positions_for_kinds(trace, BOUNDARY_KINDS)
    padding = frozenset(
        range(trace.trace.padding_start, len(trace.trace.token_ids))
    )
    off_policy = supervised - eligible
    boundary_leakage = off_policy & boundary
    padding_leakage = supervised & padding
    missing = eligible - supervised
    return PolicyAudit(
        policy=policy,
        labels=labels,
        labels_sha256=_labels_digest(labels),
        eligible_token_count=len(eligible),
        supervised_token_count=len(supervised),
        ignored_token_count=len(labels) - len(supervised),
        supervised_runs=_supervised_runs(labels),
        boundary_supervised_token_count=len(supervised & boundary),
        boundary_leakage_positions=tuple(sorted(boundary_leakage)),
        padding_leakage_positions=tuple(sorted(padding_leakage)),
        off_policy_supervision_positions=tuple(sorted(off_policy)),
        missing_eligible_positions=tuple(sorted(missing)),
    )


def audit_trace(trace: SyntheticTrace) -> AuditReport:
    """Construct and audit both v1 loss topologies."""

    empty_assistant_indices = tuple(
        index
        for index, message in enumerate(trace.messages)
        if message.role == "assistant"
        and next(
            segment
            for segment in trace.trace.segments
            if segment.kind == "message_content"
            and segment.message_index == index
        ).length
        == 0
    )
    issues = (
        ("assistant_target.empty",)
        if empty_assistant_indices
        else ()
    )
    all_labels = build_labels(trace, ALL_TOKENS)
    assistant_labels = build_labels(trace, ASSISTANT_ONLY)
    all_audit = audit_label_topology(trace, all_labels, ALL_TOKENS)
    assistant_audit = audit_label_topology(
        trace,
        assistant_labels,
        ASSISTANT_ONLY,
    )
    internal_failure = any(
        (
            all_audit.off_policy_supervision_positions,
            all_audit.missing_eligible_positions,
            assistant_audit.off_policy_supervision_positions,
            assistant_audit.missing_eligible_positions,
            assistant_audit.boundary_leakage_positions,
            assistant_audit.padding_leakage_positions,
        )
    )
    if internal_failure:
        raise TopologyError("audit.internal_invariant")
    return AuditReport(
        status="fail" if issues else "pass",
        input_sha256=trace_sha256(trace),
        example_id=trace.example_id,
        message_count=len(trace.messages),
        assistant_message_count=sum(
            message.role == "assistant" for message in trace.messages
        ),
        token_count=len(trace.trace.token_ids),
        non_padding_token_count=trace.trace.padding_start,
        padding_token_count=(
            len(trace.trace.token_ids) - trace.trace.padding_start
        ),
        empty_assistant_message_indices=empty_assistant_indices,
        issue_codes=issues,
        all_tokens=all_audit,
        assistant_only=assistant_audit,
    )


def _policy_to_primitive(audit: PolicyAudit) -> dict[str, Any]:
    return {
        "policy": audit.policy,
        "labels": list(audit.labels),
        "labels_sha256": audit.labels_sha256,
        "eligible_token_count": audit.eligible_token_count,
        "supervised_token_count": audit.supervised_token_count,
        "ignored_token_count": audit.ignored_token_count,
        "supervised_runs": [
            {"start": run.start, "end": run.end, "length": run.length}
            for run in audit.supervised_runs
        ],
        "boundary_supervised_token_count": (
            audit.boundary_supervised_token_count
        ),
        "boundary_leakage_positions": list(
            audit.boundary_leakage_positions
        ),
        "padding_leakage_positions": list(audit.padding_leakage_positions),
        "off_policy_supervision_positions": list(
            audit.off_policy_supervision_positions
        ),
        "missing_eligible_positions": list(
            audit.missing_eligible_positions
        ),
    }


def audit_to_primitive(report: AuditReport) -> dict[str, Any]:
    """Return a normalized JSON-compatible audit report."""

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "kind": AUDIT_KIND,
        "status": report.status,
        "input_sha256": report.input_sha256,
        "example_id": report.example_id,
        "trace_summary": {
            "message_count": report.message_count,
            "assistant_message_count": report.assistant_message_count,
            "token_count": report.token_count,
            "non_padding_token_count": report.non_padding_token_count,
            "padding_token_count": report.padding_token_count,
        },
        "diagnostics": {
            "empty_assistant_message_indices": list(
                report.empty_assistant_message_indices
            ),
            "issue_codes": list(report.issue_codes),
        },
        "policies": {
            ALL_TOKENS: _policy_to_primitive(report.all_tokens),
            ASSISTANT_ONLY: _policy_to_primitive(report.assistant_only),
        },
        "scope": {
            "input_class": "caller-supplied-synthetic-pretokenized-json",
            "tokenizer_executed": False,
            "tokenizer_mapping_attested": False,
            "trainer_imported_or_executed": False,
            "model_or_dataset_loaded": False,
            "causal_shift_or_loss_computed": False,
            "quality_metric": None,
        },
    }


def canonical_audit_bytes(report: AuditReport) -> bytes:
    """Return sorted, compact, newline-terminated UTF-8 audit JSON."""

    return (
        json.dumps(
            audit_to_primitive(report),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def audit_sha256(report: AuditReport) -> str:
    """Hash the canonical audit representation."""

    return hashlib.sha256(canonical_audit_bytes(report)).hexdigest()
