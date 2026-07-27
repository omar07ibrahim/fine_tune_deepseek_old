"""Strict public API for the local tokenizer-boundary differential lab."""

from .contract import (
    BoundaryCase,
    BoundaryContractError,
    boundary_case_sha256,
    canonical_boundary_case_bytes,
    parse_boundary_case,
)
from .errors import BoundaryEngineError
from .report import (
    BoundaryReport,
    BoundaryReportError,
    boundary_report_sha256,
    boundary_report_to_primitive,
    canonical_boundary_report_bytes,
)


def analyze_boundary(case: BoundaryCase) -> BoundaryReport:
    """Load the optional pinned runtime only when an audit is requested."""

    try:
        from .engine import analyze_boundary as implementation
    except ModuleNotFoundError as error:
        if error.name != "tokenizers":
            raise
        runtime_unavailable = True
    else:
        runtime_unavailable = False
    if runtime_unavailable:
        raise BoundaryEngineError("runtime.unavailable")
    return implementation(case)


__all__ = [
    "BoundaryCase",
    "BoundaryContractError",
    "BoundaryEngineError",
    "BoundaryReport",
    "BoundaryReportError",
    "analyze_boundary",
    "boundary_case_sha256",
    "boundary_report_sha256",
    "boundary_report_to_primitive",
    "canonical_boundary_case_bytes",
    "canonical_boundary_report_bytes",
    "parse_boundary_case",
]
