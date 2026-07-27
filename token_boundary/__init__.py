"""Strict contracts for the local tokenizer-boundary differential lab."""

from .contract import (
    BoundaryCase,
    BoundaryContractError,
    boundary_case_sha256,
    canonical_boundary_case_bytes,
    parse_boundary_case,
)

__all__ = [
    "BoundaryCase",
    "BoundaryContractError",
    "boundary_case_sha256",
    "canonical_boundary_case_bytes",
    "parse_boundary_case",
]
