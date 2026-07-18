# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class AQUAParameters:
    """Trial parameters for Attention Query Unblocking Alignment (AQUA-Q)."""

    start_layer_index: int
    end_layer_index: int
    preserve_answered_weight: float
    align_refused_weight: float
    overcorrect_relative_weight: float
    update_norm_weight: float


def validate_paired_queries(answered: Tensor, refused: Tensor) -> None:
    """Ensure AQUA received one answered query for every refused query."""

    if answered.shape != refused.shape:
        raise ValueError(
            "AQUA-Q requires paired prompt datasets with identical lengths and "
            "query shapes; got answered queries "
            f"{tuple(answered.shape)} and refused queries {tuple(refused.shape)}."
        )


def build_query_targets(
    answered_queries: Tensor,
    refused_queries: Tensor,
    overcorrect_relative_weight: float,
) -> Tensor:
    """Build bounded targets beyond the answered query, away from refusal queries."""

    validate_paired_queries(answered_queries, refused_queries)
    return answered_queries + overcorrect_relative_weight * (
        answered_queries - refused_queries
    )


def aqua_query_loss(
    new_answered_queries: Tensor,
    original_answered_queries: Tensor,
    new_refused_queries: Tensor,
    original_refused_queries: Tensor,
    update: Tensor,
    parameters: AQUAParameters,
) -> Tensor:
    """AQUA-Q's local query-alignment and preservation objective."""

    import torch.nn.functional as F

    targets = build_query_targets(
        original_answered_queries,
        original_refused_queries,
        parameters.overcorrect_relative_weight,
    )
    preserve_answered = F.mse_loss(
        new_answered_queries,
        original_answered_queries,
    )
    align_refused = F.mse_loss(new_refused_queries, targets)
    update_norm = update.square().mean()

    return (
        parameters.preserve_answered_weight * preserve_answered
        + parameters.align_refused_weight * align_refused
        + parameters.update_norm_weight * update_norm
    )
