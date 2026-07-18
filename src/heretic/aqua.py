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
    neighbor_count: int


def validate_query_sets(
    answered: Tensor,
    refused: Tensor,
    neighbor_count: int,
) -> None:
    """Validate independent answered and refused query collections."""

    if answered.ndim != 2 or refused.ndim != 2:
        raise ValueError(
            "AQUA-Q requires two-dimensional answered and refused query tensors; got "
            f"{tuple(answered.shape)} and refused queries {tuple(refused.shape)}."
        )
    if answered.shape[1] != refused.shape[1]:
        raise ValueError(
            "AQUA-Q answered and refused queries must have the same feature width; "
            f"got {answered.shape[1]} and {refused.shape[1]}."
        )
    maximum_neighbor_count = min(answered.shape[0], refused.shape[0])
    if not 1 <= neighbor_count <= maximum_neighbor_count:
        raise ValueError(
            "AQUA-Q neighbor_count must be between 1 and the smaller query-set "
            f"size ({maximum_neighbor_count}); got {neighbor_count}."
        )


def mean_distance_to_neighbors(
    queries: Tensor,
    references: Tensor,
    neighbor_count: int,
) -> Tensor:
    """Mean Euclidean distance to each query's nearest reference queries."""

    import torch

    distances = torch.cdist(queries, references)
    nearest_distances, _ = distances.topk(
        neighbor_count,
        dim=1,
        largest=False,
    )
    return nearest_distances.mean()


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

    validate_query_sets(
        original_answered_queries,
        original_refused_queries,
        parameters.neighbor_count,
    )
    preserve_answered = F.mse_loss(
        new_answered_queries,
        original_answered_queries,
    )
    align_refused = mean_distance_to_neighbors(
        new_refused_queries,
        original_answered_queries,
        parameters.neighbor_count,
    )
    escape_refusal = mean_distance_to_neighbors(
        new_refused_queries,
        original_refused_queries,
        parameters.neighbor_count,
    )
    update_norm = update.square().mean()

    return (
        parameters.preserve_answered_weight * preserve_answered
        + parameters.align_refused_weight
        * (align_refused - parameters.overcorrect_relative_weight * escape_refusal)
        + parameters.update_norm_weight * update_norm
    )
