# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class AQUAParameters:
    """Trial parameters for Attention Query Unblocking for Open Expression."""

    start_layer_index: int
    end_layer_index: int
    preserve_answered_weight: float
    align_refused_weight: float
    overcorrect_relative_weight: float
    update_norm_weight: float
    neighbor_count: int
    # Defaults keep older AQUA-Q Optuna trials loadable while applying the new
    # openness and preservation behavior when they are selected.
    openness_weight: float = 1.0
    openness_margin: float = 0.15
    answered_geometry_weight: float = 0.5


def validate_query_sets(
    answered: Tensor,
    refused: Tensor,
    neighbor_count: int,
) -> None:
    """Validate independent answered and refused query collections."""

    if answered.ndim != 2 or refused.ndim != 2:
        raise ValueError(
            "AQUA-OPEN requires two-dimensional answered and refused query tensors; got "
            f"{tuple(answered.shape)} and refused queries {tuple(refused.shape)}."
        )
    if answered.shape[1] != refused.shape[1]:
        raise ValueError(
            "AQUA-OPEN answered and refused queries must have the same feature width; "
            f"got {answered.shape[1]} and {refused.shape[1]}."
        )
    maximum_neighbor_count = min(answered.shape[0], refused.shape[0])
    if not 1 <= neighbor_count <= maximum_neighbor_count:
        raise ValueError(
            "AQUA-OPEN neighbor_count must be between 1 and the smaller query-set "
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


def mean_distance_to_neighbors_per_query(
    queries: Tensor,
    references: Tensor,
    neighbor_count: int,
) -> Tensor:
    """Mean nearest-neighbor distance for every query independently."""

    import torch

    distances = torch.cdist(queries, references)
    nearest_distances, _ = distances.topk(
        neighbor_count,
        dim=1,
        largest=False,
    )
    return nearest_distances.mean(dim=1)


def cosine_geometry(queries: Tensor) -> Tensor:
    """Pairwise cosine geometry used to protect normal query relationships."""

    import torch.nn.functional as F

    normalized = F.normalize(queries, p=2, dim=1)
    return normalized @ normalized.T


def aqua_query_loss(
    new_answered_queries: Tensor,
    original_answered_queries: Tensor,
    new_refused_queries: Tensor,
    original_refused_queries: Tensor,
    update: Tensor,
    parameters: AQUAParameters,
) -> Tensor:
    """AQUA-OPEN's query-opening and preservation objective.

    The refused set is treated as blocked but answerable. Its queries are moved into
    answered-query neighborhoods and are required to become closer to answered than
    refusal neighborhoods by a scale-aware contrastive margin. Answered-query values
    and their pairwise geometry are protected, while a direct weight penalty limits
    unnecessary change to the full query-projection matrix.
    """

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
    answered_distances = mean_distance_to_neighbors_per_query(
        new_refused_queries,
        original_answered_queries,
        parameters.neighbor_count,
    )
    refusal_distances = mean_distance_to_neighbors_per_query(
        new_refused_queries,
        original_refused_queries,
        parameters.neighbor_count,
    )
    align_refused = answered_distances.mean()
    escape_refusal = refusal_distances.mean()

    # Scale the contrastive margin by the original distance between the blocked
    # queries and the answered distribution. This avoids a model-size-dependent
    # fixed Euclidean margin.
    original_answered_distances = mean_distance_to_neighbors_per_query(
        original_refused_queries,
        original_answered_queries,
        parameters.neighbor_count,
    ).detach()
    openness_margin = parameters.openness_margin * original_answered_distances
    openness = F.relu(
        answered_distances - refusal_distances + openness_margin
    ).mean()

    preserve_answered_geometry = F.mse_loss(
        cosine_geometry(new_answered_queries),
        cosine_geometry(original_answered_queries),
    )
    update_norm = update.square().mean()

    return (
        parameters.preserve_answered_weight * preserve_answered
        + parameters.align_refused_weight
        * (align_refused - parameters.overcorrect_relative_weight * escape_refusal)
        + parameters.openness_weight * openness
        + parameters.answered_geometry_weight * preserve_answered_geometry
        + parameters.update_norm_weight * update_norm
    )
