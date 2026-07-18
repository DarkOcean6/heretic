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
    # AQUA-OPEN adds an isospectral attention-output transport. A zero legacy
    # default ensures old Optuna trials restore exactly as they were evaluated;
    # newly sampled trials always provide an explicit nonzero strength.
    output_transport_strength: float = 0.0
    output_transport_rank: int = 16
    output_preservation_weight: float = 1.0


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


def nearest_neighbor_targets(
    queries: Tensor,
    references: Tensor,
    neighbor_count: int,
) -> Tensor:
    """Return the mean of each query's nearest unpaired reference vectors."""

    import torch

    validate_query_sets(references, queries, neighbor_count)
    distances = torch.cdist(queries, references)
    _, nearest_indices = distances.topk(
        neighbor_count,
        dim=1,
        largest=False,
    )
    return references[nearest_indices].mean(dim=1)


def fit_orthogonal_output_transport(
    answered_outputs: Tensor,
    refused_outputs: Tensor,
    neighbor_count: int,
    rank: int,
    strength: float,
    preservation_weight: float,
) -> tuple[Tensor, Tensor]:
    """Fit a low-dimensional unpaired orthogonal output transport.

    The returned ``basis`` has orthonormal columns. ``row_transport`` is an
    orthogonal matrix in that basis which maps row-vector outputs toward their
    nearest answered neighborhoods. Lifting it as

        Q = I + basis @ (row_transport.T - I) @ basis.T

    and applying ``Q @ W`` preserves the rank and singular values of ``W`` in
    exact arithmetic. Already-answered outputs are included as identity targets,
    making the transport preservation-aware without requiring paired prompts.
    """

    import torch

    validate_query_sets(answered_outputs, refused_outputs, neighbor_count)
    if rank < 1:
        raise ValueError(
            f"AQUA-OPEN output transport rank must be positive; got {rank}."
        )
    if not 0.0 <= strength <= 1.0:
        raise ValueError(
            "AQUA-OPEN output transport strength must be between 0 and 1; "
            f"got {strength}."
        )
    if preservation_weight < 0.0:
        raise ValueError(
            "AQUA-OPEN output preservation weight cannot be negative; "
            f"got {preservation_weight}."
        )

    answered = answered_outputs.float()
    refused = refused_outputs.float()
    targets = nearest_neighbor_targets(
        refused,
        answered,
        neighbor_count,
    )

    # Use the leading right-singular directions of the observed route states as
    # the intervention subspace. The expensive orthogonal solve then happens only
    # in this small space rather than across the full model width.
    joint_outputs = torch.cat((refused, targets, answered), dim=0)
    maximum_rank = min(joint_outputs.shape)
    effective_rank = min(rank, maximum_rank)
    _, _, right_vectors = torch.linalg.svd(joint_outputs, full_matrices=False)
    basis = right_vectors[:effective_rank].T.contiguous()

    refused_coordinates = refused @ basis
    target_coordinates = targets @ basis
    answered_coordinates = answered @ basis
    preserve_scale = torch.sqrt(
        torch.as_tensor(
            preservation_weight,
            dtype=answered_coordinates.dtype,
            device=answered_coordinates.device,
        )
    )
    source = torch.cat(
        (refused_coordinates, preserve_scale * answered_coordinates),
        dim=0,
    )
    destination = torch.cat(
        (target_coordinates, preserve_scale * answered_coordinates),
        dim=0,
    )

    # Standard orthogonal Procrustes: source @ row_transport ~= destination.
    left, _, right = torch.linalg.svd(source.T @ destination)
    row_transport = left @ right

    # Polar-project a blend with identity back onto the orthogonal group. This
    # supplies a bounded strength control without attenuating any output axis.
    identity = torch.eye(
        effective_rank,
        dtype=row_transport.dtype,
        device=row_transport.device,
    )
    blended = (1.0 - strength) * identity + strength * row_transport
    blend_left, _, blend_right = torch.linalg.svd(blended)
    row_transport = blend_left @ blend_right

    if not torch.isfinite(basis).all() or not torch.isfinite(row_transport).all():
        raise FloatingPointError(
            "AQUA-OPEN produced a non-finite orthogonal output transport."
        )

    return basis, row_transport


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
    unnecessary change to each full attention-routing projection matrix.
    """

    import torch
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
    distance_scale = original_answered_distances.mean().clamp_min(
        torch.finfo(original_answered_distances.dtype).eps
    )
    openness_margin = parameters.openness_margin * original_answered_distances
    openness = F.relu(answered_distances - refusal_distances + openness_margin).mean()

    # The prior linear escape reward was unbounded below: a trial could lower its
    # loss indefinitely by making query magnitudes huge. tanh preserves the reward
    # for moving away from the refusal neighborhood while capping it in [0, 1).
    bounded_escape_reward = torch.tanh(escape_refusal / distance_scale)

    preserve_answered_geometry = F.mse_loss(
        cosine_geometry(new_answered_queries),
        cosine_geometry(original_answered_queries),
    )
    update_norm = update.square().mean()

    return (
        parameters.preserve_answered_weight * preserve_answered
        + parameters.align_refused_weight
        * (
            align_refused
            - parameters.overcorrect_relative_weight * bounded_escape_reward
        )
        + parameters.openness_weight * openness
        + parameters.answered_geometry_weight * preserve_answered_geometry
        + parameters.update_norm_weight * update_norm
    )
