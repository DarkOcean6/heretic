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
    # AQUA-OPEN's output stage learns a refusal-specific input trigger and uses
    # it to subtract the refused-to-answered output difference. A zero legacy
    # default ensures older Optuna trials restore exactly as evaluated.
    output_transport_strength: float = 0.0
    output_transport_rank: int = 16
    output_preservation_weight: float = 1.0
    output_ridge_weight: float = 0.01
    output_protection_rank: int = 16
    output_max_relative_update: float = 0.05


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


def fit_selective_output_update(
    answered_inputs: Tensor,
    answered_outputs: Tensor,
    refused_inputs: Tensor,
    refused_outputs: Tensor,
    neighbor_count: int,
    rank: int,
    strength: float,
    preservation_weight: float,
    ridge_weight: float,
    protection_rank: int,
) -> Tensor:
    """Fit a conditional full-weight update without paired prompt records.

    A low-dimensional trigger basis is estimated from refused inputs relative to
    their nearest answered-input neighborhoods. Directions heavily used by the
    answered set are projected out of that trigger. Ridge regression then fits an
    output update that subtracts the refused-to-answered difference on refused
    inputs while targeting zero change on answered inputs.

    The returned tensor has the same shape as a linear module's weight and can be
    added directly to it. Although algebraically low rank, it is a direct weight
    edit rather than a LoRA adapter or inference-time module.
    """

    import torch

    validate_query_sets(answered_inputs, refused_inputs, neighbor_count)
    validate_query_sets(answered_outputs, refused_outputs, neighbor_count)
    if answered_inputs.shape[0] != answered_outputs.shape[0]:
        raise ValueError(
            "AQUA-OPEN answered input/output collections must have equal lengths."
        )
    if refused_inputs.shape[0] != refused_outputs.shape[0]:
        raise ValueError(
            "AQUA-OPEN refused input/output collections must have equal lengths."
        )
    if rank < 1:
        raise ValueError(f"AQUA-OPEN output trigger rank must be positive; got {rank}.")
    if not 0.0 <= strength <= 2.0:
        raise ValueError(
            "AQUA-OPEN output ablation strength must be between 0 and 2; "
            f"got {strength}."
        )
    if preservation_weight < 0.0:
        raise ValueError(
            "AQUA-OPEN output preservation weight cannot be negative; "
            f"got {preservation_weight}."
        )
    if ridge_weight < 0.0:
        raise ValueError(
            f"AQUA-OPEN output ridge weight cannot be negative; got {ridge_weight}."
        )
    if protection_rank < 0:
        raise ValueError(
            "AQUA-OPEN output protection rank cannot be negative; "
            f"got {protection_rank}."
        )

    answered_input = answered_inputs.float()
    answered = answered_outputs.float()
    refused_input = refused_inputs.float()
    refused = refused_outputs.float()
    answered_input_targets = nearest_neighbor_targets(
        refused_input,
        answered_input,
        neighbor_count,
    )
    answered_output_targets = nearest_neighbor_targets(
        refused,
        answered,
        neighbor_count,
    )

    # Differences from nearby answered inputs estimate where the refusal trigger
    # lives. This remains unpaired: neighborhoods are discovered independently at
    # every edited component.
    trigger_differences = refused_input - answered_input_targets
    maximum_rank = min(rank, *trigger_differences.shape)
    _, _, trigger_vectors = torch.linalg.svd(
        trigger_differences,
        full_matrices=False,
    )
    trigger_candidates = trigger_vectors[:maximum_rank].T.contiguous()

    # Protect high-variance answered-input directions. If the candidate refusal
    # trigger is inseparable from normal routing at this component, the projection
    # collapses it and the component receives little or no update.
    effective_protection_rank = min(protection_rank, *answered_input.shape)
    if effective_protection_rank > 0:
        _, _, answered_vectors = torch.linalg.svd(
            answered_input,
            full_matrices=False,
        )
        protected_basis = answered_vectors[:effective_protection_rank].T
        trigger_candidates = trigger_candidates - protected_basis @ (
            protected_basis.T @ trigger_candidates
        )

    trigger_left, trigger_values, _ = torch.linalg.svd(
        trigger_candidates,
        full_matrices=False,
    )
    epsilon = torch.finfo(trigger_values.dtype).eps
    threshold = (
        epsilon
        * max(trigger_candidates.shape)
        * trigger_values.max().clamp_min(epsilon)
    )
    retained = int((trigger_values > threshold).sum().item())
    if retained == 0 or strength == 0.0:
        return torch.zeros(
            (answered.shape[1], answered_input.shape[1]),
            dtype=answered.dtype,
            device=answered.device,
        )
    trigger_basis = trigger_left[:, :retained]

    refused_trigger = refused_input @ trigger_basis
    answered_trigger = answered_input @ trigger_basis
    preserve_scale = torch.sqrt(
        torch.as_tensor(
            preservation_weight,
            dtype=answered_trigger.dtype,
            device=answered_trigger.device,
        )
    )
    regression_inputs = torch.cat(
        (refused_trigger, preserve_scale * answered_trigger),
        dim=0,
    )
    # At strength 1 this subtracts the complete refused-to-answered difference;
    # values above 1 deliberately overcorrect past the nearest answered output in
    # the same spirit as ARA, but only when the learned trigger activates.
    refused_delta_targets = strength * (answered_output_targets - refused)
    regression_targets = torch.cat(
        (refused_delta_targets, torch.zeros_like(answered)),
        dim=0,
    )

    gram = regression_inputs.T @ regression_inputs
    gram_scale = gram.diagonal().mean().clamp_min(epsilon)
    identity = torch.eye(
        retained,
        dtype=gram.dtype,
        device=gram.device,
    )
    coefficients = torch.linalg.solve(
        gram + (ridge_weight * gram_scale + epsilon) * identity,
        regression_inputs.T @ regression_targets,
    )
    weight_update = coefficients.T @ trigger_basis.T

    if not torch.isfinite(weight_update).all():
        raise FloatingPointError(
            "AQUA-OPEN produced a non-finite selective output update."
        )

    return weight_update


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
