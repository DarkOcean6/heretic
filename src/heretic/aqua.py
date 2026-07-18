# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class AQUAParameters:
    """Trial parameters for protected attention-output wall rewiring."""

    start_layer_index: int
    end_layer_index: int
    neighbor_count: int
    wall_ablation_strength: float = 0.0
    wall_rank: int = 4
    output_protection_rank: int = 16
    # This is a total cross-layer relative update budget. Model integration
    # divides it by sqrt(number of edited layers) before capping each tensor.
    output_max_relative_update: float = 0.05

    # Legacy fields remain loadable so older AQUA studies can still be inspected
    # and exported. The simplified runtime no longer uses them.
    preserve_answered_weight: float = 1.0
    align_refused_weight: float = 0.0
    overcorrect_relative_weight: float = 0.0
    update_norm_weight: float = 0.0
    openness_weight: float = 1.0
    openness_margin: float = 0.15
    answered_geometry_weight: float = 0.5
    output_transport_strength: float = 0.0
    output_transport_rank: int = 16
    output_preservation_weight: float = 1.0
    output_ridge_weight: float = 0.01
    routing_max_relative_update: float = 0.05
    wall_rewire_strength: float = 0.0


def per_layer_relative_budget(total_budget: float, layer_count: int) -> float:
    """Distribute a relative L2 edit budget across a layer range."""

    if not 0.0 <= total_budget <= 1.0:
        raise ValueError(
            "AQUA-OPEN total update budget must be between 0 and 1; "
            f"got {total_budget}."
        )
    if layer_count < 1:
        raise ValueError(
            f"AQUA-OPEN edited layer count must be positive; got {layer_count}."
        )
    return total_budget / math.sqrt(layer_count)


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
    if not 0.0 <= strength <= 3.0:
        raise ValueError(
            "AQUA-OPEN output transport strength must be between 0 and 3; "
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


def fit_protected_wall_rewire(
    answered_outputs: Tensor,
    refused_outputs: Tensor,
    weight: Tensor,
    neighbor_count: int,
    rank: int,
    ablation_strength: float,
    rewire_strength: float,
    protection_rank: int,
) -> Tensor:
    """Return a protected output-wall ablation and answer-route update.

    Refused outputs are compared with unpaired nearest answered outputs. The top
    difference directions are projected away from high-variance answered-output
    directions, producing a localized wall basis. Components written into this
    basis are attenuated and redirected into an answered-output basis. The result
    is a direct full-weight update for ``attn.o_proj``.
    """

    import torch

    validate_query_sets(answered_outputs, refused_outputs, neighbor_count)
    if rank < 1:
        raise ValueError(f"AQUA-OPEN wall rank must be positive; got {rank}.")
    if not 0.0 <= ablation_strength <= 2.0:
        raise ValueError(
            "AQUA-OPEN wall ablation strength must be between 0 and 2; "
            f"got {ablation_strength}."
        )
    if not 0.0 <= rewire_strength <= 2.0:
        raise ValueError(
            "AQUA-OPEN wall rewire strength must be between 0 and 2; "
            f"got {rewire_strength}."
        )
    if protection_rank < 0:
        raise ValueError(
            f"AQUA-OPEN wall protection rank cannot be negative; got {protection_rank}."
        )

    answered = answered_outputs.float()
    refused = refused_outputs.float()
    matrix = weight.float()
    targets = nearest_neighbor_targets(refused, answered, neighbor_count)
    wall_samples = refused - targets

    effective_rank = min(rank, *wall_samples.shape)
    _, _, wall_vectors = torch.linalg.svd(wall_samples, full_matrices=False)

    # Always include the classic rank-one mean refusal direction first. Higher
    # ranks add the dominant residual wall directions, so rank=1 behaves like an
    # old-school directional ablation and larger ranks behave like a localized
    # multi-direction ablation.
    mean_wall = wall_samples.mean(dim=0)
    mean_norm = torch.linalg.vector_norm(mean_wall)
    if mean_norm > torch.finfo(mean_wall.dtype).eps:
        mean_wall = (mean_wall / mean_norm).unsqueeze(1)
        wall_candidates = torch.cat(
            [mean_wall, wall_vectors[:effective_rank].T],
            dim=1,
        )
        wall_candidates = torch.linalg.qr(wall_candidates, mode="reduced").Q[
            :, :effective_rank
        ]
    else:
        wall_candidates = wall_vectors[:effective_rank].T.contiguous()

    effective_protection_rank = min(protection_rank, *answered.shape)
    if effective_protection_rank > 0:
        _, _, protected_vectors = torch.linalg.svd(answered, full_matrices=False)
        protected_basis = protected_vectors[:effective_protection_rank].T
        wall_candidates = wall_candidates - protected_basis @ (
            protected_basis.T @ wall_candidates
        )

    wall_left, wall_values, _ = torch.linalg.svd(
        wall_candidates,
        full_matrices=False,
    )
    epsilon = torch.finfo(wall_values.dtype).eps
    threshold = (
        epsilon * max(wall_candidates.shape) * wall_values.max().clamp_min(epsilon)
    )
    retained = int((wall_values > threshold).sum().item())
    if retained == 0 or (ablation_strength == 0.0 and rewire_strength == 0.0):
        return torch.zeros_like(matrix)
    wall_basis = wall_left[:, :retained]

    # Find answer directions that are distinct from the wall. These are existing
    # output routes, not newly added capacity.
    answer_rank = min(retained, *targets.shape)
    _, _, answer_vectors = torch.linalg.svd(targets, full_matrices=False)
    answer_candidates = answer_vectors[:answer_rank].T
    answer_candidates = answer_candidates - wall_basis @ (
        wall_basis.T @ answer_candidates
    )
    answer_left, answer_values, _ = torch.linalg.svd(
        answer_candidates,
        full_matrices=False,
    )
    answer_threshold = (
        epsilon * max(answer_candidates.shape) * answer_values.max().clamp_min(epsilon)
    )
    answer_retained = int((answer_values > answer_threshold).sum().item())
    shared_rank = min(retained, answer_retained)
    if shared_rank == 0:
        rewire_strength = 0.0
        shared_rank = retained
        answer_basis = torch.zeros(
            (answered.shape[1], shared_rank),
            dtype=answered.dtype,
            device=answered.device,
        )
        alignment = torch.zeros(
            (shared_rank, shared_rank),
            dtype=answered.dtype,
            device=answered.device,
        )
    else:
        wall_basis = wall_basis[:, :shared_rank]
        answer_basis = answer_left[:, :shared_rank]
        wall_coordinates = wall_samples @ wall_basis
        answer_coordinates = targets @ answer_basis
        left, _, right = torch.linalg.svd(
            wall_coordinates.T @ answer_coordinates,
        )
        alignment = left @ right

    wall_read = wall_basis.T @ matrix
    weight_update = -ablation_strength * wall_basis @ wall_read
    if rewire_strength > 0.0:
        weight_update = weight_update + rewire_strength * answer_basis @ (
            alignment.T @ wall_read
        )

    if not torch.isfinite(weight_update).all():
        raise FloatingPointError(
            "AQUA-OPEN produced a non-finite protected wall rewire."
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
    epsilon = torch.finfo(original_answered_queries.dtype).eps
    answered_energy = original_answered_queries.square().mean().clamp_min(epsilon)
    preserve_answered = (
        F.mse_loss(new_answered_queries, original_answered_queries) / answered_energy
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
    original_answered_distances = mean_distance_to_neighbors_per_query(
        original_refused_queries,
        original_answered_queries,
        parameters.neighbor_count,
    ).detach()
    distance_floor = original_answered_distances.mean().clamp_min(epsilon) * 0.001
    original_answered_distances = original_answered_distances.clamp_min(distance_floor)
    relative_answered_distance = answered_distances / original_answered_distances
    relative_refusal_distance = refusal_distances / original_answered_distances
    align_refused = relative_answered_distance.mean()
    escape_refusal = relative_refusal_distance.mean()
    openness = F.relu(
        relative_answered_distance
        - relative_refusal_distance
        + parameters.openness_margin
    ).mean()

    # The prior linear escape reward was unbounded below: a trial could lower its
    # loss indefinitely by making query magnitudes huge. tanh preserves the reward
    # for moving away from the refusal neighborhood while capping it in [0, 1).
    bounded_escape_reward = torch.tanh(escape_refusal)

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
