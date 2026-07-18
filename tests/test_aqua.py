# SPDX-License-Identifier: AGPL-3.0-or-later

import unittest
import math

import numpy as np
import torch

from heretic.aqua import (
    AQUAParameters,
    aqua_query_loss,
    fit_protected_wall_rewire,
    fit_selective_output_update,
    nearest_neighbor_targets,
    per_layer_relative_budget,
    validate_query_sets,
)


class AQUATest(unittest.TestCase):
    def setUp(self):
        self.parameters = AQUAParameters(
            start_layer_index=1,
            end_layer_index=3,
            preserve_answered_weight=2.0,
            align_refused_weight=3.0,
            overcorrect_relative_weight=0.5,
            openness_weight=1.25,
            openness_margin=0.2,
            answered_geometry_weight=0.75,
            output_transport_strength=0.8,
            output_transport_rank=2,
            output_preservation_weight=2.5,
            output_ridge_weight=0.02,
            output_protection_rank=1,
            output_max_relative_update=0.1,
            routing_max_relative_update=0.08,
            wall_ablation_strength=0.75,
            wall_rewire_strength=1.1,
            wall_rank=1,
            update_norm_weight=0.25,
            neighbor_count=2,
        )

    def test_query_sets_do_not_need_to_be_paired(self):
        answered = np.zeros((4, 3))
        refused = np.zeros((7, 3))

        validate_query_sets(answered, refused, neighbor_count=2)

    def test_total_update_budget_is_shared_across_layers(self):
        per_layer = per_layer_relative_budget(0.1, 4)

        self.assertAlmostEqual(per_layer * math.sqrt(4), 0.1)

    def test_feature_width_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same feature width"):
            validate_query_sets(np.zeros((2, 3)), np.zeros((3, 4)), 1)

    def test_neighbor_count_cannot_exceed_answered_queries(self):
        with self.assertRaisesRegex(ValueError, "neighbor_count"):
            validate_query_sets(np.zeros((2, 3)), np.zeros((3, 3)), 3)

    def test_parameters_capture_full_weight_optimization_controls(self):
        self.assertEqual(self.parameters.start_layer_index, 1)
        self.assertEqual(self.parameters.end_layer_index, 3)
        self.assertEqual(self.parameters.overcorrect_relative_weight, 0.5)
        self.assertEqual(self.parameters.openness_weight, 1.25)
        self.assertEqual(self.parameters.openness_margin, 0.2)
        self.assertEqual(self.parameters.answered_geometry_weight, 0.75)
        self.assertEqual(self.parameters.output_transport_strength, 0.8)
        self.assertEqual(self.parameters.output_transport_rank, 2)
        self.assertEqual(self.parameters.output_preservation_weight, 2.5)
        self.assertEqual(self.parameters.output_ridge_weight, 0.02)
        self.assertEqual(self.parameters.output_protection_rank, 1)
        self.assertEqual(self.parameters.output_max_relative_update, 0.1)
        self.assertEqual(self.parameters.routing_max_relative_update, 0.08)
        self.assertEqual(self.parameters.wall_ablation_strength, 0.75)
        self.assertEqual(self.parameters.wall_rewire_strength, 1.1)
        self.assertEqual(self.parameters.wall_rank, 1)
        self.assertEqual(self.parameters.neighbor_count, 2)

    def test_older_aqua_trial_parameters_receive_open_defaults(self):
        parameters = AQUAParameters(
            start_layer_index=1,
            end_layer_index=3,
            preserve_answered_weight=2.0,
            align_refused_weight=3.0,
            overcorrect_relative_weight=0.5,
            update_norm_weight=0.25,
            neighbor_count=2,
        )

        self.assertEqual(parameters.openness_weight, 1.0)
        self.assertEqual(parameters.openness_margin, 0.15)
        self.assertEqual(parameters.answered_geometry_weight, 0.5)
        self.assertEqual(parameters.output_transport_strength, 0.0)
        self.assertEqual(parameters.output_transport_rank, 16)
        self.assertEqual(parameters.output_preservation_weight, 1.0)
        self.assertEqual(parameters.output_ridge_weight, 0.01)
        self.assertEqual(parameters.output_protection_rank, 16)
        self.assertEqual(parameters.output_max_relative_update, 0.05)
        self.assertEqual(parameters.routing_max_relative_update, 0.05)
        self.assertEqual(parameters.wall_ablation_strength, 0.0)
        self.assertEqual(parameters.wall_rewire_strength, 0.0)
        self.assertEqual(parameters.wall_rank, 4)

    def test_nearest_neighbor_targets_do_not_require_pairs(self):
        answered = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        refused = torch.tensor([[9.0, 1.0], [19.0, 1.0]])

        targets = nearest_neighbor_targets(refused, answered, neighbor_count=1)

        torch.testing.assert_close(
            targets,
            torch.tensor([[10.0, 0.0], [20.0, 0.0]]),
        )

    def test_selective_output_update_improves_refused_alignment(self):
        answered_inputs = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
        answered_outputs = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
        refused_inputs = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
        refused_outputs = torch.tensor([[1.0, 0.0], [2.0, 0.0]])

        weight_update = fit_selective_output_update(
            answered_inputs,
            answered_outputs,
            refused_inputs,
            refused_outputs,
            neighbor_count=1,
            rank=1,
            strength=1.0,
            preservation_weight=10.0,
            ridge_weight=0.001,
            protection_rank=1,
        )
        new_refused_outputs = refused_outputs + refused_inputs @ weight_update.T
        new_answered_outputs = answered_outputs + answered_inputs @ weight_update.T
        original_distance = (
            torch.cdist(refused_outputs, answered_outputs).min(dim=1).values.mean()
        )
        updated_distance = (
            torch.cdist(new_refused_outputs, answered_outputs).min(dim=1).values.mean()
        )

        self.assertLess(updated_distance.item(), original_distance.item())
        torch.testing.assert_close(new_answered_outputs, answered_outputs)

    def test_selective_output_update_is_low_rank(self):
        answered_inputs = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
        answered_outputs = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
        refused_inputs = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
        refused_outputs = torch.tensor([[1.0, 0.0], [2.0, 0.0]])

        weight_update = fit_selective_output_update(
            answered_inputs,
            answered_outputs,
            refused_inputs,
            refused_outputs,
            neighbor_count=1,
            rank=1,
            strength=1.0,
            preservation_weight=1.0,
            ridge_weight=0.01,
            protection_rank=1,
        )

        self.assertLessEqual(torch.linalg.matrix_rank(weight_update).item(), 1)

    def test_selective_output_update_accepts_full_search_strength(self):
        answered_inputs = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
        answered_outputs = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
        refused_inputs = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
        refused_outputs = torch.tensor([[1.0, 0.0], [2.0, 0.0]])

        update = fit_selective_output_update(
            answered_inputs,
            answered_outputs,
            refused_inputs,
            refused_outputs,
            neighbor_count=1,
            rank=1,
            strength=3.0,
            preservation_weight=1.0,
            ridge_weight=0.01,
            protection_rank=0,
        )

        self.assertTrue(torch.isfinite(update).all())

    def test_protected_wall_rewire_ablates_and_redirects(self):
        answered_outputs = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
        refused_outputs = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
        weight = torch.eye(2)

        weight_update = fit_protected_wall_rewire(
            answered_outputs,
            refused_outputs,
            weight,
            neighbor_count=1,
            rank=1,
            ablation_strength=1.0,
            rewire_strength=1.0,
            protection_rank=1,
        )
        new_outputs = refused_outputs @ (weight + weight_update).T
        original_distance = (
            torch.cdist(refused_outputs, answered_outputs).min(dim=1).values.mean()
        )
        updated_distance = (
            torch.cdist(new_outputs, answered_outputs).min(dim=1).values.mean()
        )

        self.assertLess(updated_distance.item(), original_distance.item())

    def test_wall_ablation_accepts_old_school_overcorrection(self):
        answered_outputs = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
        refused_outputs = torch.tensor([[1.0, 0.0], [2.0, 0.0]])

        update = fit_protected_wall_rewire(
            answered_outputs,
            refused_outputs,
            torch.eye(2),
            neighbor_count=1,
            rank=1,
            ablation_strength=2.0,
            rewire_strength=0.0,
            protection_rank=0,
        )

        self.assertTrue(torch.isfinite(update).all())
        self.assertLessEqual(torch.linalg.matrix_rank(update).item(), 1)

    def test_open_queries_are_preferred_over_refusal_neighborhood(self):
        parameters = AQUAParameters(
            start_layer_index=0,
            end_layer_index=1,
            preserve_answered_weight=0.0,
            align_refused_weight=0.0,
            overcorrect_relative_weight=0.0,
            openness_weight=1.0,
            openness_margin=0.2,
            answered_geometry_weight=0.0,
            update_norm_weight=0.0,
            neighbor_count=1,
        )
        answered = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
        refused = torch.tensor([[10.0, 0.0], [10.0, 1.0]])
        still_refused = refused.clone()
        opened = torch.tensor([[0.1, 0.0], [0.1, 1.0]])
        update = torch.zeros((2, 2))

        refused_loss = aqua_query_loss(
            answered,
            answered,
            still_refused,
            refused,
            update,
            parameters,
        )
        opened_loss = aqua_query_loss(
            answered,
            answered,
            opened,
            refused,
            update,
            parameters,
        )

        self.assertGreater(refused_loss.item(), opened_loss.item())
        self.assertEqual(opened_loss.item(), 0.0)

    def test_answered_geometry_change_is_penalized(self):
        parameters = AQUAParameters(
            start_layer_index=0,
            end_layer_index=1,
            preserve_answered_weight=0.0,
            align_refused_weight=0.0,
            overcorrect_relative_weight=0.0,
            openness_weight=0.0,
            openness_margin=0.0,
            answered_geometry_weight=1.0,
            update_norm_weight=0.0,
            neighbor_count=1,
        )
        answered = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        changed = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        refused = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        update = torch.zeros((2, 2))

        preserved_loss = aqua_query_loss(
            answered,
            answered,
            refused,
            refused,
            update,
            parameters,
        )
        changed_loss = aqua_query_loss(
            changed,
            answered,
            refused,
            refused,
            update,
            parameters,
        )

        self.assertEqual(preserved_loss.item(), 0.0)
        self.assertGreater(changed_loss.item(), preserved_loss.item())


if __name__ == "__main__":
    unittest.main()
