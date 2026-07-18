# SPDX-License-Identifier: AGPL-3.0-or-later

import unittest

import numpy as np
import torch

from heretic.aqua import (
    AQUAParameters,
    aqua_query_loss,
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
            update_norm_weight=0.25,
            neighbor_count=2,
        )

    def test_query_sets_do_not_need_to_be_paired(self):
        answered = np.zeros((4, 3))
        refused = np.zeros((7, 3))

        validate_query_sets(answered, refused, neighbor_count=2)

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
