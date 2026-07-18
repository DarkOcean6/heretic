# SPDX-License-Identifier: AGPL-3.0-or-later

import unittest

import numpy as np

from heretic.aqua import (
    AQUAParameters,
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
        self.assertEqual(self.parameters.neighbor_count, 2)


if __name__ == "__main__":
    unittest.main()
