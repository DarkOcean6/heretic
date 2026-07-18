# SPDX-License-Identifier: AGPL-3.0-or-later

import unittest

import numpy as np

from heretic.aqua import (
    AQUAParameters,
    build_query_targets,
    validate_paired_queries,
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
        )

    def test_targets_align_and_overcorrect_away_from_refusal(self):
        answered = np.array([[2.0, 4.0]])
        refused = np.array([[0.0, 2.0]])

        targets = build_query_targets(answered, refused, 0.5)

        np.testing.assert_allclose(targets, np.array([[3.0, 5.0]]))

    def test_pair_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "paired prompt datasets"):
            validate_paired_queries(np.zeros((2, 3)), np.zeros((3, 3)))

    def test_parameters_capture_merge_optimization_controls(self):
        self.assertEqual(self.parameters.start_layer_index, 1)
        self.assertEqual(self.parameters.end_layer_index, 3)
        self.assertEqual(self.parameters.overcorrect_relative_weight, 0.5)


if __name__ == "__main__":
    unittest.main()
