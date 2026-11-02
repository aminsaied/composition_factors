import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from sage.all import Partition, QQ

from composition_factors import (
    CompositionFactors,
    Decomposition,
    Instruction,
    Visualisations,
    s,
)


class CompositionFactorsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cf = CompositionFactors(3)

    def test_degree_three_matrix(self):
        np.testing.assert_array_equal(
            self.cf.matrix,
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [1, 1, 1, 0, 1, 0],
                [0, 1, 1, 0, 0, 1],
            ],
        )

    def test_readme_coefficient(self):
        cf = CompositionFactors(6)
        self.assertEqual(cf.coeff([4, 2], [1, 1, 1]), 2)

    def test_degree_three_against_direct_plethysm(self):
        lie = s([1]) + s([1, 1]) + s([2, 1])
        for mu in self.cf.coeffs:
            expected = s(mu).plethysm(lie).monomial_coefficients()
            for lambda_, row in self.cf.coeffs.items():
                self.assertEqual(row.get(mu, 0), expected.get(lambda_, 0))

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "coefficients")
            self.cf.save(path)
            restored = CompositionFactors.load(path)
        self.assertEqual(restored.top_degree, self.cf.top_degree)
        self.assertEqual(restored.coeffs, self.cf.coeffs)
        np.testing.assert_array_equal(restored.matrix, self.cf.matrix)

    def test_decomposition_iteration_and_exact_overcount(self):
        part = Partition([2])
        decomposition = Decomposition((part, part))
        self.assertEqual(list(decomposition), [part, part])
        self.assertEqual(decomposition.overcount, QQ(1) / 2)

    def test_instruction_copies_have_independent_steps(self):
        instruction = Instruction([2, 1])
        first = instruction.add_step_and_copy((0, 3))
        second = first.add_step_and_copy((1, 0))
        self.assertEqual(instruction.steps, {0: None, 1: None})
        self.assertEqual(first.steps, {0: 3, 1: None})
        self.assertEqual(second.steps, {0: 3, 1: 0})
        self.assertFalse(first.full())
        self.assertTrue(second.full())

    def test_plots_use_their_own_coefficients(self):
        before = self.cf.matrix.copy()
        vis = Visualisations(self.cf)
        for figure in (
            vis.display(width=3),
            vis.PD_stability(width=3),
            vis.FI_stability(width=3),
        ):
            plt.close(figure)
        np.testing.assert_array_equal(self.cf.matrix, before)

    def test_stability_includes_the_top_degree(self):
        vis = Visualisations(self.cf)
        one = Partition([1])
        self.assertEqual(vis._push_diagonal(one, one), ([2, 4, 6], [1, 1, 1]))
        self.assertEqual(vis._push_down(one, one), ([1, 2, 3], [1, 0, 0]))

    def test_stability_minimum_length_is_inclusive(self):
        figure = Visualisations(self.cf).PD_stability(seq_length=3, width=3)
        try:
            self.assertTrue(
                any(list(line.get_xdata()) == [2, 4, 6] for line in figure.axes[0].lines)
            )
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
