import os
import tempfile
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from sage.all import Partition, QQ

from composition_factors import (
    CompositionFactors,
    CompositionFactorsAtDegree,
    Decomposition,
    Instruction,
    InstructionManual,
    Lie,
    MuPartition,
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

    def test_structural_zero_and_invalid_queries(self):
        self.assertEqual(self.cf.coeff([1], [2]), 0)
        for lam, mu in (([4], [1]), ([1], [4]), ([], [1]), ([1], [])):
            with self.subTest(lam=lam, mu=mu):
                with self.assertRaisesRegex(ValueError, "partition|computed"):
                    self.cf.coeff(lam, mu)

    def test_positive_integer_degrees(self):
        for constructor in (CompositionFactors, CompositionFactorsAtDegree, Lie):
            for degree in (0, -1, 1.5, True):
                with self.subTest(constructor=constructor.__name__, degree=degree):
                    with self.assertRaises((TypeError, ValueError)):
                        constructor(degree)

    def test_matrix_is_lazy_and_save_does_not_construct_it(self):
        cf = CompositionFactors(3)
        self.assertIsNone(cf._matrix)
        self.assertEqual(cf.coeff([2, 1], [1]), 1)
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / "lazy")
            cf.save(filename)
            restored = CompositionFactors.load(filename)
        self.assertIsNone(cf._matrix)
        self.assertIsNone(restored._matrix)
        np.testing.assert_array_equal(restored.matrix, self.cf.matrix)
        self.assertIs(restored.matrix, restored.matrix)

    def test_cached_and_uncached_assemblies_preserve_multiplicities(self):
        cached = CompositionFactors(8)
        uncached = CompositionFactors(8, cache_assemblies=False)
        self.assertEqual(cached.coeffs, uncached.coeffs)
        self.assertLess(len(set(Lie(8).as_list)), len(Lie(8).as_list))

    def test_degree_ten_exact_regression_hash(self):
        import hashlib
        from compute import encode_coefficients

        _, encoded = encode_coefficients(CompositionFactors(10))
        self.assertEqual(
            hashlib.sha256(encoded).hexdigest(),
            "1813638dbb0ae6385ffb911cbb0b12b0071f604c6a8d03a55534f5e67ed3d504",
        )

    def test_requested_decomposition_shape_is_lazy_and_order_preserving(self):
        from sage.all import Partitions

        mu = MuPartition([3, 2, 1])
        shape = Partition([2, 2, 1, 1])
        expected = []
        for combo in product(*(Partitions(size, outer=mu.partition) for size in shape)):
            ordered = tuple(sorted(combo, key=lambda part: (sum(part), len(part)), reverse=True))
            if ordered not in expected:
                expected.append(ordered)
        actual = mu.decompositions_for(shape)
        self.assertEqual([item.partitions_tuple for item in actual], expected)
        self.assertEqual(set(mu._decompositions), {shape})
        self.assertIs(actual, mu.decompositions_for(shape))
        self.assertEqual(set(mu.decompositions), set(mu.shapes))

    def test_equal_size_and_length_partitions_have_one_decomposition_order(self):
        mu = MuPartition([5, 3])
        first, second = Partition([3, 1]), Partition([2, 2])
        decompositions = mu.decompositions_for(Partition([4, 4]))
        unordered = [tuple(sorted(item.partitions_tuple, reverse=True)) for item in decompositions]
        self.assertEqual(len(unordered), len(set(unordered)))
        matching = [item for item in decompositions if set(item) == {first, second}]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0].partitions_tuple, (first, second))
        self.assertEqual(matching[0].overcount, 1)
        repeated = [item for item in decompositions if tuple(item) == (first, first)]
        self.assertEqual(len(repeated), 1)
        self.assertEqual(repeated[0].overcount, QQ(1) / 2)

    def test_tied_decomposition_assembly_counts_each_assignment_once(self):
        mu = MuPartition([5, 3])
        first, second = Partition([3, 1]), Partition([2, 2])
        lie = Lie(2)
        manual = InstructionManual(Partition([4, 4]), lie.sizes, 12)
        actual = sum(
            mu.lr_coeff(item) * item.overcount * manual.assemble(item, lie)
            for item in mu.decompositions_for(Partition([4, 4]))
            if set(item) == {first, second}
        )
        expected = s(first) * s(second).plethysm(s([1, 1])) + s(second) * s(first).plethysm(
            s([1, 1])
        )
        self.assertEqual(actual, expected)

    def test_one_lie_object_is_shared_by_degree_and_lookup(self):
        with patch("composition_factors.Lie._lie", wraps=Lie._lie) as character:
            degree = CompositionFactorsAtDegree(6)
        self.assertIs(degree.lie, degree.lookup.lie)
        self.assertEqual(character.call_count, 6)
        self.assertEqual(degree.lie.sizes_up_to(0), [])
        self.assertEqual(degree.lie.count_up_to(4), 5)

    def test_progress_counts_completed_mu_partitions(self):
        events = []
        CompositionFactorsAtDegree(4, progress=events.append)
        completed = [event for event in events if event["stage"] == "coefficients"]
        self.assertEqual([event["completed"] for event in completed], list(range(1, 7)))
        self.assertTrue(all(event["total"] == 6 for event in completed))

    def test_ticks_are_hidden_or_grouped_by_degree(self):
        for ticks in (False, "degree"):
            figure = self.cf.display(ticks=ticks, width=3)
            try:
                axis = figure.axes[0]
                if ticks is False:
                    self.assertEqual(len(axis.get_xticks()), 0)
                    self.assertEqual(len(axis.get_yticks()), 0)
                else:
                    self.assertEqual(
                        [label.get_text() for label in axis.get_xticklabels()], ["1", "2", "3"]
                    )
            finally:
                plt.close(figure)

    def test_palette_normalization_and_plot_input_validation(self):
        from matplotlib.colors import LogNorm

        before = self.cf.matrix.copy()
        figure = self.cf.display(cmap="historical", normalization="log", width=3)
        try:
            image = figure.axes[0].images[0]
            self.assertEqual(image.cmap.N, 256)
            self.assertIsInstance(image.norm, LogNorm)
        finally:
            plt.close(figure)
        np.testing.assert_array_equal(self.cf.matrix, before)
        for kwargs in (
            {"ticks": "invalid"},
            {"normalization": "invalid"},
            {"resolution": 0},
            {"width": -1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self.cf.display(**kwargs)

    def test_unique_stability_sequences_keep_earliest_eligible_chain(self):
        vis = Visualisations(self.cf)
        sequences = vis.stability_sequences("PD", unique=True)
        diagonal = [item for item in sequences if item["lambda"] == (1,) and item["mu"] == (1,)]
        self.assertEqual(len(diagonal), 1)
        self.assertEqual(diagonal[0]["sizes"], [2, 4, 6])
        self.assertEqual(diagonal[0]["coefficients"], [1, 1, 1])
        self.assertFalse(any(item["lambda"] == (2,) and item["mu"] == (2,) for item in sequences))
        figure = vis.PD_stability(unique=True, width=3)
        try:
            self.assertEqual(len(figure.axes[0].lines), len(sequences))
        finally:
            plt.close(figure)
        with self.assertRaises(ValueError):
            vis.stability_sequences("unknown")
        with self.assertRaises(ValueError):
            vis.stability_sequences(seq_length=0)


if __name__ == "__main__":
    unittest.main()
