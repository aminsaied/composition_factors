"""Exact, type-selected Whitney calculations; these cases require SageMath."""

import csv
from itertools import permutations, product
import json
from math import factorial, prod
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from sage.all import Integer, Partition

from coefficient_data import _specht_dimension, partitions
from composition_factors import CompositionFactors
from whitney_calculation import WhitneyCalculator
from whitney_data import conjugate_partition


def set_partitions(n):
    if n == 0:
        yield ()
        return
    for part in set_partitions(n - 1):
        yield part + ((n - 1,),)
        for index, block in enumerate(part):
            yield part[:index] + (block + (n - 1,),) + part[index + 1 :]


def permute(g, part, ordered=False):
    image = tuple(tuple(sorted(g[x] for x in block)) for block in part)
    return image if ordered else tuple(sorted(image))


def coefficient_map(result):
    return {
        (tuple(row["lambda"]), tuple(row.get("mu", []))): row["multiplicity"]
        for row in result["coefficients"]
    }


class WhitneyCalculationTests(unittest.TestCase):
    def setUp(self):
        self.calculator = WhitneyCalculator()

    def test_native_sage_inputs_produce_portable_integers(self):
        expected = self.calculator.decomposition(4, 2, block_type=[2, 2], extended=True)
        result = self.calculator.decomposition(
            Integer(4), Integer(2), block_type=Partition([2, 2]), extended=True
        )
        self.assertEqual(result, expected)
        self.assertIs(type(result["degree"]), int)
        self.assertIs(type(result["blocks"]), int)
        self.assertTrue(all(type(part) is int for part in result["block_type"]))
        self.assertTrue(all(type(row["multiplicity"]) is int for row in result["coefficients"]))

    def test_ordinary_type_selected_examples(self):
        triples = self.calculator.decomposition(4, 2, block_type=[3, 1], nonzero=True)
        pairs = self.calculator.decomposition(4, 2, block_type=[2, 2], nonzero=True)
        self.assertEqual(
            coefficient_map(triples), {((3, 1), ()): 1, ((2, 2), ()): 1, ((2, 1, 1), ()): 1}
        )
        self.assertEqual(coefficient_map(pairs), {((3, 1), ()): 1})
        self.assertEqual(triples["block_type"], [3, 1])
        self.assertEqual(triples["calculation"], "signed_plethysm")
        self.assertEqual(triples["source"]["algorithm_version"], 1)
        self.assertEqual(len(triples["source"]["source_sha256"]), 64)
        self.assertNotIn("coefficient_sha256", triples["source"])

    def test_extended_even_block_conjugation(self):
        result = self.calculator.decomposition(4, 2, block_type=[2, 2], extended=True, nonzero=True)
        self.assertEqual(
            coefficient_map(result),
            {((3, 1), (2,)): 1, ((4,), (1, 1)): 1, ((2, 2), (1, 1)): 1},
        )
        self.assertEqual(
            len(
                self.calculator.decomposition(4, 2, block_type=[2, 2], extended=True)[
                    "coefficients"
                ]
            ),
            10,
        )

    def test_totals_agree_with_composition_factors_through_six(self):
        source = CompositionFactors(6)
        for degree in range(2, 7):
            for blocks in range(1, degree):
                result = self.calculator.decomposition(degree, blocks, extended=True)
                self.assertIsNone(result["block_type"])
                for row in result["coefficients"]:
                    with self.subTest(degree=degree, blocks=blocks, row=row):
                        self.assertEqual(
                            row["multiplicity"],
                            source.coeff(
                                list(conjugate_partition(row["lambda"])),
                                list(conjugate_partition(row["mu"])),
                            ),
                        )

    def test_type_sums_dimensions_and_block_invariants(self):
        for degree in range(2, 7):
            for blocks in range(1, degree):
                total = coefficient_map(
                    self.calculator.decomposition(degree, blocks, extended=True)
                )
                accumulated = dict.fromkeys(total, 0)
                for profile in (part for part in partitions(degree) if len(part) == blocks):
                    ordinary = coefficient_map(
                        self.calculator.decomposition(degree, blocks, block_type=profile)
                    )
                    extended = coefficient_map(
                        self.calculator.decomposition(
                            degree, blocks, block_type=profile, extended=True
                        )
                    )
                    expected = factorial(degree) // prod(
                        size ** profile.count(size) * factorial(profile.count(size))
                        for size in set(profile)
                    )
                    self.assertEqual(
                        sum(value * _specht_dimension(lam) for (lam, _), value in ordinary.items()),
                        expected,
                    )
                    self.assertEqual(
                        sum(
                            value * _specht_dimension(lam) * _specht_dimension(mu)
                            for (lam, mu), value in extended.items()
                        ),
                        factorial(blocks) * expected,
                    )
                    for (lam, mu), value in extended.items():
                        accumulated[lam, mu] += value
                        if mu == (blocks,):
                            self.assertEqual(value, ordinary[lam, ()])
                self.assertEqual(accumulated, total)

    def test_actual_rank_two_interval_characters_for_each_type(self):
        lattice = list(set_partitions(4))
        atoms = [part for part in lattice if len(part) == 3]
        pairings = [part for part in lattice if tuple(map(len, part)) == (2, 2)]
        labels = [(4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1)]
        for profile in ((3, 1), (2, 2)):
            targets = [
                ordering
                for part in lattice
                if tuple(sorted(map(len, part), reverse=True)) == profile
                for ordering in permutations(part)
            ]
            incidences = [
                (atom, target)
                for atom, target in product(atoms, targets)
                if all(any(set(block) <= set(outer) for outer in target) for block in atom)
            ]
            coefficients = coefficient_map(
                self.calculator.decomposition(4, 2, block_type=profile, extended=True)
            )
            for g in permutations(range(4)):
                sign = (-1) ** sum(g[i] > g[j] for i in range(4) for j in range(i + 1, 4))
                standard = sum(i == g[i] for i in range(4)) - 1
                two_dimensional = sum(permute(g, part) == part for part in pairings) - 1
                characters = dict(
                    zip(labels, (1, standard, two_dimensional, sign * standard, sign))
                )
                for swap in (False, True):

                    def fixed(part):
                        image = permute(g, part, ordered=True)
                        return (image[::-1] if swap else image) == part

                    actual = sum(
                        permute(g, atom) == atom and fixed(target) for atom, target in incidences
                    ) - sum(fixed(target) for target in targets)
                    expected = sum(
                        value * characters[lam] * (-1 if swap and mu == (1, 1) else 1)
                        for (lam, mu), value in coefficients.items()
                    )
                    self.assertEqual(actual, expected, (profile, g, swap))

    def test_invalid_requests_and_exports_are_rejected(self):
        invalid = [
            (0, 1, None),
            (4, 0, None),
            (4, 4, None),
            (True, 1, None),
            (4, 2.0, None),
            (4, 2, []),
            (4, 2, [1, 3]),
            (4, 2, [True, 3]),
            (4, 2, [3.0, 1]),
            (4, 2, [3, 2]),
            (4, 2, [4]),
            (4, 2, "3,1"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid.json"
            for degree, blocks, profile in invalid:
                with self.subTest(degree=degree, blocks=blocks, profile=profile):
                    with self.assertRaises(ValueError):
                        self.calculator.export_json(output, degree, blocks, block_type=profile)
                    self.assertFalse(output.exists())

    def test_exports_preserve_exact_records_type_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            for profile in (None, [2, 2]):
                expected = self.calculator.decomposition(
                    4, 2, block_type=profile, extended=True, nonzero=True
                )
                for extension in ("json", "csv"):
                    output = Path(directory) / f"{profile}.{extension}"
                    exporter = getattr(self.calculator, f"export_{extension}")
                    exporter(output, 4, 2, block_type=profile, extended=True, nonzero=True)
                    if extension == "json":
                        self.assertEqual(json.loads(output.read_text()), expected)
                    else:
                        with output.open(newline="") as stream:
                            rows = list(csv.DictReader(stream))
                        self.assertEqual(len(rows), len(expected["coefficients"]))
                        for row, record in zip(rows, expected["coefficients"]):
                            self.assertEqual(row["block_type"], "" if profile is None else "2,2")
                            self.assertEqual(row["calculation"], "signed_plethysm")
                            self.assertEqual(
                                row["source_sha256"], expected["source"]["source_sha256"]
                            )
                            self.assertEqual(int(row["multiplicity"]), record["multiplicity"])
                            self.assertEqual(row["lambda"], ",".join(map(str, record["lambda"])))
                            self.assertEqual(row["mu"], ",".join(map(str, record["mu"])))
                    before = output.read_bytes()
                    with self.assertRaises(FileExistsError):
                        exporter(output, 4, 2)
                    self.assertEqual(output.read_bytes(), before)

    def test_cli_computes_without_a_checkpoint(self):
        script = Path(__file__).resolve().parents[1] / "compute.py"
        with tempfile.TemporaryDirectory() as directory:
            args = [
                sys.executable,
                str(script),
                "whitney",
                "compute",
                "--degree",
                "4",
                "--blocks",
                "2",
                "--type",
                "2,2",
                "--extended",
                "--nonzero",
            ]
            result = subprocess.run(args, cwd=directory, capture_output=True, text=True, check=True)
            expected = self.calculator.decomposition(
                4, 2, block_type=[2, 2], extended=True, nonzero=True
            )
            self.assertEqual(json.loads(result.stdout), expected)
            for extension in ("json", "csv"):
                output = Path(directory) / f"result.{extension}"
                command = args + ["--output", str(output), "--format", extension]
                subprocess.run(command, cwd=directory, capture_output=True, text=True, check=True)
                before = output.read_bytes()
                failed = subprocess.run(command, cwd=directory, capture_output=True, text=True)
                self.assertNotEqual(failed.returncode, 0)
                self.assertIn("error:", failed.stderr)
                self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
