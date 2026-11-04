"""Read-only, positive-rank Whitney views; no Sage or published data required."""

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from coefficient_data import encode_records, partitions, read_checkpoint
from whitney_data import WhitneyData, conjugate_partition


class WhitneyDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        parts = [part for degree in range(1, 5) for part in partitions(degree)]
        # The degree-four matrix from CompositionFactors' documented example.
        matrix = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
            [0, 2, 1, 0, 1, 1, 0, 0, 1, 0, 0],
            [1, 1, 2, 1, 2, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        ]
        records = [
            {"lambda": list(lam), "mu": list(mu), "multiplicity": matrix[i][j]}
            for i, lam in enumerate(parts)
            for j, mu in enumerate(parts)
            if sum(mu) <= sum(lam)
        ]
        encoded = encode_records(records)
        self.report = {
            "format_version": 1,
            "algorithm_version": 2,
            "degree": 4,
            "source_sha256": "a" * 64,
            "coefficient_sha256": hashlib.sha256(encoded).hexdigest(),
            "reference_status": "not_requested",
        }
        (self.root / "coefficients.json").write_bytes(encoded + b"\n")
        (self.root / "report.json").write_text(json.dumps(self.report))
        self.data = read_checkpoint(self.root)
        self.whitney = WhitneyData(self.data)

    def run_cli(self, *args, check=True):
        script = Path(__file__).resolve().parents[1] / "compute.py"
        return subprocess.run(
            [sys.executable, "-S", str(script), "whitney", *args],
            cwd=self.root,
            check=check,
            capture_output=True,
            text=True,
        )

    def test_partition_conjugation_is_an_involution(self):
        self.assertEqual(conjugate_partition([4, 2, 1]), (3, 2, 1, 1))
        for degree in range(1, 8):
            for part in partitions(degree):
                self.assertEqual(conjugate_partition(conjugate_partition(part)), part)
        for invalid in ([], [1, 2], [True], [1.0], [0], "2,1"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    conjugate_partition(invalid)

    def test_extended_queries_conjugate_both_labels_not_matrix_axes(self):
        self.assertEqual(self.whitney.extended_coeff([3, 1], [2]), 2)
        self.assertEqual(self.whitney.extended_coeff([3, 1], [1, 1]), 1)
        self.assertEqual(self.whitney.extended_coeff([2, 1, 1], [2]), 1)
        self.assertEqual(self.whitney.extended_coeff([1, 1, 1, 1], [2]), 0)

    def test_ordinary_queries_select_the_trivial_block_representation(self):
        for degree in range(2, 5):
            for blocks in range(1, degree):
                for lam in partitions(degree):
                    self.assertEqual(
                        self.whitney.ordinary_coeff(lam, blocks),
                        self.whitney.extended_coeff(lam, [blocks]),
                    )
        self.assertEqual(self.whitney.ordinary_coeff([2], 1), 1)
        self.assertEqual(self.whitney.ordinary_coeff([1, 1], 1), 0)

    def test_ordinary_decomposition_and_provenance(self):
        result = self.whitney.decomposition(4, 2, nonzero=True)
        self.assertEqual(result["kind"], "ordinary")
        self.assertEqual((result["degree"], result["blocks"], result["rank"]), (4, 2, 2))
        self.assertEqual(result["characteristic"], 0)
        self.assertEqual(result["convention"], "positive_rank")
        self.assertEqual(result["calculation"], "partition_conjugation")
        self.assertTrue(result["nonzero_only"])
        self.assertEqual(result["source"]["coefficient_sha256"], self.report["coefficient_sha256"])
        self.assertEqual(
            result["coefficients"],
            [
                {"lambda": [3, 1], "multiplicity": 2},
                {"lambda": [2, 2], "multiplicity": 1},
                {"lambda": [2, 1, 1], "multiplicity": 1},
            ],
        )
        self.assertEqual(len(self.whitney.decomposition(4, 2)["coefficients"]), 5)

    def test_extended_decomposition_has_all_pairs_including_zeros(self):
        result = self.whitney.decomposition(4, 2, extended=True)
        self.assertEqual(result["kind"], "extended")
        records = result["coefficients"]
        self.assertEqual(len(records), 10)
        self.assertEqual(
            {(tuple(row["lambda"]), tuple(row["mu"])) for row in records},
            {(lam, mu) for lam in partitions(4) for mu in partitions(2)},
        )
        for row in records:
            self.assertEqual(
                row["multiplicity"], self.whitney.extended_coeff(row["lambda"], row["mu"])
            )
        self.assertTrue(any(row["multiplicity"] == 0 for row in records))

    def test_degree_four_dimensions(self):
        dimensions = {(4,): 1, (3, 1): 3, (2, 2): 2, (2, 1, 1): 3, (1, 1, 1, 1): 1}
        for blocks, expected in ((1, 6), (2, 11), (3, 6)):
            result = self.whitney.decomposition(4, blocks)
            actual = sum(
                dimensions[tuple(row["lambda"])] * row["multiplicity"]
                for row in result["coefficients"]
            )
            self.assertEqual(actual, expected)

    def test_out_of_domain_and_uncomputed_inputs_are_errors(self):
        for degree, blocks in (
            (0, 0),
            (1, 1),
            (4, 4),
            (3, 4),
            (4, 0),
            (True, 1),
            (4, True),
            (4.0, 2),
            (4, 2.0),
            (5, 1),
        ):
            with self.subTest(degree=degree, blocks=blocks):
                with self.assertRaises(ValueError):
                    self.whitney.decomposition(degree, blocks)
        for lam, mu in (
            ([], [1]),
            ([1, 2], [1]),
            ([True], [1]),
            ([3], []),
            ([3], [3]),
            ([3], [4]),
            ([5], [1]),
        ):
            with self.subTest(lam=lam, mu=mu):
                with self.assertRaises(ValueError):
                    self.whitney.extended_coeff(lam, mu)
        for blocks in (0, True, 2.0, 3, 4):
            with self.subTest(blocks=blocks):
                with self.assertRaises(ValueError):
                    self.whitney.ordinary_coeff([3], blocks)

    def test_missing_source_data_does_not_become_a_zero(self):
        del self.data.coeffs[(2, 1, 1)][(1, 1)]
        with self.assertRaisesRegex(ValueError, "Missing coefficient"):
            self.whitney.ordinary_coeff([3, 1], 2)

    def test_json_and_csv_exports_are_exact_and_do_not_overwrite(self):
        for extended in (False, True):
            with self.subTest(extended=extended):
                expected = self.whitney.decomposition(4, 2, extended=extended, nonzero=True)
                for extension in ("json", "csv"):
                    output = self.root / f"{extended}.{extension}"
                    exporter = getattr(self.whitney, f"export_{extension}")
                    exporter(output, 4, 2, extended=extended, nonzero=True)
                    if extension == "json":
                        self.assertEqual(json.loads(output.read_text()), expected)
                    else:
                        with output.open(newline="") as stream:
                            rows = list(csv.DictReader(stream))
                        self.assertEqual(len(rows), len(expected["coefficients"]))
                        for row, record in zip(rows, expected["coefficients"]):
                            self.assertEqual(row["kind"], expected["kind"])
                            self.assertEqual(
                                (row["degree"], row["blocks"], row["rank"]), ("4", "2", "2")
                            )
                            self.assertEqual(row["lambda"], ",".join(map(str, record["lambda"])))
                            self.assertEqual(row["mu"], ",".join(map(str, record.get("mu", []))))
                            self.assertEqual(int(row["multiplicity"]), record["multiplicity"])
                            self.assertEqual(
                                row["source_coefficient_sha256"], self.report["coefficient_sha256"]
                            )
                    before = output.read_bytes()
                    with self.assertRaises(FileExistsError):
                        exporter(output, 4, 2)
                    self.assertEqual(output.read_bytes(), before)

    def test_invalid_export_does_not_create_a_file(self):
        for extension in ("json", "csv"):
            output = self.root / f"invalid.{extension}"
            with self.assertRaisesRegex(ValueError, "positive rank"):
                getattr(self.whitney, f"export_{extension}")(output, 4, 4)
            self.assertFalse(output.exists())

    def test_cli_queries_decompositions_and_exports_need_no_sage(self):
        self.assertIn("decompose", self.run_cli("--help").stdout)
        for selector in ("--blocks", "--mu"):
            self.assertEqual(
                self.run_cli("query", str(self.root), "--lambda", "3,1", selector, "2").stdout,
                "2\n",
            )
        args = ("--degree", "4", "--blocks", "2", "--extended", "--nonzero")
        expected = self.whitney.decomposition(4, 2, extended=True, nonzero=True)
        result = self.run_cli("decompose", str(self.root), *args)
        self.assertEqual(json.loads(result.stdout), expected)
        for extension in ("json", "csv"):
            output = self.root / f"cli.{extension}"
            self.run_cli(
                "export", str(self.root), *args, "--format", extension, "--output", str(output)
            )
            self.assertTrue(output.exists())
            if extension == "json":
                self.assertEqual(json.loads(output.read_text()), expected)
            else:
                with output.open(newline="") as stream:
                    self.assertEqual(
                        len(list(csv.DictReader(stream))), len(expected["coefficients"])
                    )

    def test_cli_errors_are_explicit_and_never_report_success(self):
        cases = [
            ("query", str(self.root), "--lambda", "3", "--blocks", "3"),
            ("query", str(self.root), "--lambda", "3", "--mu", "1,2"),
            ("query", str(self.root), "--lambda", "3", "--blocks", "1", "--mu", "1"),
            ("query", str(self.root), "--lambda", "3"),
            ("decompose", str(self.root), "--degree", "5", "--blocks", "1"),
            ("decompose", str(self.root), "--degree", "3", "--blocks", "0"),
        ]
        for args in cases:
            with self.subTest(args=args):
                result = self.run_cli(*args, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("error:", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_calculation_help_validation_and_missing_sage(self):
        self.assertIn("--type", self.run_cli("compute", "--help").stdout)
        result = self.run_cli("compute", "--degree", "4", "--blocks", "2", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("require SageMath", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        cases = [
            ("--degree", "4", "--blocks", "4"),
            ("--degree", "4", "--blocks", "2", "--type", "3,2"),
            ("--degree", "4", "--blocks", "2", "--type", "4"),
            ("--degree", "4", "--blocks", "2", "--format", "csv"),
            ("--degree", "4", "--blocks", "2", "--output", str(self.root / "unused.json")),
        ]
        for args in cases:
            result = self.run_cli("compute", *args, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("error:", result.stderr)
            self.assertNotIn("require SageMath", result.stderr)
        self.assertFalse((self.root / "unused.json").exists())

    def test_operations_preserve_the_source_and_reject_corruption(self):
        before = {
            name: (self.root / name).read_bytes() for name in ("coefficients.json", "report.json")
        }
        coefficients_before = encode_records(list(self.data.records()))
        self.whitney.decomposition(4, 2, extended=True)
        self.whitney.export_json(self.root / "result.json", 4, 2)
        self.whitney.export_csv(self.root / "result.csv", 4, 2, extended=True)
        self.run_cli("decompose", str(self.root), "--degree", "4", "--blocks", "2")
        self.assertEqual(encode_records(list(self.data.records())), coefficients_before)
        self.assertEqual({name: (self.root / name).read_bytes() for name in before}, before)
        self.report["coefficient_sha256"] = "0" * 64
        (self.root / "report.json").write_text(json.dumps(self.report))
        result = self.run_cli(
            "query", str(self.root), "--lambda", "3,1", "--blocks", "2", check=False
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("hash", result.stderr)


if __name__ == "__main__":
    unittest.main()
