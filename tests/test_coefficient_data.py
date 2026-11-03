"""Exact-data reader and CLI operations that do not require Sage."""

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from coefficient_data import read_checkpoint


class CoefficientDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        parts = [(1,), (2,), (1, 1), (3,), (2, 1), (1, 1, 1)]
        matrix = [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [1, 1, 1, 0, 1, 0],
            [0, 1, 1, 0, 0, 1],
        ]
        self.records = [
            {"lambda": list(lam), "mu": list(mu), "multiplicity": matrix[i][j]}
            for i, lam in enumerate(parts)
            for j, mu in enumerate(parts)
            if sum(mu) <= sum(lam)
        ]
        self.report = {
            "format_version": 1,
            "algorithm_version": 1,
            "degree": 3,
            "source_sha256": "a" * 64,
            "save_load_equal": True,
            "reference_status": "not_requested",
        }
        self.write_checkpoint()

    def write_checkpoint(self):
        records = sorted(self.records, key=lambda row: (row["lambda"], row["mu"]))
        encoded = json.dumps(records, separators=(",", ":")).encode()
        self.report["coefficient_sha256"] = hashlib.sha256(encoded).hexdigest()
        (self.root / "coefficients.json").write_bytes(encoded + b"\n")
        (self.root / "report.json").write_text(json.dumps(self.report))

    def test_exact_queries_and_legacy_format(self):
        for legacy in (False, True):
            if legacy:
                self.report.pop("format_version")
                self.report.pop("algorithm_version")
                self.write_checkpoint()
            data = read_checkpoint(self.root)
            self.assertEqual(data.degree, 3)
            self.assertEqual(data.coeff([2, 1], [1]), 1)
            self.assertEqual(data.coeff([1], [2]), 0)
            self.assertEqual(len(list(data.records())), 25)
            with self.assertRaises(ValueError):
                data.coeff([4], [1])

    def test_rejects_hash_mismatch(self):
        self.report["coefficient_sha256"] = "0" * 64
        (self.root / "report.json").write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError, "hash"):
            read_checkpoint(self.root)

    def test_rejects_duplicate_missing_and_invalid_records_even_with_matching_hash(self):
        original = list(self.records)
        variants = [
            original + [original[0]],
            original[1:],
            [{**original[0], "multiplicity": -1}, *original[1:]],
            [{**original[0], "multiplicity": True}, *original[1:]],
            [{**original[0], "multiplicity": 1.0}, *original[1:]],
            [{**original[0], "lambda": [1, 2]}, *original[1:]],
            [{**original[0], "lambda": []}, *original[1:]],
            [{**original[0], "mu": [4]}, *original[1:]],
            [{**original[0], "multiplicity": 2}, *original[1:]],
        ]
        for records in variants:
            with self.subTest(first=records[0]):
                self.records = records
                self.write_checkpoint()
                with self.assertRaises(ValueError):
                    read_checkpoint(self.root)

    def test_rejects_unknown_format_failed_reference_and_invalid_degree(self):
        for key, value in (
            ("format_version", 99),
            ("reference_status", "failed"),
            ("degree", True),
        ):
            with self.subTest(key=key):
                original = self.report[key]
                self.report[key] = value
                self.write_checkpoint()
                with self.assertRaises(ValueError):
                    read_checkpoint(self.root)
                self.report[key] = original

    def test_versioned_reports_require_algorithm_identity_and_exact_comparison_scope(self):
        original = dict(self.report)
        variants = [
            {key: value for key, value in original.items() if key != "algorithm_version"},
            {**original, "algorithm_version": True},
            {**original, "reference_status": "passed"},
            {
                **original,
                "reference_status": "passed",
                "direct_plethysm_max_mu_size": 1,
                "direct_plethysm_pairs_checked": 5,
            },
        ]
        for report in variants:
            with self.subTest(report=report):
                self.report = report
                self.write_checkpoint()
                with self.assertRaises(ValueError):
                    read_checkpoint(self.root)

    def run_cli_without_site_packages(self, *args):
        script = Path(__file__).resolve().parents[1] / "compute.py"
        return subprocess.run(
            [sys.executable, "-S", str(script), *args],
            check=True,
            capture_output=True,
            text=True,
            cwd=self.root,
        ).stdout

    def test_help_inspect_and_query_do_not_import_sage(self):
        self.assertIn("inspect", self.run_cli_without_site_packages("--help"))
        summary = json.loads(self.run_cli_without_site_packages("inspect", str(self.root)))
        self.assertEqual(summary["stored_coefficient_count"], 25)
        self.assertEqual(
            self.run_cli_without_site_packages(
                "query", str(self.root), "--lambda", "1", "--mu", "2"
            ),
            "0\n",
        )

    def test_exact_csv_export_and_overwrite_protection(self):
        output = self.root / "coefficients.csv"
        self.run_cli_without_site_packages(
            "export", str(self.root), "--output", str(output), "--nonzero"
        )
        with output.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(rows)
        self.assertTrue(all(int(row["multiplicity"]) > 0 for row in rows))
        original = output.read_bytes()
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_cli_without_site_packages("export", str(self.root), "--output", str(output))
        self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
