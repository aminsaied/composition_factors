import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

import compute
import composition_factors as module
from composition_factors import CompositionFactors


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source_sha256 = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()

    def make_checkpoint(self):
        directory = self.root / "checkpoint"
        compute.save_checkpoint(CompositionFactors(2), directory, 0.0, self.source_sha256)
        return directory

    def test_resume_matches_fresh_computation_and_preserves_checkpoint(self):
        first, second = self.root / "first", self.root / "second"
        compute.main(["--degree", "2", "--output", str(first)])
        checkpoint = first / "degree2"
        original = (checkpoint / "composition_factors.sobj").read_bytes()
        compute.main(
            [
                "--degree",
                "4",
                "--output",
                str(second),
                "--resume",
                str(checkpoint),
                "--reference",
            ]
        )
        self.assertTrue((first / "degree1" / "report.json").is_file())
        self.assertTrue((second / "degree3" / "report.json").is_file())
        self.assertFalse((second / "degree2").exists())
        restored = compute.load_checkpoint(second / "degree4", self.source_sha256)
        expected = CompositionFactors(4)
        self.assertEqual(restored.coeffs, expected.coeffs)
        np.testing.assert_array_equal(restored.matrix, expected.matrix)
        self.assertEqual((checkpoint / "composition_factors.sobj").read_bytes(), original)

    def test_checkpoint_rejects_changed_source_or_coefficient_hash(self):
        directory = self.make_checkpoint()
        path = directory / "report.json"
        original = json.loads(path.read_text())
        for field in ("source_sha256", "coefficient_sha256"):
            with self.subTest(field=field):
                report = {**original, field: "0" * 64}
                path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    compute.load_checkpoint(directory, self.source_sha256)

    def test_existing_output_is_not_overwritten(self):
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "keep"
        sentinel.write_text("original")
        with self.assertRaises(FileExistsError):
            compute.main(["--degree", "1", "--output", str(output)])
        self.assertEqual(sentinel.read_text(), "original")
        self.assertEqual(list(output.iterdir()), [sentinel])

    def test_invalid_resume_target_does_not_create_output(self):
        checkpoint = self.make_checkpoint()
        output = self.root / "invalid"
        with self.assertRaises(SystemExit):
            compute.main(
                [
                    "--degree",
                    "2",
                    "--resume",
                    str(checkpoint),
                    "--output",
                    str(output),
                ]
            )
        self.assertFalse(output.exists())

    def test_failed_comparison_is_recorded_and_blocks_resume(self):
        output = self.root / "failed"
        with patch("compute.compare_direct_plethysm", side_effect=ValueError("mismatch")):
            with self.assertRaisesRegex(ValueError, "mismatch"):
                compute.main(
                    [
                        "--degree",
                        "1",
                        "--output",
                        str(output),
                        "--reference",
                    ]
                )
        checkpoint = output / "degree1"
        report = json.loads((checkpoint / "report.json").read_text())
        self.assertEqual(report["reference_status"], "failed")
        with self.assertRaisesRegex(ValueError, "failed"):
            compute.load_checkpoint(checkpoint, self.source_sha256)

    def test_independent_comparison_reports_its_exact_scope(self):
        cf = CompositionFactors(3)
        report = compute.compare_direct_plethysm(cf, max_mu_size=1)
        self.assertEqual(report["direct_plethysm_pairs_checked"], len(cf.coeffs))
        self.assertEqual(report["direct_plethysm_max_mu_size"], 1)
        self.assertEqual(report["reference_status"], "passed")
        full = compute.compare_direct_plethysm(cf)
        self.assertEqual(full["direct_plethysm_pairs_checked"], len(cf.coeffs) ** 2)
        self.assertEqual(full["direct_plethysm_max_mu_size"], 3)

    def test_checkpoint_does_not_force_visualization_matrix(self):
        cf = CompositionFactors(3)
        directory = self.root / "lazy"
        report = compute.save_checkpoint(cf, directory, 0.0, self.source_sha256)
        self.assertIsNone(cf._matrix)
        self.assertEqual(report["matrix_entry_count"], 36)
        self.assertEqual(report["format_version"], 1)
        self.assertEqual(report["algorithm_version"], 2)
        self.assertEqual(report["dimension_columns_checked"], 10)

    def test_checkpoint_rejects_overcount_before_writing_files(self):
        from sage.all import Partition

        cf = CompositionFactors(3)
        cf.coeffs[Partition([2, 1])][Partition([1, 1])] += 1
        directory = self.root / "invalid-dimensions"
        with self.assertRaisesRegex(ValueError, "dimension"):
            compute.save_checkpoint(cf, directory, 0.0, self.source_sha256)
        self.assertFalse(directory.exists())

    def test_checkpoint_checks_json_and_pickle_agree(self):
        directory = self.make_checkpoint()
        path = directory / "coefficients.json"
        records = json.loads(path.read_text())
        records[0]["multiplicity"] += 1
        path.write_text(json.dumps(records))
        with self.assertRaises(ValueError):
            compute.load_checkpoint(directory, self.source_sha256)

    def test_resume_rejects_unknown_algorithm_even_if_source_hash_matches(self):
        directory = self.make_checkpoint()
        path = directory / "report.json"
        report = json.loads(path.read_text())
        report["algorithm_version"] = 99
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "algorithm"):
            compute.load_checkpoint(directory, self.source_sha256)

    def test_explicit_migration_preserves_data_and_original_checkpoint(self):
        directory = self.make_checkpoint()
        path = directory / "report.json"
        report = json.loads(path.read_text())
        report["source_sha256"] = "a" * 64
        path.write_text(json.dumps(report))
        original = path.read_bytes()
        output = self.root / "migrated"
        compute.main(
            [
                "migrate",
                str(directory),
                "--output",
                str(output),
                "--reference-max-mu-size",
                "2",
            ]
        )
        restored = compute.load_checkpoint(output, self.source_sha256)
        self.assertEqual(restored.coeffs, CompositionFactors(2).coeffs)
        self.assertEqual(path.read_bytes(), original)
        migrated = json.loads((output / "report.json").read_text())
        self.assertEqual(migrated["coefficient_sha256"], report["coefficient_sha256"])
        self.assertEqual(migrated["migration"]["source_sha256"], "a" * 64)
        self.assertEqual(migrated["direct_plethysm_pairs_checked"], 9)

    def test_migration_accepts_version_one_only_below_the_first_affected_degree(self):
        from coefficient_data import CoefficientData, read_checkpoint

        directory = self.make_checkpoint()
        path = directory / "report.json"
        report = json.loads(path.read_text())
        report["algorithm_version"] = 1
        path.write_text(json.dumps(report))
        compute.migrate_checkpoint(read_checkpoint(directory), self.root / "migrated-v1", 2)
        for algorithm in (None, 1):
            with self.subTest(algorithm=algorithm):
                old = CoefficientData(
                    12,
                    {},
                    {
                        "algorithm_version": algorithm,
                        "source_sha256": compute.LEGACY_SOURCE_SHA256,
                    },
                )
                with self.assertRaisesRegex(ValueError, "degree 12"):
                    compute.migrate_checkpoint(old, self.root / "unsafe", 4)
                self.assertFalse((self.root / "unsafe").exists())

    def test_migration_rejects_unknown_algorithms_and_unknown_legacy_sources(self):
        directory = self.make_checkpoint()
        path = directory / "report.json"
        original = json.loads(path.read_text())
        reports = [
            {**original, "algorithm_version": 99},
            {
                key: ("a" * 64 if key == "source_sha256" else value)
                for key, value in original.items()
                if key not in ("algorithm_version", "format_version")
            },
        ]
        for report in reports:
            with self.subTest(report=report):
                path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    compute.main(["migrate", str(directory), "--output", str(self.root / "no")])
                self.assertFalse((self.root / "no").exists())

    def test_compare_existing_checkpoint_does_not_modify_it(self):
        directory = self.make_checkpoint()
        before = {path.name: path.read_bytes() for path in directory.iterdir()}
        compute.main(["compare", str(directory), "--reference-max-mu-size", "1"])
        after = {path.name: path.read_bytes() for path in directory.iterdir()}
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
