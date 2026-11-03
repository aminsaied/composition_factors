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
        compute.save_checkpoint(
            CompositionFactors(2), directory, 0.0, self.source_sha256
        )
        return directory

    def test_resume_matches_fresh_computation_and_preserves_checkpoint(self):
        first, second = self.root / "first", self.root / "second"
        compute.main(["--degree", "2", "--output", str(first)])
        checkpoint = first / "degree2"
        original = (checkpoint / "composition_factors.sobj").read_bytes()
        compute.main([
            "--degree", "4", "--output", str(second),
            "--resume", str(checkpoint), "--reference",
        ])
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
            compute.main([
                "--degree", "2", "--resume", str(checkpoint),
                "--output", str(output),
            ])
        self.assertFalse(output.exists())

    def test_failed_comparison_is_recorded_and_blocks_resume(self):
        output = self.root / "failed"
        with patch("compute.compare_direct_plethysm", side_effect=ValueError("mismatch")):
            with self.assertRaisesRegex(ValueError, "mismatch"):
                compute.main([
                    "--degree", "1", "--output", str(output), "--reference",
                ])
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


if __name__ == "__main__":
    unittest.main()
