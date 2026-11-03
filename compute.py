"""Compute exact coefficients with a saved checkpoint after each degree."""

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from sage.all import Partition, Partitions, QQ, SymmetricFunctions, ZZ, divisors, moebius
from sage.env import SAGE_VERSION

import composition_factors as module
from composition_factors import CompositionFactors, CompositionFactorsAtDegree


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def peak_rss_mib():
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def encode_coefficients(cf):
    records = []
    for lam, row in cf.coeffs.items():
        for mu, value in row.items():
            if value not in ZZ or value < 0:
                raise ValueError(f"Invalid multiplicity: {lam}, {mu}: {value}")
            records.append({
                "lambda": [int(part) for part in lam],
                "mu": [int(part) for part in mu],
                "multiplicity": int(value),
            })
    records.sort(key=lambda item: (item["lambda"], item["mu"]))
    return records, json.dumps(records, separators=(",", ":")).encode()


def write_report(path, report):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def load_checkpoint(directory, source_sha256):
    """Load only a trusted checkpoint; Sage's .sobj format uses pickle."""
    report = json.loads((directory / "report.json").read_text())
    if report["source_sha256"] != source_sha256:
        raise ValueError("Checkpoint and current mathematical source differ.")
    if not report.get("save_load_equal"):
        raise ValueError("Checkpoint has no successful save/load verification.")
    if report.get("reference_status") == "failed":
        raise ValueError("Checkpoint failed its independent comparison.")
    cf = CompositionFactors.load(str(directory / "composition_factors"))
    if cf.top_degree != report["degree"] or cf.top_degree < 1:
        raise ValueError("Checkpoint degree is invalid or disagrees with its report.")
    expected_rows = {
        part
        for degree in range(1, cf.top_degree + 1)
        for part in Partitions(degree)
    }
    if set(cf.coeffs) != expected_rows:
        raise ValueError("Checkpoint does not contain every required partition row.")
    _, encoded = encode_coefficients(cf)
    if hashlib.sha256(encoded).hexdigest() != report["coefficient_sha256"]:
        raise ValueError("Checkpoint coefficients disagree with their recorded hash.")
    if report.get("reference_status") == "pending":
        log("Warning: the checkpoint's optional independent comparison is unfinished.")
    return cf


def extend_one_degree(previous, degree):
    if previous is not None and degree != previous.top_degree + 1:
        raise ValueError("Each checkpoint must advance exactly one degree.")
    if previous is None and degree != 1:
        raise ValueError("A computation without a checkpoint must start at degree 1.")
    old_rows = {} if previous is None else previous.coeffs
    new_rows = CompositionFactorsAtDegree(degree).table.table
    if set(new_rows) != set(Partitions(degree)):
        raise ValueError("New coefficient rows do not cover exactly the target degree.")
    if old_rows.keys() & new_rows.keys():
        raise ValueError("New coefficient rows overlap the saved checkpoint.")
    cf = CompositionFactors(degree, _coeffs={**old_rows, **new_rows})
    if any(cf.coeffs[lam] != row for lam, row in old_rows.items()):
        raise ValueError("Extending the checkpoint changed an earlier coefficient row.")
    return cf


def save_checkpoint(cf, directory, elapsed, source_sha256, reference_pending=False):
    directory.mkdir()
    records, encoded = encode_coefficients(cf)
    (directory / "coefficients.json").write_bytes(encoded + b"\n")
    cf.save(str(directory / "composition_factors"))
    restored = CompositionFactors.load(str(directory / "composition_factors"))
    if restored.coeffs != cf.coeffs or not np.array_equal(restored.matrix, cf.matrix):
        raise ValueError("Save/load changed the computed results.")
    report = {
        "sage": SAGE_VERSION,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "degree": int(cf.top_degree),
        "computation_seconds": elapsed,
        "computation_scope": f"degree {cf.top_degree} only; earlier degrees reused",
        "reused_through_degree": int(cf.top_degree) - 1,
        "partition_count": len(cf.coeffs),
        "matrix_entry_count": int(cf.matrix.size),
        "stored_coefficient_count": len(records),
        "nonzero_coefficient_count": sum(row["multiplicity"] != 0 for row in records),
        "coefficient_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_sha256": source_sha256,
        "save_load_equal": True,
        "reference_status": "pending" if reference_pending else "not_requested",
        "process_peak_rss_mib": peak_rss_mib(),
    }
    write_report(directory / "report.json", report)
    return report


def compare_direct_plethysm(cf, max_mu_size=None):
    """Compare all lambda rows with the requested mu columns, including zeros."""
    start = time.perf_counter()
    degree = cf.top_degree
    max_mu = degree if max_mu_size is None else min(max_mu_size, degree)
    if max_mu < 1:
        raise ValueError("The independent comparison needs a positive mu-size bound.")
    sym = SymmetricFunctions(QQ)
    p, s = sym.power(), sym.schur()
    lie = p.zero()
    for size in range(1, degree + 1):
        for divisor in divisors(size):
            lie += QQ(moebius(divisor)) / size * p([divisor] * (size // divisor))
    substitutions = {
        k: p.sum_of_terms(
            (Partition([k * part for part in rho]), coefficient)
            for rho, coefficient in lie
            if k * sum(rho) <= degree
        )
        for k in range(1, max_mu + 1)
    }

    def truncate(polynomial):
        return p.sum_of_terms(
            (rho, coefficient)
            for rho, coefficient in polynomial
            if sum(rho) <= degree
        )

    checked = 0
    for size in range(1, max_mu + 1):
        for mu in Partitions(size):
            polynomial = p.zero()
            for rho, coefficient in p(s(mu)):
                term = p.one()
                for k in rho:
                    term = truncate(term * substitutions[k])
                polynomial += coefficient * term
            expected = s(polynomial).monomial_coefficients()
            for lam, row in cf.coeffs.items():
                actual = row.get(mu, 0)
                if actual != expected.get(lam, 0):
                    raise ValueError(
                        f"Direct plethysm disagrees at {lam}, {mu}: "
                        f"{actual} != {expected.get(lam, 0)}"
                    )
                checked += 1
    return {
        "reference_status": "passed",
        "direct_plethysm_pairs_checked": checked,
        "direct_plethysm_seconds": time.perf_counter() - start,
        "direct_plethysm_max_mu_size": int(max_mu),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, required=True, help="Final positive degree.")
    parser.add_argument("--output", type=Path, required=True, help="New run directory.")
    parser.add_argument(
        "--resume", type=Path, help="Trusted degree checkpoint directory containing report.json."
    )
    parser.add_argument("--reference", action="store_true", help="Compare direct plethysms.")
    parser.add_argument(
        "--reference-max-mu-size", type=int, help="Limit the independent comparison's mu columns."
    )
    args = parser.parse_args(argv)
    if args.degree < 1:
        parser.error("--degree must be positive")
    if args.reference_max_mu_size is not None:
        if not args.reference:
            parser.error("--reference-max-mu-size requires --reference")
        if not 1 <= args.reference_max_mu_size <= args.degree:
            parser.error("--reference-max-mu-size must be between 1 and --degree")
    source_sha256 = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    cf = None
    first_degree = 1
    if args.resume:
        cf = load_checkpoint(args.resume, source_sha256)
        if args.degree <= cf.top_degree:
            parser.error("--degree must exceed the checkpoint's degree")
        first_degree = cf.top_degree + 1
        log(f"Loaded verified checkpoint through degree {cf.top_degree}.")
    args.output.mkdir(parents=True)
    for degree in range(first_degree, args.degree + 1):
        log(f"Computing degree {degree}.")
        start = time.perf_counter()
        cf = extend_one_degree(cf, degree)
        elapsed = time.perf_counter() - start
        directory = args.output / f"degree{degree}"
        report = save_checkpoint(cf, directory, elapsed, source_sha256, args.reference)
        log(f"Saved {directory}; new degree computed in {elapsed:.3f} seconds.")
        if args.reference:
            try:
                report.update(compare_direct_plethysm(cf, args.reference_max_mu_size))
            except ValueError as error:
                report["reference_status"] = "failed"
                report["reference_error"] = str(error)
                write_report(directory / "report.json", report)
                raise
            report["process_peak_rss_mib"] = peak_rss_mib()
            write_report(directory / "report.json", report)
            log(f"Independent comparison passed for {report['direct_plethysm_pairs_checked']} pairs.")
    log(f"Complete through degree {args.degree}; results in {args.output}.")


if __name__ == "__main__":
    main()
