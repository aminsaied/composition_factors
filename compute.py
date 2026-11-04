"""Compute, inspect, query, export, compare, migrate, and plot exact coefficients."""

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from coefficient_data import CoefficientData, FORMAT_VERSION, encode_records, read_checkpoint

LEGACY_SOURCE_SHA256 = "3f6234d0539e7bca6fa4548416c6b728bb2807666b27c1b1d924e8ec8dc8241b"


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def peak_rss_mib():
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def encode_coefficients(cf):
    from sage.all import ZZ

    records = []
    for lam, row in cf.coeffs.items():
        for mu, value in row.items():
            if value not in ZZ or value < 0:
                raise ValueError(f"Invalid multiplicity: {lam}, {mu}: {value}")
            records.append(
                {
                    "lambda": [int(part) for part in lam],
                    "mu": [int(part) for part in mu],
                    "multiplicity": int(value),
                }
            )
    records.sort(key=lambda item: (item["lambda"], item["mu"]))
    return records, encode_records(records)


def write_report(path, report):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def load_checkpoint(directory, source_sha256):
    """Load only a trusted checkpoint; Sage's .sobj format uses pickle."""
    from composition_factors import ALGORITHM_VERSION, CompositionFactors

    directory = Path(directory)
    data = read_checkpoint(directory)
    report = data.report
    algorithm = report.get("algorithm_version")
    if algorithm is not None and algorithm != ALGORITHM_VERSION:
        raise ValueError(f"Unsupported checkpoint algorithm version: {algorithm!r}.")
    if report["source_sha256"] != source_sha256:
        raise ValueError(
            "Checkpoint and current mathematical source differ; use explicit migration."
        )
    if report.get("save_load_equal") is not True:
        raise ValueError("Checkpoint has no successful save/load verification.")
    cf = CompositionFactors.load(str(directory / "composition_factors"))
    if cf.top_degree != data.degree:
        raise ValueError("Checkpoint degree is invalid or disagrees with its report.")
    _, encoded = encode_coefficients(cf)
    if hashlib.sha256(encoded).hexdigest() != report["coefficient_sha256"]:
        raise ValueError("Checkpoint pickle coefficients disagree with the validated JSON hash.")
    if cf._matrix is not None:
        import numpy as np

        if not np.array_equal(cf._matrix, cf._compute_matrix()):
            raise ValueError("Checkpoint visualization matrix disagrees with exact coefficients.")
    if report.get("reference_status") == "pending":
        log("Warning: the checkpoint's optional independent comparison is unfinished.")
    return cf


def extend_one_degree(previous, degree, *, progress=None):
    from sage.all import Partitions
    from composition_factors import CompositionFactors, CompositionFactorsAtDegree

    if previous is not None and degree != previous.top_degree + 1:
        raise ValueError("Each checkpoint must advance exactly one degree.")
    if previous is None and degree != 1:
        raise ValueError("A computation without a checkpoint must start at degree 1.")
    old_rows = {} if previous is None else previous.coeffs
    new_rows = CompositionFactorsAtDegree(degree, progress=progress).table.table
    if set(new_rows) != set(Partitions(degree)):
        raise ValueError("New coefficient rows do not cover exactly the target degree.")
    if old_rows.keys() & new_rows.keys():
        raise ValueError("New coefficient rows overlap the saved checkpoint.")
    cf = CompositionFactors(degree, _coeffs={**old_rows, **new_rows})
    if any(cf.coeffs[lam] != row for lam, row in old_rows.items()):
        raise ValueError("Extending the checkpoint changed an earlier coefficient row.")
    return cf


def save_checkpoint(cf, directory, elapsed, source_sha256, reference_pending=False):
    from sage.env import SAGE_VERSION
    from composition_factors import ALGORITHM_VERSION, CompositionFactors

    directory = Path(directory)
    records, encoded = encode_coefficients(cf)
    rows = {
        tuple(map(int, lam)): {tuple(map(int, mu)): int(value) for mu, value in row.items()}
        for lam, row in cf.coeffs.items()
    }
    dimension_columns = CoefficientData(int(cf.top_degree), rows, {}).validate_dimensions()
    directory.mkdir()
    (directory / "coefficients.json").write_bytes(encoded + b"\n")
    cf.save(str(directory / "composition_factors"))
    restored = CompositionFactors.load(str(directory / "composition_factors"))
    if restored.coeffs != cf.coeffs:
        raise ValueError("Save/load changed the computed results.")
    if cf._matrix is not None:
        import numpy as np

        if restored._matrix is None or not np.array_equal(restored._matrix, cf._matrix):
            raise ValueError("Save/load changed the visualization matrix.")
    report = {
        "format_version": FORMAT_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "sage": SAGE_VERSION,
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "degree": int(cf.top_degree),
        "computation_seconds": elapsed,
        "computation_scope": f"degree {cf.top_degree} only; earlier degrees reused",
        "reused_through_degree": int(cf.top_degree) - 1,
        "partition_count": len(cf.coeffs),
        "matrix_entry_count": len(cf.coeffs) ** 2,
        "stored_coefficient_count": len(records),
        "nonzero_coefficient_count": sum(row["multiplicity"] != 0 for row in records),
        "coefficient_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_sha256": source_sha256,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "save_load_equal": True,
        "reference_status": "pending" if reference_pending else "not_requested",
        "dimension_columns_checked": dimension_columns,
        "process_peak_rss_mib": peak_rss_mib(),
    }
    write_report(directory / "report.json", report)
    return report


def compare_direct_plethysm(cf, max_mu_size=None):
    """Compare all lambda rows with the requested mu columns, including zeros."""
    from sage.all import Partition, Partitions, QQ, SymmetricFunctions, divisors, moebius

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
            (rho, coefficient) for rho, coefficient in polynomial if sum(rho) <= degree
        )

    power_images = {}
    checked = 0
    for size in range(1, max_mu + 1):
        for mu in Partitions(size):
            polynomial = p.zero()
            for rho, coefficient in p(s(mu)):
                if rho not in power_images:
                    term = p.one()
                    for k in rho:
                        term = truncate(term * substitutions[k])
                    power_images[rho] = term
                polynomial += coefficient * power_images[rho]
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


def mathematical_source_sha256():
    import composition_factors as module

    return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


def from_exact_data(data):
    from sage.all import Partition, Partitions
    from composition_factors import CompositionFactors

    parts = [part for degree in range(1, data.degree + 1) for part in Partitions(degree)]
    rows = {
        lam: {
            Partition(mu): data.coeffs[tuple(lam)][tuple(mu)] for mu in parts if sum(mu) <= sum(lam)
        }
        for lam in parts
    }
    return CompositionFactors(data.degree, _coeffs=rows)


def progress_logger(interval):
    last_stage, last_time = None, 0.0

    def report(event):
        nonlocal last_stage, last_time
        stage = event["degree"], event["stage"]
        now = time.monotonic()
        complete = "completed" in event and event["completed"] == event["total"]
        if stage != last_stage or complete or now - last_time >= interval:
            message = f"Degree {event['degree']}: {event['stage']}"
            if "completed" in event:
                message += f"; completed {event['completed']}/{event['total']} mu partitions"
            log(message + ".")
            last_stage, last_time = stage, now

    return report


def make_parser():
    parser = argparse.ArgumentParser(
        prog="composition-factors",
        description=__doc__,
        epilog="Legacy usage without the 'compute' subcommand is still accepted.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    compute = commands.add_parser("compute", help="Compute with per-degree checkpoints.")
    compute.add_argument("--degree", type=int, required=True, help="Final positive degree.")
    compute.add_argument("--output", type=Path, required=True, help="New run directory.")
    compute.add_argument(
        "--resume", type=Path, help="Trusted degree checkpoint directory containing report.json."
    )
    compute.add_argument("--reference", action="store_true", help="Compare direct plethysms.")
    compute.add_argument(
        "--reference-max-mu-size", type=int, help="Limit the independent comparison's mu columns."
    )
    compute.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help="Seconds between completed-input progress messages (default: 30).",
    )
    for name, help_text in (
        ("inspect", "Summarize validated exact JSON without Sage."),
        ("query", "Look up one exact coefficient without Sage."),
        ("export", "Export exact coefficients to CSV without Sage."),
        ("compare", "Independently compare an existing checkpoint without modifying it."),
        ("migrate", "Validate compatible exact data and create a current-source checkpoint."),
        ("plot", "Plot an existing checkpoint without recomputing coefficients."),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("checkpoint", type=Path)
        if name == "query":
            command.add_argument("--lambda", dest="lambda_", type=parse_partition, required=True)
            command.add_argument("--mu", type=parse_partition, required=True)
        if name in ("export", "migrate", "plot"):
            command.add_argument("--output", type=Path, required=True, help="New output path.")
        if name == "export":
            command.add_argument("--nonzero", action="store_true", help="Omit zero coefficients.")
        if name in ("compare", "migrate"):
            command.add_argument(
                "--reference-max-mu-size",
                type=int,
                default=4 if name == "migrate" else None,
                help="Limit comparison columns (migration default: 4; compare default: full).",
            )
        if name == "plot":
            command.add_argument("--kind", choices=("matrix", "pd", "fi"), default="matrix")
            command.add_argument(
                "--ticks", choices=("partitions", "degree", "none"), default="degree"
            )
            command.add_argument(
                "--palette", default="Paired", help="Palette name or 'historical'."
            )
            command.add_argument("--normalization", choices=("linear", "log"), default="linear")
            command.add_argument("--width", type=float, default=10.0)
            command.add_argument("--resolution", type=float, default=float("inf"))
            command.add_argument("--seq-length", type=int, default=2)
            command.add_argument(
                "--unique", action="store_true", help="Omit stability-chain suffixes."
            )
    return parser


def parse_partition(text):
    from coefficient_data import partition_key

    try:
        return partition_key([int(part.strip()) for part in text.split(",")])
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def run_computation(args, parser):
    if args.output.exists():
        raise FileExistsError(f"Output directory already exists: {args.output}")
    if args.reference_max_mu_size is not None:
        if not args.reference:
            parser.error("--reference-max-mu-size requires --reference")
        if not 1 <= args.reference_max_mu_size <= args.degree:
            parser.error("--reference-max-mu-size must be between 1 and --degree")
    source_sha256 = mathematical_source_sha256()
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
        cf = extend_one_degree(cf, degree, progress=progress_logger(args.progress_interval))
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
            log(
                f"Independent comparison passed for {report['direct_plethysm_pairs_checked']} pairs."
            )
    log(f"Complete through degree {args.degree}; results in {args.output}.")


def migrate_checkpoint(data, output, max_mu_size):
    from composition_factors import ALGORITHM_VERSION

    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    algorithm = data.report.get("algorithm_version")
    if algorithm is None:
        if data.report["source_sha256"] != LEGACY_SOURCE_SHA256:
            raise ValueError("Migration requires a recognized legacy source or algorithm version.")
    elif type(algorithm) is not int or algorithm not in (1, ALGORITHM_VERSION):
        raise ValueError(f"Unsupported checkpoint algorithm version: {algorithm!r}.")
    if algorithm in (None, 1) and data.degree >= 12:
        raise ValueError(
            "The historical algorithm can overcount from degree 12 onward. "
            "Recompute from a checkpoint through degree 11 or use corrected data."
        )
    cf = from_exact_data(data)
    comparison = compare_direct_plethysm(cf, max_mu_size)
    _, encoded = encode_coefficients(cf)
    if hashlib.sha256(encoded).hexdigest() != data.report["coefficient_sha256"]:
        raise ValueError("Migration changed the exact coefficient hash.")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = save_checkpoint(cf, output, 0.0, mathematical_source_sha256(), reference_pending=True)
    report.update(comparison)
    report.update(
        {
            "computation_scope": "imported existing exact coefficients; no new degree computed",
            "reused_through_degree": data.degree,
            "migration": {
                "source_sha256": data.report["source_sha256"],
                "coefficient_sha256": data.report["coefficient_sha256"],
                "format_version": data.report.get("format_version", 0),
                "algorithm_version": algorithm,
                "reference_status": data.report.get("reference_status", "unknown"),
            },
        }
    )
    write_report(output / "report.json", report)
    log(
        f"Migrated degree {data.degree}; compared {comparison['direct_plethysm_pairs_checked']} pairs."
    )


def plot_checkpoint(data, args):
    import matplotlib.pyplot as plt
    from composition_factors import Visualisations

    if args.output.exists():
        raise FileExistsError(f"Output file already exists: {args.output}")
    vis = Visualisations(from_exact_data(data))
    if args.kind == "matrix":
        figure = vis.display(
            args.resolution,
            args.width,
            {"partitions": True, "degree": "degree", "none": False}[args.ticks],
            cmap=args.palette,
            normalization=args.normalization,
        )
    else:
        plot = vis.PD_stability if args.kind == "pd" else vis.FI_stability
        figure = plot(args.resolution, args.seq_length, args.width, unique=args.unique)
    try:
        extension = args.output.suffix.lstrip(".").lower()
        if extension not in figure.canvas.get_supported_filetypes():
            raise ValueError("The output filename must have a supported image extension.")
        with args.output.open("xb") as stream:
            figure.savefig(stream, format=extension, dpi=150)
    finally:
        plt.close(figure)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv.insert(0, "compute")
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.command == "compute":
        if args.degree < 1:
            parser.error("--degree must be positive")
        if not math.isfinite(args.progress_interval) or args.progress_interval < 0:
            parser.error("--progress-interval must be finite and nonnegative")
        run_computation(args, parser)
        return
    if args.command in ("compare", "migrate"):
        if args.reference_max_mu_size is not None and args.reference_max_mu_size < 1:
            parser.error("--reference-max-mu-size must be positive")
    if args.command == "plot":
        if args.seq_length < 1 or not args.resolution > 0:
            parser.error("--seq-length and --resolution must be positive")
        if not 0 < args.width < float("inf"):
            parser.error("--width must be positive and finite")
        if args.kind == "matrix" and args.unique:
            parser.error("--unique applies only to stability plots")
        if args.kind != "matrix" and (
            args.palette != "Paired" or args.normalization != "linear" or args.ticks != "degree"
        ):
            parser.error("--palette, --normalization, and --ticks apply only to matrix plots")
        os.environ.setdefault("MPLBACKEND", "Agg")
    data = read_checkpoint(args.checkpoint)
    if args.command == "inspect":
        print(json.dumps(data.summary(), indent=2))
    elif args.command == "query":
        print(data.coeff(args.lambda_, args.mu))
    elif args.command == "export":
        data.export_csv(args.output, nonzero=args.nonzero)
    elif args.command == "compare":
        report = compare_direct_plethysm(from_exact_data(data), args.reference_max_mu_size)
        report["coefficient_sha256"] = data.report["coefficient_sha256"]
        print(json.dumps(report, indent=2))
    elif args.command == "migrate":
        migrate_checkpoint(data, args.output, args.reference_max_mu_size)
    elif args.command == "plot":
        plot_checkpoint(data, args)


def entrypoint():
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(entrypoint())
