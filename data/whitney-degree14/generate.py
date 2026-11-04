"""Generate and validate exact ordinary/extended Whitney data through a degree."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import cache
import hashlib
import json
from math import factorial, prod
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from coefficient_data import _specht_dimension, partitions, read_checkpoint
from whitney_data import WhitneyData


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def coefficients(result):
    records = result["coefficients"]
    values = {
        (tuple(row["lambda"]), tuple(row.get("mu", []))): row["multiplicity"] for row in records
    }
    require(len(values) == len(records), "Duplicate coefficient labels.")
    require(
        all(type(value) is int and value >= 0 for value in values.values()),
        "Multiplicities must be nonnegative integers.",
    )
    return values


dimension = cache(_specht_dimension)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=14)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calculator-commit", required=True)
    args = parser.parse_args()
    require(args.degree >= 2, "At least degree two is required for positive rank.")
    require(
        len(args.calculator_commit) == 40
        and all(char in "0123456789abcdef" for char in args.calculator_commit),
        "Supply the full Git commit containing the calculator.",
    )
    data = read_checkpoint(args.reference)
    require(data.degree >= args.degree, "The reference does not cover the requested degree.")

    from whitney_calculation import WhitneyCalculator

    calculator = WhitneyCalculator()
    lookup = WhitneyData(data)
    args.output.mkdir()
    payload = args.output / "decompositions.jsonl"
    temporary = args.output / "decompositions.jsonl.tmp"
    report_path = args.output / "report.json"
    started = time.perf_counter()
    report = {
        "format": "whitney-decompositions-jsonl",
        "format_version": 1,
        "status": "running",
        "maximum_degree": args.degree,
        "characteristic": 0,
        "convention": "positive_rank",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "calculator_commit": args.calculator_commit,
        "reference_coefficient_sha256": data.report["coefficient_sha256"],
        "source_file_sha256": {
            name: digest(ROOT / name)
            for name in (
                "whitney_calculation.py",
                "whitney_data.py",
                "coefficient_data.py",
                "data/whitney-degree14/generate.py",
            )
        },
        "bidegrees": 0,
        "block_types": 0,
        "decompositions": 0,
        "stored_coefficients": 0,
        "nonzero_coefficients": 0,
        "ordinary_total_coefficients_checked": 0,
        "extended_total_coefficients_checked": 0,
        "type_dimension_checks": 0,
        "type_invariant_coefficients_checked": 0,
        "degrees": [],
    }
    save_report(report_path, report)

    def write(stream, result):
        stream.write(json.dumps(result, separators=(",", ":")) + "\n")
        report["decompositions"] += 1
        report["stored_coefficients"] += len(result["coefficients"])
        report["nonzero_coefficients"] += sum(
            row["multiplicity"] != 0 for row in result["coefficients"]
        )

    complete = False
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            for degree in range(2, args.degree + 1):
                degree_start = time.perf_counter()
                for blocks in range(1, degree):
                    report["current"] = {"degree": degree, "blocks": blocks}
                    save_report(report_path, report)
                    totals = {}
                    accumulated = {}
                    for extended in (False, True):
                        result = calculator.decomposition(degree, blocks, extended=extended)
                        values = coefficients(result)
                        expected = coefficients(
                            lookup.decomposition(degree, blocks, extended=extended)
                        )
                        require(
                            values == expected,
                            f"Total mismatch at {degree=}, {blocks=}, {extended=}.",
                        )
                        totals[extended] = values
                        accumulated[extended] = dict.fromkeys(values, 0)
                        counter = "extended" if extended else "ordinary"
                        report[f"{counter}_total_coefficients_checked"] += len(values)
                        write(stream, result)
                    profiles = (part for part in partitions(degree) if len(part) == blocks)
                    for profile in profiles:
                        selected = {}
                        z = prod(
                            size**count * factorial(count)
                            for size, count in Counter(profile).items()
                        )
                        expected_dimension = factorial(degree) // z
                        for extended in (False, True):
                            result = calculator.decomposition(
                                degree, blocks, block_type=profile, extended=extended
                            )
                            values = coefficients(result)
                            require(
                                values.keys() == totals[extended].keys(), "Incomplete type labels."
                            )
                            actual_dimension = sum(
                                value * dimension(lam) * (dimension(mu) if extended else 1)
                                for (lam, mu), value in values.items()
                            )
                            expected = expected_dimension * (factorial(blocks) if extended else 1)
                            require(
                                actual_dimension == expected, f"Type dimension mismatch: {profile}."
                            )
                            for key, value in values.items():
                                accumulated[extended][key] += value
                            report["type_dimension_checks"] += 1
                            selected[extended] = values
                            write(stream, result)
                        for (lam, _), value in selected[False].items():
                            require(
                                value == selected[True][lam, (blocks,)], "Block invariant mismatch."
                            )
                            report["type_invariant_coefficients_checked"] += 1
                        report["block_types"] += 1
                    require(accumulated == totals, f"Type sum mismatch at {degree=}, {blocks=}.")
                    report["bidegrees"] += 1
                    print(
                        f"Validated degree {degree}, blocks {blocks}; {report['block_types']} types.",
                        flush=True,
                    )
                stream.flush()
                os.fsync(stream.fileno())
                report["degrees"].append(
                    {
                        "degree": degree,
                        "seconds": time.perf_counter() - degree_start,
                        "cumulative_decompositions": report["decompositions"],
                    }
                )
                save_report(report_path, report)
        temporary.replace(payload)
        require(
            report["bidegrees"] == args.degree * (args.degree - 1) // 2,
            "Incomplete bidegree coverage.",
        )
        require(
            report["block_types"]
            == sum(sum(1 for _ in partitions(degree)) - 1 for degree in range(2, args.degree + 1)),
            "Incomplete block-type coverage.",
        )
        require(
            report["decompositions"] == 2 * (report["bidegrees"] + report["block_types"]),
            "Incomplete ordinary/extended coverage.",
        )
        report["payload"] = {
            "file": payload.name,
            "sha256": digest(payload),
            "bytes": payload.stat().st_size,
        }
        complete = True
    finally:
        report["status"] = "passed" if complete else "failed"
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        report["seconds"] = time.perf_counter() - started
        save_report(report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
