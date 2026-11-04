"""Versioned exact coefficient data, readable with the Python standard library."""

import csv
import hashlib
import json
from dataclasses import dataclass
from math import factorial, prod
from pathlib import Path
from typing import Iterator

FORMAT_VERSION = 1
PartitionKey = tuple[int, ...]


def positive_integer(value, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def partition_key(value) -> PartitionKey:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("A partition must be a nonempty list or tuple.")
    parts = tuple(positive_integer(part, "Partition part") for part in value)
    if any(left < right for left, right in zip(parts, parts[1:])):
        raise ValueError("A partition must be weakly decreasing.")
    return parts


def partitions(degree: int, maximum: int | None = None) -> Iterator[PartitionKey]:
    """Enumerate partitions in the degree block's reverse lexicographic order."""
    if degree == 0:
        yield ()
        return
    for first in range(min(degree, maximum if maximum is not None else degree), 0, -1):
        for tail in partitions(degree - first, first):
            yield (first, *tail)


def encode_records(records: list[dict]) -> bytes:
    ordered = sorted(records, key=lambda row: (row["lambda"], row["mu"]))
    return json.dumps(ordered, separators=(",", ":")).encode("utf-8")


def _specht_dimension(partition: PartitionKey) -> int:
    columns = [sum(row >= column for row in partition) for column in range(1, partition[0] + 1)]
    hooks = (row - j + columns[j] - i - 1 for i, row in enumerate(partition) for j in range(row))
    return factorial(sum(partition)) // prod(hooks)


@dataclass
class CoefficientData:
    degree: int
    coeffs: dict[PartitionKey, dict[PartitionKey, int]]
    report: dict

    def coeff(self, lambda_, mu) -> int:
        lam, mu = partition_key(lambda_), partition_key(mu)
        if max(sum(lam), sum(mu)) > self.degree:
            raise ValueError(f"The requested partition exceeds computed degree {self.degree}.")
        if sum(mu) > sum(lam):
            return 0
        try:
            return self.coeffs[lam][mu]
        except KeyError as error:
            raise ValueError(f"Missing coefficient data for partitions {lam}, {mu}.") from error

    def records(self) -> Iterator[dict]:
        for lam in sorted(self.coeffs):
            for mu, value in sorted(self.coeffs[lam].items()):
                yield {"lambda": list(lam), "mu": list(mu), "multiplicity": value}

    def validate_dimensions(self) -> int:
        """Check each column against Specht dimensions and unsigned Stirling numbers."""
        blocks = {}
        for part in self.coeffs:
            blocks.setdefault(sum(part), []).append(part)
        dimensions = {part: _specht_dimension(part) for part in self.coeffs}
        stirling = [1] + [0] * self.degree
        columns, checked = [], 0
        for degree in range(1, self.degree + 1):
            for size in range(degree, 0, -1):
                stirling[size] = stirling[size - 1] + (degree - 1) * stirling[size]
            stirling[0] = 0
            block = blocks[degree]
            columns.extend(block)
            totals = dict.fromkeys(columns, 0)
            for lam in block:
                for mu, value in self.coeffs[lam].items():
                    totals[mu] += dimensions[lam] * value
            for mu in columns:
                expected = dimensions[mu] * stirling[sum(mu)]
                if totals[mu] != expected:
                    raise ValueError(
                        f"Coefficient dimension mismatch at degree {degree}, mu={mu}: "
                        f"got {totals[mu]}, expected {expected}. "
                        "Recompute with the corrected algorithm; do not resume this checkpoint."
                    )
                checked += 1
        return checked

    def summary(self) -> dict:
        return {
            "degree": self.degree,
            "format_version": self.report.get("format_version", 0),
            "algorithm_version": self.report.get("algorithm_version"),
            "partition_count": len(self.coeffs),
            "matrix_entry_count": len(self.coeffs) ** 2,
            "stored_coefficient_count": sum(len(row) for row in self.coeffs.values()),
            "nonzero_coefficient_count": sum(
                value != 0 for row in self.coeffs.values() for value in row.values()
            ),
            "source_sha256": self.report["source_sha256"],
            "coefficient_sha256": self.report["coefficient_sha256"],
            "reference_status": self.report.get("reference_status", "unknown"),
            "direct_plethysm_pairs_checked": self.report.get("direct_plethysm_pairs_checked"),
            "direct_plethysm_max_mu_size": self.report.get("direct_plethysm_max_mu_size"),
            "dimension_columns_checked": self.report.get("dimension_columns_checked"),
        }

    def export_csv(self, path: Path, *, nonzero: bool = False) -> None:
        with Path(path).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["lambda", "mu", "multiplicity"])
            writer.writeheader()
            for row in self.records():
                if not nonzero or row["multiplicity"]:
                    writer.writerow(
                        {
                            "lambda": ",".join(map(str, row["lambda"])),
                            "mu": ",".join(map(str, row["mu"])),
                            "multiplicity": row["multiplicity"],
                        }
                    )


def read_checkpoint(directory: Path) -> CoefficientData:
    """Validate exact JSON, without importing Sage or unpickling a .sobj file."""
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    records = json.loads((directory / "coefficients.json").read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not isinstance(records, list):
        raise ValueError("Expected a checkpoint report object and a coefficient record list.")
    version = report.get("format_version", 0)
    if type(version) is not int or version not in (0, FORMAT_VERSION):
        raise ValueError(f"Unsupported coefficient format version: {version!r}.")
    if version == FORMAT_VERSION or "algorithm_version" in report:
        positive_integer(report.get("algorithm_version"), "Algorithm version")
    degree = positive_integer(report.get("degree"), "Checkpoint degree")
    status = report.get("reference_status", "unknown")
    if status == "failed":
        raise ValueError("Checkpoint failed its independent comparison.")
    if status not in ("passed", "pending", "not_requested", "unknown"):
        raise ValueError(f"Unknown independent comparison status: {status!r}.")
    for name in ("source_sha256", "coefficient_sha256"):
        digest = report.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(f"Missing or invalid {name} hash.")

    rows = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"lambda", "mu", "multiplicity"}:
            raise ValueError("Invalid coefficient record fields.")
        lam, mu = partition_key(record["lambda"]), partition_key(record["mu"])
        value = record["multiplicity"]
        if type(value) is not int or value < 0:
            raise ValueError(f"Multiplicity must be a nonnegative integer: {lam}, {mu}.")
        if sum(lam) > degree or sum(mu) > sum(lam):
            raise ValueError(
                f"Coefficient record is outside the stored degree triangle: {lam}, {mu}."
            )
        row = rows.setdefault(lam, {})
        if mu in row:
            raise ValueError(f"Duplicate coefficient record: {lam}, {mu}.")
        if sum(lam) == sum(mu) and value != int(lam == mu):
            raise ValueError(f"Invalid same-degree coefficient: {lam}, {mu}.")
        row[mu] = value

    expected = set()
    for size in range(1, degree + 1):
        block = list(partitions(size))
        expected.update(block)
        for lam in block:
            if lam not in rows or set(rows[lam]) != expected:
                raise ValueError(f"Missing coefficient row or columns for partition {lam}.")
    if set(rows) != expected:
        raise ValueError("Checkpoint partition rows do not match the reported degree.")
    data = CoefficientData(degree, rows, report)
    encoded = encode_records(list(data.records()))
    if hashlib.sha256(encoded).hexdigest() != report["coefficient_sha256"]:
        raise ValueError("Checkpoint coefficients disagree with their recorded hash.")
    summary = data.summary()
    for name in (
        "partition_count",
        "matrix_entry_count",
        "stored_coefficient_count",
        "nonzero_coefficient_count",
    ):
        if name in report and (type(report[name]) is not int or report[name] != summary[name]):
            raise ValueError(f"Checkpoint {name} disagrees with its coefficient data.")
    if status == "passed":
        max_mu = positive_integer(
            report.get("direct_plethysm_max_mu_size"), "Independent comparison mu-size bound"
        )
        checked = positive_integer(
            report.get("direct_plethysm_pairs_checked"), "Independent comparison pair count"
        )
        if max_mu > degree:
            raise ValueError("Independent comparison bound exceeds the checkpoint degree.")
        columns = sum(sum(part) <= max_mu for part in rows)
        if checked != len(rows) * columns:
            raise ValueError("Independent comparison pair count disagrees with its stated scope.")
    dimension_columns = data.validate_dimensions()
    if "dimension_columns_checked" in report and (
        type(report["dimension_columns_checked"]) is not int
        or report["dimension_columns_checked"] != dimension_columns
    ):
        raise ValueError("Checkpoint dimension comparison count disagrees with its data.")
    return data
