"""Total Whitney multiplicities in characteristic zero and positive lattice rank.

These are read-only views of validated composition-factor data, not a new
homology or signed-plethysm calculation. Here blocks counts blocks, not rank.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from coefficient_data import (
    CoefficientData,
    PartitionKey,
    partition_key,
    partitions,
    positive_integer,
)


def conjugate_partition(value) -> PartitionKey:
    """Transpose a nonempty partition, retaining the exact-data input contract."""
    part = partition_key(value)
    return tuple(sum(row >= column for row in part) for column in range(1, part[0] + 1))


def validate_whitney_request(degree: int, blocks: int, block_type=None) -> PartitionKey | None:
    """Validate positive rank and an optional partition of the underlying set size."""
    positive_integer(degree, "Degree")
    positive_integer(blocks, "Block count")
    if blocks >= degree:
        raise ValueError(
            "Whitney coefficients currently require positive rank: 1 <= blocks < degree. "
            "Rank-zero and empty-case conventions are not implemented."
        )
    if block_type is None:
        return None
    profile = partition_key(block_type)
    if sum(profile) != degree:
        raise ValueError("The block-size profile must sum to the requested degree.")
    if len(profile) != blocks:
        raise ValueError("The block-size profile must have the requested number of blocks.")
    return profile


def write_whitney_json(result: dict, path: Path) -> None:
    """Write a derived decomposition without overwriting an existing file."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


def write_whitney_csv(result: dict, path: Path) -> None:
    """Keep lookup CSVs compatible and identify fresh calculations explicitly."""
    context = {key: result[key] for key in ("kind", "degree", "blocks", "rank")}
    if result["calculation"] == "partition_conjugation":
        extra = {"source_coefficient_sha256": result["source"]["coefficient_sha256"]}
    elif result["calculation"] == "signed_plethysm":
        extra = {
            "block_type": ",".join(map(str, result["block_type"] or [])),
            "characteristic": result["characteristic"],
            "convention": result["convention"],
            "calculation": result["calculation"],
            "calculation_version": result["source"]["algorithm_version"],
            "source_sha256": result["source"]["source_sha256"],
            "sage": result["source"]["sage"],
        }
    else:
        raise ValueError(f"Unknown Whitney calculation method: {result['calculation']!r}.")
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=[*context, "lambda", "mu", "multiplicity", *extra]
        )
        writer.writeheader()
        for row in result["coefficients"]:
            writer.writerow(
                {
                    **context,
                    "lambda": ",".join(map(str, row["lambda"])),
                    "mu": ",".join(map(str, row.get("mu", []))),
                    "multiplicity": row["multiplicity"],
                    **extra,
                }
            )


@dataclass(frozen=True)
class WhitneyData:
    """A read-only interface over data returned by coefficient_data.read_checkpoint."""

    data: CoefficientData

    def _validate_bidegree(self, degree: int, blocks: int) -> None:
        validate_whitney_request(degree, blocks)
        if degree > self.data.degree:
            raise ValueError(f"The requested degree exceeds computed degree {self.data.degree}.")

    def extended_coeff(self, lambda_, mu) -> int:
        """Multiplicity of P_mu tensor P_lambda, with mu labeling the block action."""
        lam, mu = partition_key(lambda_), partition_key(mu)
        self._validate_bidegree(sum(lam), sum(mu))
        return self.data.coeff(conjugate_partition(lam), conjugate_partition(mu))

    def ordinary_coeff(self, lambda_, blocks: int) -> int:
        """Multiplicity of P_lambda after taking invariants under the block action."""
        positive_integer(blocks, "Block count")
        return self.extended_coeff(lambda_, (blocks,))

    def decomposition(
        self, degree: int, blocks: int, *, extended: bool = False, nonzero: bool = False
    ) -> dict:
        """Return exact multiplicities and provenance for one positive-rank bidegree."""
        self._validate_bidegree(degree, blocks)
        mus = list(partitions(blocks)) if extended else [(blocks,)]
        records = []
        for lam in partitions(degree):
            for mu in mus:
                value = self.extended_coeff(lam, mu)
                if nonzero and value == 0:
                    continue
                record = {"lambda": list(lam), "multiplicity": value}
                if extended:
                    record["mu"] = list(mu)
                records.append(record)
        return {
            "format_version": 1,
            "kind": "extended" if extended else "ordinary",
            "degree": degree,
            "blocks": blocks,
            "rank": degree - blocks,
            "characteristic": 0,
            "convention": "positive_rank",
            "calculation": "partition_conjugation",
            "nonzero_only": nonzero,
            "source": {
                "degree": self.data.degree,
                "format_version": self.data.report.get("format_version", 0),
                "algorithm_version": self.data.report.get("algorithm_version"),
                "coefficient_sha256": self.data.report["coefficient_sha256"],
            },
            "coefficients": records,
        }

    def export_json(
        self, path: Path, degree: int, blocks: int, *, extended: bool = False, nonzero: bool = False
    ) -> None:
        """Write a derived decomposition, not a resumable computation checkpoint."""
        result = self.decomposition(degree, blocks, extended=extended, nonzero=nonzero)
        write_whitney_json(result, path)

    def export_csv(
        self, path: Path, degree: int, blocks: int, *, extended: bool = False, nonzero: bool = False
    ) -> None:
        """Export exact rows with block/rank labels and the source coefficient hash."""
        result = self.decomposition(degree, blocks, extended=extended, nonzero=nonzero)
        write_whitney_csv(result, path)
