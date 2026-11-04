"""Independent, type-selected Whitney characters via exact Sage plethysm.

Implements Saied's thesis, Theorem 6.2.19 and Remark 6.2.21, in characteristic
zero and positive rank. No composition-factor checkpoint or assembly is used.
"""

from collections import Counter
import hashlib
from itertools import product
from operator import index as integer_index
from pathlib import Path

from sage.all import Partition, QQ, SymmetricFunctions, ZZ, divisors, moebius
from sage.env import SAGE_VERSION

from coefficient_data import PartitionKey, partitions
from whitney_data import validate_whitney_request, write_whitney_csv, write_whitney_json

ALGORITHM_VERSION = 1


def _integer_input(value) -> int:
    if isinstance(value, bool):
        raise ValueError("Whitney parameters must be integers, not booleans.")
    try:
        return integer_index(value)
    except TypeError as error:
        raise ValueError("Whitney parameters must be exact integers.") from error


class WhitneyCalculator:
    """Compute ordinary or extended multiplicities, optionally for one block type.

    Caches twisted Lie characters and Schur plethysms only for this instance's
    lifetime. The reference algorithm favors clarity over large-degree scaling.
    """

    def __init__(self):
        symmetric = SymmetricFunctions(QQ)
        self._p, self._s = symmetric.power(), symmetric.schur()
        self._lie = {}
        self._factors = {}
        self._source = {
            "algorithm_version": ALGORITHM_VERSION,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "sage": SAGE_VERSION,
        }

    def _twisted_lie(self, degree: int):
        if degree not in self._lie:
            lie = self._p.sum_of_terms(
                (
                    Partition([int(d)] * (degree // int(d))),
                    QQ(moebius(d)) / degree,
                )
                for d in divisors(degree)
            )
            self._lie[degree] = self._s(lie.omega())
        return self._lie[degree]

    def _factor(self, size: int, nu: PartitionKey):
        key = (size, nu)
        if key not in self._factors:
            outer = Partition(nu)
            if size % 2 == 0:
                outer = outer.conjugate()
            self._factors[key] = self._s(outer).plethysm(self._twisted_lie(size))
        return self._factors[key]

    def _type_components(self, profile: PartitionKey, extended: bool):
        groups = sorted(Counter(profile).items())
        if not extended:
            character = self._s.one()
            for size, count in groups:
                character *= self._factor(size, (count,))
            return {(len(profile),): character}

        components = {}
        for choices in product(*(partitions(count) for _, count in groups)):
            block_character, ambient_character = self._s.one(), self._s.one()
            for (size, _), nu in zip(groups, choices):
                # The block character uses nu, not its even-index conjugate.
                block_character *= self._s(Partition(nu))
                ambient_character *= self._factor(size, nu)
            # Schur multiplication gives the restriction multiplicities by reciprocity.
            for mu, multiplicity in block_character:
                key = tuple(map(int, mu))
                components[key] = (
                    components.get(key, self._s.zero()) + multiplicity * ambient_character
                )
        return components

    def decomposition(
        self,
        degree: int,
        blocks: int,
        *,
        block_type=None,
        extended: bool = False,
        nonzero: bool = False,
    ) -> dict:
        """Compute a complete bidegree or type, accepting Python and Sage integers."""
        degree, blocks = _integer_input(degree), _integer_input(blocks)
        if block_type is not None:
            try:
                block_type = tuple(_integer_input(part) for part in block_type)
            except TypeError as error:
                raise ValueError("A block-size profile must be an iterable of integers.") from error
        profile = validate_whitney_request(degree, blocks, block_type)
        profiles = (
            (profile,)
            if profile is not None
            else (part for part in partitions(degree) if len(part) == blocks)
        )
        mus = list(partitions(blocks)) if extended else [(blocks,)]
        components = {mu: self._s.zero() for mu in mus}
        for part in profiles:
            for mu, character in self._type_components(part, extended).items():
                components[mu] += character

        coefficients = {}
        for mu, character in components.items():
            column = {}
            for lam, value in character:
                if sum(lam) != degree or value not in ZZ or value < 0:
                    raise ValueError(f"Invalid Whitney multiplicity at {lam}, {mu}: {value}.")
                column[tuple(map(int, lam))] = int(value)
            coefficients[mu] = column

        records = []
        for lam in partitions(degree):
            for mu in mus:
                value = coefficients[mu].get(lam, 0)
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
            "calculation": "signed_plethysm",
            "block_type": list(profile) if profile is not None else None,
            "nonzero_only": nonzero,
            "source": dict(self._source),
            "coefficients": records,
        }

    def export_json(
        self,
        path: Path,
        degree: int,
        blocks: int,
        *,
        block_type=None,
        extended: bool = False,
        nonzero: bool = False,
    ) -> None:
        result = self.decomposition(
            degree, blocks, block_type=block_type, extended=extended, nonzero=nonzero
        )
        write_whitney_json(result, path)

    def export_csv(
        self,
        path: Path,
        degree: int,
        blocks: int,
        *,
        block_type=None,
        extended: bool = False,
        nonzero: bool = False,
    ) -> None:
        result = self.decomposition(
            degree, blocks, block_type=block_type, extended=extended, nonzero=nonzero
        )
        write_whitney_csv(result, path)
