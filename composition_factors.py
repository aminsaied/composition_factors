#!/usr/bin/env sage
r"""Composition Factors of `T(V)^{\otimes n}`

Computes the composition factors of the n-th tensor power of the free
associative algebra in terms of coefficients `c_{\lambda\mu}` indexing the
terms in the irreducible decomposition.

This file implements:

    - A fast algorithm computing the coefficients `c_{\lambda\mu}`.

    - A data-structure representing the free Lie algebra `\mathcal{L}(V)`.

    - Visualisations of the composition factors.

Examples:
    sage: cf = CompositionFactors(6)
    sage: lambda_ = Partition([4,2])
    sage: mu = Partition([1,1,1])
    sage: cf[lambda_][mu]
    2

    sage: lie = Lie(5)
    sage: lie
    Lie(s[2, 1, 1, 1] + s[2, 2, 1] + s[3, 1, 1] + s[3, 2] + s[4, 1])

Attributes:
    s -- Schur symmetric functions basis
    power -- power symmetric functions basis

AUTHOR:

- Amin Saied (2017-10-21)
"""

#*****************************************************************************
#       Copyright (C) 2017 Amin Saied <amin.saied@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#                  http://www.gnu.org/licenses/
#*****************************************************************************

from collections import Counter
from itertools import product
from copy import copy
from operator import index as integer_index
from sage.all import (
    Integer,
    Partition,
    Partitions,
    QQ,
    SymmetricFunctions,
    divisors,
    factorial,
    load,
    moebius,
    save,
)
from sage.libs.lrcalc.lrcalc import lrcoef as lr

s = SymmetricFunctions(QQ).schur()
power = SymmetricFunctions(QQ).power()

ALGORITHM_VERSION = 1


def _positive_degree(value, name="degree"):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer.")
    try:
        value = integer_index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a positive integer.") from error
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _positive_partition(value):
    try:
        raw = tuple(value)
        parts = tuple(integer_index(part) for part in raw)
    except TypeError as error:
        raise ValueError("A partition must contain positive integers.") from error
    if (
        not parts
        or any(isinstance(part, bool) for part in raw)
        or any(part < 1 for part in parts)
        or any(left < right for left, right in zip(parts, parts[1:]))
    ):
        raise ValueError("A partition must be nonempty and weakly decreasing.")
    return Partition(parts)


class CompositionFactors(object):
    """Compute composition factors up to a fixed degree.

    Runs :class:`CompositionFactorsAtDegree` for all degrees up to at most the
    target degree. Holds coefficients in form of a matrix as well as in the
    form of nested dict.

    The nested dict enables quick look-ups of a specific coefficient indexed
    by a pair ``lambda_`` and ``mu`` of partitions.

    The matrix form is used to visualise large-scale patterns in the data
    where partitions are indexed in reverse lexicographic order. Allows for
    visualisation with a :meth:`display`.

    Attributes:
        top_degree (int): The top degree of symmetric functions computed.
        matrix (array): Matrix of coefficients. The cols correspond to mu-
            partitions, and the rows to lambda_partitions. In both cases the
            partitions are ordered by rev-lex.
        coeffs (dict of partition: (dict of partition: int)): Nested dict
            holding coefficients indexed by pairs of partitions. The outer
            partition is the lambda-partition, and the inner partition is the
            mu-partition.

    Example:
        sage: cf = CompositionFactors(3)
        sage: cf.matrix
        array([[ 1.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.,  0.,  0.],
               [ 1.,  0.,  1.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  1.,  0.,  0.],
               [ 1.,  1.,  1.,  0.,  1.,  0.],
               [ 0.,  1.,  1.,  0.,  0.,  1.]])

        sage: cf = CompositionFactors(4)
        sage: cf.matrix
        array([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  1.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
               [ 1.,  1.,  1.,  1.,  1.,  0.,  0.,  1.,  0.,  0.,  0.],
               [ 0.,  2.,  1.,  0.,  1.,  1.,  0.,  0.,  1.,  0.,  0.],
               [ 1.,  1.,  2.,  1.,  2.,  1.,  0.,  0.,  0.,  1.,  0.],
               [ 0.,  1.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  1.]])

    For larger partitions it is possibly more convenient to look-up a specific
    coefficient.

        sage: cf = CompositionFactors(10)
        sage: lambda_ = Partition([4,3,3])
        sage: mu= Partition([3,2])
        sage: cf[lambda_][mu]
        84

    Even better, use the :meth:`display` method to get a birds-eye-view.

        sage: cf.display()

    This returns a matplotlib plot representing the data.
    """

    def __init__(self, top_degree, _coeffs=None, _matrix=None, *,
                 progress=None, cache_assemblies=True):
        """Initialise ``self``.

        Computes the coefficients for all degree up to ``top_degree``.

        Args:
            top_degree (int): The top degree of symmetric functions computed.
        """
        self.top_degree = _positive_degree(top_degree)
        self._partition_list = None

        if _coeffs is None:
            self.coeffs = self._compute(self.top_degree, progress, cache_assemblies)
        else:
            self.coeffs = _coeffs

        self._matrix = _matrix

    @property
    def matrix(self):
        """The lazily constructed floating-point visualization matrix."""
        if self._matrix is None:
            self._matrix = self._compute_matrix()
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self._matrix = value

    def __getitem__(self, key):
        """Return ``self.coeffs`` value stored at ``key``.

        Args:
            key (partition): A lambda-partition of degree at most
                ``self.max_degree``.
        """
        return self.coeffs[key]

    def save(self, filename):
        """Save composition factors to file.

        Args:
            filename (str): The filename under which to save.

        Example:
            sage: cf = CompositionFactors(15)
            sage: cf.save('composition_factors_15')
        """
        save_coeffs = [self.top_degree, self.coeffs, self._matrix]
        save(save_coeffs, filename)

    @classmethod
    def load(cls, filename):
        """Load saved composition factors.

        Args:
            filename (str): The filename to load.

        Example:
            sage: cf = CompositionFactors.load('composition_factors_15')
        """
        load_coeffs = load(filename)
        return cls(*load_coeffs)

    def coeff(self, lambda_, mu):
        """Look-up the coefficient of lambda_ and mu.

        Args:
            lambda_ (Partition): The lambda-partition.
            mu (Partition): The mu-partition.

        Example:
            sage: cf = CompositionFactors(5)
            sage: lambda_ = Partition([3,1])
            sage: mu = Partition([1,1])
            sage: cf.coeff(lambda_, mu)
            1
        """
        lambda_ = _positive_partition(lambda_)
        mu = _positive_partition(mu)
        if max(sum(lambda_), sum(mu)) > self.top_degree:
            raise ValueError(f"The requested partition exceeds computed degree {self.top_degree}.")
        if sum(mu) > sum(lambda_):
            return Integer(0)
        try:
            return self.coeffs[lambda_][mu]
        except KeyError as error:
            raise ValueError(f"Missing coefficient data for partitions {lambda_}, {mu}.") from error

    def _compute(self, top_degree, progress, cache_assemblies):
        """Compute coefficients up to ``top_degree``."""
        coeffs = {}

        for i in range(1, top_degree+1):
            cf_deg = CompositionFactorsAtDegree(
                i, progress=progress, cache_assemblies=cache_assemblies
            )
            coeffs.update(cf_deg.table.table)

        return coeffs

    def _compute_matrix(self):
        """Return array storing the coefficients.

        Returns numpy array storing the coefficients. Rows correspond to
        LambdaPartitions and columns to MuPartitions.
        """
        import numpy as np

        positions = {part: position for position, part in self._partitions()}
        n = len(positions)
        M = np.zeros((n, n))

        for lambda_, row in self.coeffs.items():
            for mu, value in row.items():
                M[positions[lambda_], positions[mu]] = value
        return M

    def _partitions(self):
        """Generator enumerating all partitions up to top degree."""
        if self._partition_list is None:
            self._partition_list = tuple(
                part for degree in range(1, self.top_degree + 1) for part in Partitions(degree)
            )
        yield from enumerate(self._partition_list)

    def display(self, resolution=float("inf"), width=20, ticks=True, *,
                cmap="Paired", normalization="linear"):
        """Displays a representation of :attr:`matrix`.

        Uses the :class:`Visualisations` to display an image of `self.matrix`.

        When considered as an array of integers, the data can be visualised as
        with ``plt.imshow``. For large degrees, the coefficients can get very
        large, skewing potentially interesting lower degree information.
        Coefficients are capped at ``resolution`` to bring out these lower
        dimensional features.
        """
        vis = Visualisations(self)
        return vis.display(resolution, width, ticks, cmap=cmap, normalization=normalization)


class CompositionFactorsAtDegree(object):
    """Computes the composition factors of a fixed degree.

    Brings everything together to efficiently compute the composition factors
    of a fixed degree ``target``.

    First computes an :class:`InstructionLookup` corresponding to target.

    Then initialises a :class:`CoefficientTable`. It then computes the
    coefficients as follows. For each mu-partition of size at most ``target``,
    compute each decomposition. For each decomposition, lookup in
    ``self.lookup`` the :class:`InstructionManual`, using it to assemble a
    symmetric function of fixed degree ``target``.

    Finally, these are added to ``self.table``.

    Attributes:
        target (int): The degree of symmetric functions computed.
        lie (:class:`Lie`): The free Lie algebra.
        lookup (:class:`InstructionLookup`): Holds the instruction manuals.
        table (:class:`CoefficientTable`): Holds the coefficients.
    """

    def __init__(self, target, *, progress=None, cache_assemblies=True):
        """Computes the composition factors of a fixed degree.

        Args:
            target (int): The degree of symmetric functions computed.

        Example:
            sage: cf_deg4 = CompositionFactorsAtDegree(4)
            sage: lambda_ = Partition([2,1,1])
            sage: mu = Partition([1,1])
            sage: cf_deg4[lambda_][mu]
            2
        """
        self.target = _positive_degree(target)
        self.progress = progress
        self.cache_assemblies = cache_assemblies
        self._notify("lie")
        self.lie = Lie(self.target)
        self._notify("instructions")
        self.lookup = InstructionLookup(self.target, lie=self.lie)
        self.table = CoefficientTable(self.target)

        self._compute()

    def __getitem__(self, key):
        """Return ``self.table`` value at ``key``.

        Args:
            key (partition): A lambda-partition.
        """
        return self.table[key]

    def shape_mu_pairs(self):
        """Yield a tuple of partition, :class:`MuPartition`.

        Yields:
            tuple of partition, :class:`MuPartition`: The partition cycles
                through all shapes, while the MuPartition cycles through all
                MuPartitions of the given shape.
        """
        for mu in self._mu_partitions():
            for shape in mu.shapes:
                yield shape, mu

    def _mu_partitions(self):
        for degree in range(1, self.target):
            for part in Partitions(degree):
                yield MuPartition(part)

    def _notify(self, stage, **counts):
        if self.progress is not None:
            self.progress({"degree": self.target, "stage": stage, **counts})

    def _compute(self):
        """Implements the algorithm computing composition factors."""
        assembled = {}
        total = int(sum(Partitions(degree).cardinality() for degree in range(1, self.target)))
        for completed, mu in enumerate(self._mu_partitions(), 1):
            for shape in mu.shapes:
                if shape not in self.lookup:
                    continue
                decompositions = mu.decompositions_for(shape)
                manual = self.lookup[shape]

                for decomposition in decompositions:
                    lr_coeff = mu.lr_coeff(decomposition)
                    if lr_coeff > 0:
                        if self.cache_assemblies:
                            key = (shape, decomposition.partitions_tuple)
                            if key not in assembled:
                                assembled[key] = manual.assemble(decomposition, self.lie)
                            schur = assembled[key]
                        else:
                            schur = manual.assemble(decomposition, self.lie)
                        over = decomposition.overcount
                        self.table.add(schur, mu.partition, lr_coeff*over)
            self._notify("coefficients", completed=completed, total=total)
        if total == 0:
            self._notify("coefficients", completed=0, total=0)


class MuPartition(object):
    """Implements decomposition and calculates iterated Littlewood-Richardson.

    Computes decompositions of a partition. A decomposition of a partition mu
    is a set of partitions whose sizes sum to the size of mu. Such a
    decomposition can be assembled into a Schur polynomial by pairing off
    terms with irreducible components of the free Lie algebra.

    Decompositions are organsied by their shape. The shape of a decomposition
    is a weakly decreasing list of the sizes of constituant sub-partitions.
    The shape is therefore stored as a partition.

    To each decomposition we associate an integer, the iterated Littlewood-
    Richardson coefficient. This counts how many ways there are to obtain the
    decomposition from mu.

    Attributes:
        partition (partition): The underlying partition.
        length (int): Length of the partition
        shapes (list): List of shapes of decompositions of mu.
        decompositions (list): List of decompositions of mu.
    """

    def __init__(self, partition):
        """Initialise self by computing the decompositions.

        Args:
            partition (partition): The underlying partition.
        """
        self.partition = Partition(partition)
        self.length = sum(partition)

        self._shapes = None
        self._decompositions = {}

    def __str__(self):
        """Return string representation of ``self``."""
        return "MuPartition({})".format(self.partition)

    def __repr__(self):
        """Return string representation of ``self``."""
        return self.__str__()

    def lr_coeff(self, decomposition):
        """Return the iterated Littlewood-Richardson coefficient.

        Args:
            decomposition (:class:`Decomposition`): A decomposition of
                ``self.partition``.

        Returns:
            int: The iterated Littlewood-Richardson coefficient.

        Example:
            sage: mu = MuPartition([3,1])
            sage: decomposition = Decomposition(([2,1],[1]))
            sage: mu.lr_coeff(decomposition)
            1
        """
        return self.iter_LR(self.partition, decomposition)

    @classmethod
    def iter_LR(cls, p, decomp):
        r"""Recursively computes the iterate Littlewood-Richardson coefficient.

        Args:
            p (partition): The outer partition.
            decomp (list): The decomposition (either list or tuple) of ``p``.

        Algorithm:
            Let:

            ..MATH::

                c = LR(p, [d_0, \ldots, d_k])

            denote the iterated Littlewood-Richardson coefficient of ``p``
            with a decomposition ``[d_0, \ldots, d_k]``. We make use of a
            formula computing ``c`` in terms of:

            ..MATH::

                LR(q, [d_1, \ldots, d_k])

            and `LR(p, d_0, q)`.

            where ``q`` ranges over all partitions of size ``|p|-|d_0|``. The
            latter is a standard Littlewood-Richardson, and the former is
            computed recursively.
        """
        k = len(decomp)

        if k == 1:
            return 1 if (decomp[0] == p) else 0

        elif k == 2:
            return lr(p, decomp[0], decomp[1])

        else:

            size = sum(p) - sum(decomp[0])
            c = 0
            for eta in Partitions(size, outer=p):

                x = lr(p, decomp[0], eta)

                if x > 0:
                    c += x * cls.iter_LR(eta, decomp[1:])

            return c

    @property
    def shapes(self):
        """A list of all possible shape partitions.

        Returns:
            list: A list of partitions.

        Example:
            sage: mu = MuPartition([2,1])
            sage: mu.shapes
            [[3], [2, 1], [1, 1, 1]]
        """
        if self._shapes is None:
            self._shapes = list(Partitions(self.length))
            return self._shapes
        else:
            return self._shapes

    @property
    def decompositions(self):
        """Return decompositions of ``self`` organised by their shape.

        Returns:
            dict of (partition, list): The list accessed by a shape partition
                is of all decompositions of ``self.partition`` of that shape.

        Example:
            sage: mu = MuPartition([1,1])
            sage: mu.decompositions
            {[1, 1]: [Decomp(([1], [1]))], [2]: [Decomp(([1, 1],))]}
        """
        for shape in self.shapes:
            self.decompositions_for(shape)
        return self._decompositions

    def decompositions_for(self, shape):
        """Compute only this shape, preserving the historical ordering."""
        shape = _positive_partition(shape)
        if sum(shape) != self.length:
            raise ValueError("The decomposition shape must have the same size as mu.")
        if shape not in self._decompositions:
            decompositions, seen = [], set()
            partitions = [Partitions(size, outer=self.partition) for size in shape]
            for combo in product(*partitions):
                ordered = tuple(sorted(combo, reverse=True, key=lambda part: (sum(part), len(part))))
                if ordered not in seen:
                    seen.add(ordered)
                    decompositions.append(Decomposition(ordered))
            self._decompositions[shape] = decompositions
        return self._decompositions[shape]


class Lie(object):
    """The free Lie algebra on a vector space.

    Handles the representation theory of the free Lie algebra. Holds both the
    list of irreducible representations (in the form of partitions) and the
    underlying symmetric function in the Schur basis.

    Attributes:
        max degree (int): The maximum degree considered.
        lie (:class:`SymmetricFunctions`): Top degree as a symmetric function.
        as_list (list): List of partitions describing irreducibles.
        sizes (list): List of sizes of partitions describing irreducibles.

    Examples:
        sage: lie = Lie(5)
        sage: lie.lie
        s[2, 1, 1, 1] + s[2, 2, 1] + s[3, 1, 1] + s[3, 2] + s[4, 1]

        sage: lie = Lie(4)
        sage: lie.as_list
        [[1], [1, 1], [2, 1], [2, 1, 1], [3, 1]]

        sage: lie.sizes
        [1,2,3,4,4]
    """

    def __init__(self, n):
        """Initialise ``self``.

        Args:
            n (int): The maximum degree of the free Lie algebra considered.
        """
        self.max_degree = _positive_degree(n)
        characters = [self._lie(degree) for degree in range(1, self.max_degree + 1)]
        self.lie = characters[-1]
        self.as_list = [
            part for character in characters for part, multiplicity in character
            for _ in range(int(multiplicity))
        ]
        self.sizes = [int(sum(part)) for part in self.as_list]
        counts = Counter(self.sizes)
        self._sizes_dict = {0: 0}
        for degree in range(1, self.max_degree + 1):
            self._sizes_dict[degree] = self._sizes_dict[degree - 1] + counts[degree]
        self._plethysms = {}

    def __str__(self):
        """Return a string representation of ``self``."""
        return "Lie({})".format(self.lie)

    def __repr__(self):
        """Return a string representation of ``self``."""
        return self.__str__()

    @classmethod
    def up_to(cls, n):
        """Collects irreducible representations in a list.

        Return list of partitions corresponding to irreducible Lie
        representations of size at most n.

        Args:
            n (int): The maximum degree of irreducibles.

        Returns:
            list: List of partitions of the free Lie algebra.

        Example:
            sage: Lie(7).up_to(4)
            [[1], [1, 1], [2, 1], [2, 1, 1], [3, 1]]
        """
        lies = []

        for i in range(n):

            for term in (cls._lie(i+1)):
                lies += [term[0]]*int(term[1])

        return lies

    @classmethod
    def length_up_to(cls, n):
        """Counts the number of irreducuble representations up to ``n``.

        Computes the number of irreducible Lie representations of size at most
        ``n``.

        Args:
            n (int): The cut-off for the count.

        Returns:
            int: Length of the list of irreducibles of degree at most ``n``.

        Example:
            sage: Lie(7).length_up_to(4)
            5
        """
        return len(cls.up_to(n))

    def sizes_up_to(self, max_):
        """Return list of sizes(/degrees) of irreducibles up to ``max_``.

        Args:
            max_ (int): The maximum degree of irreducibles considered.

        Returns:
            list: The sizes(/degrees) of each irreducible representations.

        Example:
            sage: lie = Lie(5)
            sage: lie.sizes_up_to(5)
            [1,2,3,4,4,5,5,5,5,5]
        """
        count = self.count_up_to(max_)
        if max_ == self.max_degree:
            return self.sizes
        else:
            return self.sizes[:count]

    def count_up_to(self, degree):
        """Count summands, including multiplicity, without rebuilding Lie data."""
        if isinstance(degree, bool):
            raise TypeError("The Lie degree bound must be an integer.")
        degree = integer_index(degree)
        if not 0 <= degree <= self.max_degree:
            raise ValueError(f"The Lie degree bound must be between 0 and {self.max_degree}.")
        return self._sizes_dict[degree]

    def _plethysm(self, partition, lie_id):
        """Cache repeated Schur plethysms for this Lie instance's lifetime."""
        key = (Partition(partition), self.as_list[lie_id])
        if key not in self._plethysms:
            self._plethysms[key] = s(key[0]).plethysm(s(key[1]))
        return self._plethysms[key]

    @staticmethod
    def _lie(n):
        """Symmetric polynomial representation of free Lie algebra.

        Computes Lie in the power basis for symmetric functions and
        converts to Schur polynomials.

        Args:
            n (int): The degree of lie to compute.

        Returns:
            Symmetric function in Schur basis of the corresponding degree.
        """
        power = SymmetricFunctions(QQ).power()

        terms = []
        for d in divisors(n):
            x = (Partition([d for j in range(n // d)]), QQ(moebius(d))/n)
            terms.append(x)

        lie_power = power.sum_of_terms(terms)

        return s(lie_power)

    @classmethod
    def _compute_sizes(cls, n):
        """Return a list of sizes of irreducibles."""
        sizes = []

        for i in range(1, n+1):
            coeffs = cls._lie(i).coefficients()
            count = int(sum(coeffs))
            sizes += [i] * count

        return sizes

    @classmethod
    def _compute_sizes_dict(cls, n):
        """Return a dictionary mapping indices to lengths."""
        counts = Counter(cls._compute_sizes(n))
        index_dict = {0: 0}
        for i in range(1, n+1):
            index_dict[i] = index_dict[i - 1] + counts[i]
        return index_dict


class DegreeBounds(object):
    """Upper bounds for the degree of the free Lie algebra.

    Given a shape partition and a target size, find the maximum possible size
    of an irreducible Lie representation that can be used to construct a
    Lambda partition of the given target size.

    Computes a dictionary mapping partitions (corresponding to the shape of a
    decomposition) to an integer (corresponding to a maximum degree in the
    free Lie algebra).

    Attributes:
        lie (:class:`Lie`): A free Lie algebra object.
        target (int): The size of resultant partitions.
        bounds (dict of partition: int):

    Example:
        sage: bounds = DegreeBounds(lie, 4)
        sage: bounds.bounds
        {[1]: 4, [1, 1]: 3, [2]: 2, [2, 1]: 2, [3]: 1, [4]: 1}
    """

    def __init__(self, target, lie=None):
        """Initialise self.

        Args:
            target (int): The desired size of resultant partitions.
            lie (:class:`Lie`) (optional): The free Lie algebra.
        """
        self.target = target
        self._lie = Lie(target) if not lie else lie
        self.bounds = self._compute_bounds()

    def __getitem__(self, key):
        """Look in ``bound`` dictionary."""
        return self.bounds[key]

    def __iter__(self):
        """Iterate through ``bound`` keys."""
        return iter(self.bounds)

    def __str__(self):
        return "DegreeBounds(target={})".format(self.target)

    def __repr__(self):
        return self.__str__()

    def iteritem(self):
        """A generator for (shape, bound) tuples.

        Yields:
            (partition, int): The shape partition of a decomposition and its
                corresponding upper bound.
        """
        for shape, bound in self.bounds.items():
            yield shape, bound

    def _compute_bounds(self):
        """Computes the bounds dictionary.

        Computes shapes_sizes dictionary, then throws out any shape that
        cannot contribute to a LambdaPartition of the target size on the
        grounds of their not being enough irreducible Lie representations of
        the required sizes.

        Returns:
            dict of (partition: int): The shape partition of a decomposition
                and its corresponding upper bound.
        """
        bounds = self._initialise_bounds()

        for shape in list(bounds):

            max_lie = bounds[shape]
            n_lie = self._lie.count_up_to(max_lie) if max_lie > 0 else 0

            if n_lie < len(shape):
                del bounds[shape]

        return bounds

    def _initialise_bounds(self):
        """Initialise bounds dictionary by calling :meth:``_bound_algorithm``.

        Returns:
            dict of (partition: int): The shape partition of a decomposition
                and its corresponding upper bound.
        """
        lie_sizes = self._lie.sizes_up_to(self.target)
        bounds = {}

        for i in range(1, self.target+1):

            for shape in Partitions(i):

                if self.target < i:
                    bounds[shape] = 0

                else:
                    m = self._bound_algorithm(shape, lie_sizes,
                                              self.target)

                    bounds[shape] = m

        return bounds

    @staticmethod
    def _bound_algorithm(shape, lie_sizes, target):
        """Algorithm computing the upper bounds of a given shape.

        Algorithm:
            Make pass through shape pairing with Lie partitions at the same
            index. Lie partitions are weakly increasing and the shape is
            weakly decreasing, this maintains the minimial possible degree at
            each step. Halt at penultimate index. Now find the largest
            possible element of Lie that can be paired with the remaining
            shape to achieve target size.

        Returns:
            int: The upper bound of the degree in the free Lie algebra
                corresponding to the given shape of a decomposition.
        """
        shape_list = list(shape)
        depth = len(shape_list)

        i = 0
        ast = 0
        while (ast < target) and (i < depth-1):
            ast += shape_list[i] * lie_sizes[i]
            i += 1

        remaining = target - ast
        last_shape = shape_list[depth-1]

        return remaining // last_shape


class Instruction(object):
    r"""A pairing up of shape indices and indices of the free Lie algebra.

    Given a shape `(s_0, \ldots, s_k)`, an instruction is a mapping (in the
    form of a dictionary) of the indices `0, \ldots, k` to distinct indices of
    the list of irreducible representations of the free Lie algebra.

    Instructions are used to assemble Schur polynomials of a fixed degree.

    Attributes:
        shape (partition): The shape partition of a decomposition.
        steps (dict of int: int): A dictionary mapping indices of shape to
            indices of Lie.
    """

    def __init__(self, shape):
        """Initialise an empty steps dict.

        Args:
            shape (partition): The shape parition of a decomposition.
        """
        self.shape = Partition(shape)
        self.steps = {step_id: None for step_id in range(len(shape))}
        self._empty_slots = len(shape)

    def __str__(self):
        """Return a string representation of ``self``.

        Example:
            sage: print(Instruction([2,1,1]))
            {0: None, 1: None, 2: None}
        """
        return str(self.steps)

    def __repr__(self):
        """Return a string representation of ``self``.

        Example:
            sage: Instruction([2,1,1])
            {0: None, 1: None, 2: None}
        """
        return self.__str__()

    def add_step(self, step):
        """Add a step to the instruction.

        Args:
            step (tuple of int, int): A tuple of indices. The first index
                corresponds to the shape partition, the second to Lie.

        Example:
            sage: instruction = Instruction([2,1,1])
            sage: instruction.add_step((0,4))
            sage: print(instruction)
            {0: 4, 1: None, 2: None}
        """
        shape_id, lie_id = step
        self.steps[shape_id] = lie_id
        self._empty_slots -= 1

    def add_step_and_copy(self, step):
        """Return a copy with independent steps and the new step added.

        Args:
            step (int, int): A tuple of the form (``shape_id``, ``lie_id``).
                The ``shape_id`` is the index in the shape decomposition and the
                ``lie_id`` is the index in the free Lie algebra.

        Returns:
            :class:`Instruction` : A copy of ``self`` with the step added.

        Example:
            sage: instruction = Instruction([2,1,1])
            sage: step = (0,4)
            sage: new_instruction = instruction.add_step_and_copy(step)
            sage: new_instruction
            {0: 4, 1: None, 2: None}
        """
        new_instruction = copy(self)
        new_instruction.steps = self.steps.copy()
        new_instruction.add_step(step)
        return new_instruction

    def full(self):
        """Bool checking if there are any vacant spots in the steps dict.

        Returns:
            bool: True is there are no ``None`` values in ``self.steps``.
                False otherwise.
        """
        if self._empty_slots == 0:
            return True
        else:
            return False


class InstructionManual(object):
    """Builds and assembles instructions.

    Given a shape partition and a target degree, finds all matching
    instructions. See :class:`Instruction` for more details. Further,
    the :meth:`assemble` method computes the associated Schur polynomial.

    The workhorse of the class is the :meth:`_build_instructions` method that
    implements an efficient recursive algorithm computing instructions.

    Attributes:
        shape (partition): The partition associated to a decomposition.
        lie_sizes (list): List of sizes of irreducible Lie representations.
        target (int): The size of resultant partitions.
        instructions (list): List of :class:`Instruction`s.

    Example:
        sage: manual = InstructionManual(Partition([2,1]), lie.sizes, 5)
        sage: manual
        Manual(target=5, shape=[2, 1], n_instructions=2)

        sage: manual.instructions
        [{0: 0, 1: 2}, {0: 1, 1: 0}]
    """

    def __init__(self, shape, lie_sizes, target):
        """Builds list of instrictions of given shape.

        Builds a list of all instructions that turn decompositions of the
        given shape into lambda-partitions of the target size.

        Args:
            shape (partition): The partition associated to a decomposition.
            lie_sizes (list): List of sizes of irreducible Lie representations.
            target (int): The size of resultant partitions.
        """
        self.shape = shape
        self.lie_sizes = lie_sizes
        self.target = target
        self.instructions = []
        self.build()

    def __str__(self):
        """Return a string representation of ``self``.

        Example:
            sage: manual = InstructionManual(Partition([2,1]), lie.sizes, 5)
            sage: print(manual)
            Manual(target=5, shape=[2, 1], n_instructions=2)
        """
        target = self.target
        shape = self.shape
        n = self.n_instructions
        print_string = ("Manual(target={}, shape={}, "
                        "n_instructions={})".format(target, shape, n))
        return print_string

    def __repr__(self):
        """Return a string representation of ``self``.

        Example:
            sage: manual = InstructionManual(Partition([2,1]), lie.sizes, 5)
            sage: manual
            Manual(target=5, shape=[2, 1], n_instructions=2)
        """
        return self.__str__()

    def __iter__(self):
        """Iterate through ``self.instructions``."""
        return iter(self.instructions)

    @property
    def n_instructions(self):
        """The number of instructions in ``self.instructions``.

        Returns:
            int: The number of instructions in ``self.instruction``.

        """
        return len(self.instructions)

    def add(self, instruction):
        """Add an instruction to ``self.instructions``.

        Args:
            instruction (:class:`Instruction`): The instruction to be added.
        """
        assert type(instruction) == Instruction
        self.instructions.append(instruction)

    def assemble(self, decomposition, lie):
        """Assembles a Schur polynomial using the list of instructions.

        Uses ``self.instructions`` to convert a decomposition into a Schur
        polynomial of degree ``self.target``. The terms in this polynomial
        correspond to the irreducible representations we call lambda-
        partitions.

        The actual assembly involves taking plethysms and tenson products of
        symmetric functions (or equivalently, of representations of symmetric
        groups). An instruction is a pairing of a decomposition with the free
        Lie algebra. Concretely, the assembly first takes plethyms of the
        paired terms, taking the tensor product of the resulting terms.

        Args:
            decomposition (list or :class:`Decomposition`): either a list of
                decompositions, or a single instance of
                :class:`Decomposition`.
            lie (:class:`Lie`): The free Lie algebra.

        Returns:
            The Schur polynomial assocated with the decomposition.

        Example:
            sage: shape = Partition([1,1])
            sage: lie, target = Lie(3), 3
            sage: manual = InstructionManual(shape, lie.sizes, target)
            sage: mu = MuPartition([2])
            sage: manual.assemble(mu.decompositions[shape], lie)
            2*s[1, 1, 1] + 2*s[2, 1]
        """

        if type(decomposition) == list:

            schur_poly = 0

            for x in decomposition:
                schur_poly += self.assemble(x, lie)

            return schur_poly

        else:

            schur_poly = 0

            for instruction in self.instructions:

                d = instruction.steps
                f = s([])

                for shape_id, p in decomposition.iteritem():

                    lie_id = d[shape_id]
                    f *= lie._plethysm(p, lie_id)

                schur_poly += f

            return schur_poly

    def build(self):
        """Recursively construct instructions of the given shape and target.

        Wrapper for :meth:`_build_instructions` method which implements the
        recursive algorithm computing all possible ways to assemble Schur
        polynomials of the target degree.
        """
        instruction = Instruction(self.shape)
        lie_pairs = _LieSizeIndex(self.lie_sizes)

        self._build_instructions(lie_pairs, self.target, instruction)

        return self

    def _build_instructions(self, lie_pairs, target, instruction,
                            pointer=0):
        """Recursive algorithm computing assembly instructions.

        Algorithm:
            The base case is a shape of length 1. Find indices of lie_sizes
            whose product with shape is target.

            Given a shape of length at least 2, consider all possible
            products of ``shape[0]`` and ``lie_sizes`` whose product is at
            most the target. Add each of these products to the instructions
            recursively computed from ``shape[1:]``.
        """
        if len(self.shape[pointer:]) == 1:

            row = self.shape[pointer]

            for pair in lie_pairs:

                value = target - (row * pair.size)

                if value == 0:
                    step = (pointer, pair.id)
                    new_instruction = instruction.add_step_and_copy(step)

                    if new_instruction.full():
                        self.add(new_instruction)
        else:

            row = self.shape[pointer]

            for pair in lie_pairs:

                value = target - (row * pair.size)

                if value <= 0:
                    continue

                step = (pointer, pair.id)
                new_instruction = instruction.add_step_and_copy(step)

                new_lie_pairs = lie_pairs.copy_and_remove(pair)

                self._build_instructions(new_lie_pairs, value,
                                         new_instruction, pointer+1)


class InstructionLookup(object):
    """A dictionary mapping shapes to :class:`InstructionManual`s.

    Given a target degree of Schur polynomials to be computed from an assemly
    process, and a shape partition, an :class:`InstructionManual` can be
    created *without* knowledge of the underlying decomposition. This makes it
    considerably more efficient to compute :class:`InstructionManual`s once
    per shape, and then look them up later with the desired decomposition in
    hand. The :attr:`lookup_dict` does just that, mapping shapes to manuals.

    Attributes:
        target (int): The target of the :class:`InstructionManual`s.
        lie (:class:`Lie`): The free Lie algebra.
        bounds (:class:DegreeBounds): Bounds the degree of Lie.
        lookup_dict (dict of partition: :class:`InstructionManual`): Provides
            mapping from shape partitions to instruction manuals.
    """

    def __init__(self, target, lie=None):
        """Computes the lookup dict.

        Args:
            target (int): The target size for the instruction manuals.
        """
        self.target = _positive_degree(target)
        self.lie = Lie(self.target) if lie is None else lie
        if self.lie.max_degree < self.target:
            raise ValueError("The supplied Lie data does not reach the target degree.")
        self.bounds = DegreeBounds(self.target, self.lie)
        self.lookup_dict = self._compute_lookup_dict()

    def __getitem__(self, key):
        """Return the instruction manuals indexed by a shape.

        Args:
            key (partition): The shape of a decomposition.

        Example:
            sage: lookup = InstructionLookup(4)
            sage: shape = Partition([2,1])
            sage: lookup[shape]
            Manual(target=4, shape=[2, 1], n_instructions=1)
        """
        return self.lookup_dict[key]

    def __iter__(self):
        """Return an iterator of the ``self.lookup_dict``."""
        return iter(self.lookup_dict)

    def __contains__(self, key):
        """Checks if key is in ``self.lookup_dict``.

        Args:
            key (partition): A shape of a decomposition.
        """
        return key in self.lookup_dict

    def __str__(self):
        """Return a string representation of ``self``."""
        return str(self.lookup_dict)

    def __repr__(self):
        """Return a string representation of ``self``."""
        return self.__str__()

    def _compute_lookup_dict(self):
        """Computes ``self.lookup_dict``.

        For each shape, first accesses the upper bound in ``self.bounds`` to
        cap the degree of ``self.lie``. Then computes the corresponding
        :class:`InstructionManual` for that shape.
        """
        d = {}

        for shape, bound in self.bounds.iteritem():

            lie_sizes = self.lie.sizes_up_to(bound)

            manual = InstructionManual(shape, lie_sizes, self.target)

            if manual.n_instructions > 0:

                d[shape] = manual

        return d


class CoefficientTable(object):
    """Data-structure holding the composition factor coefficients.

    The composition factor coefficients are indexed by pairs of partitions, a
    so-called lambda-partition and a mu-partition. This data-structure is a
    nested dictionary that allows for quick look-ups.

    Attributes:
        target (int): The size of the lambda-partitions indexing the inner
            dicts.
        table (dict of partition: dict of partition: int): The nested dict
            holding the coefficients.

    Examples:
        sage: table = CoefficientTable(6)
        sage: lambda_ = Partition([4,2])
        sage: mu = Partition([3,2])
        sage: table[lambda_][mu]
        0

    All initial values are 0 except in the case of 'diagonal-entries', that
    is, entries with the first and second partitions agree, where it is always
    1.

        sage: table = CoefficientTable(6)
        sage: lambda_ = Partition([4,2])
        sage: table[lambda_][lambda_]
        1
    """

    def __init__(self, target):
        """Initialise ``self``.

        Sets up nested dictionary as a placeholder for the coefficients.

        Args:
            target (int): The size of the lambda-partitions indexing the inner
                dicts.
        """
        self.target = target
        self.table = self._initialise_table()

    def __getitem__(self, key):
        """Return value of ``self.table`` indexed by ``key``.

        Args:
            key (partition): A lambda-partition.
        """
        return self.table[key]

    def __str__(self):
        """Return a string representation of ``self``."""
        return str(self.table)

    def __repr__(self):
        """Return a string representation of ``self``."""
        return self.__str__()

    def _initialise_table(self):
        """Initialise dictionary storing composition factors coefficients.

        Initially coefficients are set to zero, except for those indexed by
        the same partition, in which case the default value is 1. This is
        because it can be shown that every lambda-partition contains itself as
        a mu-partition.
        """
        table = {}

        for lambda_ in Partitions(self.target):

            init_dict = {}
            for i in range(1, self.target+1):
                for mu in Partitions(i):
                    init_dict[mu] = 0

            table[lambda_] = init_dict
            table[lambda_][lambda_] = 1

        return table

    def add(self, schur_poly, mu, lr_coeff):
        """Updates coefficients in ``self.table``.

        Given a schur polynomial built from a MuPartition, updates the table
        of coefficients.

        Args:
            schur_poly (symmetric function in Schur basis): The symmetric
                function computed from assembling a decomposition.
            mu (partition): The partition from which the decomposition used to
                assemble the symmetric function `schur_poly` arose.
            lr_coeff (int): The iterated Littlewood-Richardson coefficient of
                said decomposition.

        Example:
            sage: schur_poly = 3*s([1,1]) + s([2])
            sage: mu = MuPartition([1])
            sage: lr_coeff = 2
            sage: table.add(schur_poly, mu, lr_coeff)
            sage: table
            {
             [2]: {[2]: 1, [1, 1]: 0, [1]: 2},
             [1, 1]: {[2]: 0, [1, 1]: 1, [1]: 6}
             }
        """
        for lambda_, multiplicity in schur_poly:
            lambda_ = Partition(lambda_)
            self.table[lambda_][mu] += multiplicity * lr_coeff


class Decomposition(object):
    """A collection of partitions.

    A decomposition of a partition is a collection of sub-partitions with the
    property that the sum of sizes of the sub-partitions is the size of
    original partition.

    Attributes:
        partitions_tuple (tuple): A tuple of partitions.
        overcount (rational): A number of the form ``1/n`` where `n` accounts for
            repititions in ``self.partition_tuple``.
    """

    def __init__(self, partitions_tuple):
        """Initialise ``self``.

        Args:
            partitions_tuple (tuple): A tuple of partitions.
        """
        self.partitions_tuple = partitions_tuple
        self._overcount = None

    def __getitem__(self, key):
        """Return partition of ``self.partitions_tuple`` indexed by key.

        Args:
            key (int): An index of ``self.partitions_tuple``.
        """
        if isinstance(key, slice):
            return Decomposition(self.partitions_tuple[key])
        return self.partitions_tuple[key]

    def __iter__(self):
        """Return an iterator of ``self.partition_tuple``.

        Yields:
            partition: The partitions of ``self.partition_tuple``.
        """
        return iter(self.partitions_tuple)

    def __str__(self):
        """Return a string representation of ``self``."""
        return "Decomp({})".format(self.partitions_tuple)

    def __repr__(self):
        """Return a string representation of ``self``."""
        return self.__str__()

    def __len__(self):
        """Return length of ``self.partitions_tuple``."""
        return len(self.partitions_tuple)

    def __eq__(self, item):
        """Return true if the underlying partitions_tuple are in agreement.

        Args:
            item (:class:Decomposition): A Decomposition to be compared with
                ``self``.
        """
        if type(item) == tuple:
            return item == self.partitions_tuple
        else:
            assert type(item) == type(self)
            return item.partitions_tuple == self.partitions_tuple

    def iteritem(self):
        """Yield an (index, partition) tuple.

        Yields:
            tuple of (int, partition): A tuple of an index and a partition.
        """
        for index, partition in enumerate(self.partitions_tuple):
            yield index, partition

    @property
    def overcount(self):
        r"""A number to account for the overcounting caused by repitition.

        Repititions in ``self.partitions_tuple`` lead to unwanted
        multiplicity when assembling Schur polynomials (see
        :class:`InstructionManual`).

        Returns:
            rational: A number of form ``\frac{1}{n}`` where ``n`` is the number
                of repititions caused by the structure of the decomposition.

        Example:
            sage: p = Partition([2])
            sage: Decomposition((p, p)).overcount
            1/2

        """
        if self._overcount is None:
            self._overcount = self._compute_overcount(self.partitions_tuple)
            return self._overcount
        else:
            return self._overcount

    @staticmethod
    def _compute_overcount(partitions_tuple):
        """Computes the amount of overcount coming from this decomposition."""
        if len(set(partitions_tuple)) == len(partitions_tuple):
            return 1
        else:
            counter = Counter(partitions_tuple)
            res = 1
            for count in counter.values():
                res *= factorial(count)
            return Integer(1)/Integer(res)


class _LieSizeIndex(object):
    """Interfaces with list of sizes of Lie irreducibles.

    Essentially performs enumerate on lie_sizes, but with nicer interface.

    Attributes:
        pairs (list): List of :class:`_SizeIndexPair` tuples.

    Example:

    """
    def __init__(self, lie_sizes):
        """Enumerates lie_sizes and wraps with :class:`_SizeIndexPair`.

        Args:
            lie_sizes (list): Sizes of the irreducible Lie representations.
        """
        self.pairs = [_SizeIndexPair(tup) for tup in enumerate(lie_sizes)]

    def __str__(self):
        return str([pair.__str__() for pair in self.pairs])

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return iter(self.pairs)

    def copy_and_remove(self, pair):
        """Return a copy with an independent pair list and one pair removed."""
        new = copy(self)
        new.pairs = self.pairs.copy()
        new.pairs.remove(pair)
        return new


class _SizeIndexPair(tuple):
    """Wraps tuples of the form (index, size).

    Provides a simple wrapper to better interface with tuples of the form
    (index, size). The size is an integer corresponding to the degree of an
    irreducible representation in the free Lie algebra, and the index is the
    index of that representation in the list of irreducibles (as in the
    :attr:.`as_list` in :class:`Lie`).

    Attributes:
        id (int): The index.
        size (int): The size.
    """
    def __init__(self, index_size_tuple):
        """Initialise ``self``.

        Args:
            index_size_tuple (tuple): The first position corresponds to an
                index, the second to a size.
        """
        self.id = index_size_tuple[0]
        self.size = index_size_tuple[1]

    def __str__(self):
        """Overwrite __str__ method for tuple."""
        return "Pair(size={}, id={})".format(self.size, self.id)


class Visualisations(object):
    """Provides data visualisation for composition factors.

    We generate a lot of data in :class:`CompositionFactors`. Here we provide
    tools to help find and analyse patters in the data.

    The most general visualisation we provide is in the :meth:`display`, which
    returns a visual representation of the full matrix of coefficients stored
    in a :class:`CompositionFactors` object.

    Example:
        sage: cf = CompositionFactors(10)
        sage: vis = Visualisations(cf)
        sage: figure = vis.display()
        sage: figure.savefig("cf_degree_10.png")

    We explore stability phenomina in the category FI and the category PD.

    Example:
        sage: cf = CompositionFactors(10)
        sage: vis = Visualisations(cf)
        sage: vis.PD_stability()

    This returns a matplotlib visualisation of the coefficients stabalising in
    PD-direction. Concretely, we take two partitions ``lambda`` and ``mu``,
    and successively compute the coefficients adding one box to the top row of
    each partition as we go. We produce a plot displaying how the coefficients
    grow.
    """

    def __init__(self, cf):
        """Initialise ``self`` with a :class:`CompositionFactors` instance.

        Args:
            cf (:class:`CompositionFactors`): Computed composition factors.
        """
        self.cf = cf

    def display(self, resolution=float("inf"), width=20, ticks=True, *,
                cmap="Paired", normalization="linear"):
        """Displays an array representing :attr:`matrix`.

        When considered as an array of integers, the data can be visualised as
        with ``plt.imshow``. For large degrees, the coefficients can get very
        large, skewing potentially interesting lower degree information.
        Coefficients are capped at ``resolution`` to bring out these lower
        dimensional features.

        Args:
            resolution (int): (default ``np.inf``) Caps the coefficients
                appearing in :attr:`matrix`.
            width (int): (default `20`) The width to display the figure.
            ticks (bool or str): Partition ticks, no ticks, or ``"degree"``.
            cmap (str): Matplotlib palette name, or ``"historical"``.
            normalization (str): ``"linear"`` or ``"log"``.

        Returns:
            A figure displaying :attr:`matrix`, (possibly) with values capped
                at ``resolution``.
        """

        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, LogNorm

        if ticks not in (True, False, "degree"):
            raise ValueError("ticks must be True, False, or 'degree'.")
        if normalization not in ("linear", "log"):
            raise ValueError("normalization must be 'linear' or 'log'.")
        if not resolution > 0:
            raise ValueError("resolution must be positive.")
        if not 0 < width < float("inf"):
            raise ValueError("width must be positive and finite.")
        if cmap == "historical":
            palette = LinearSegmentedColormap.from_list(
                "historical_paired", plt.get_cmap("Paired").colors, N=256
            )
        else:
            palette = plt.get_cmap(cmap).copy()
        palette.set_bad("white", 1.0)
        mask = np.minimum(self.cf.matrix, resolution)
        mask[mask == 0] = np.nan
        norm = LogNorm() if normalization == "log" else None
        fig, ax = plt.subplots(figsize=(width, width))
        im = ax.imshow(mask, interpolation="none", cmap=palette, norm=norm)
        fig.colorbar(im, ax=ax)

        parts = [part for _, part in self.cf._partitions()]
        if ticks is False:
            ax.set_xticks([])
            ax.set_yticks([])
        elif ticks == "degree":
            counts = Counter(sum(part) for part in parts)
            positions, labels, start = [], [], 0
            for degree in range(1, self.cf.top_degree + 1):
                positions.append(start + (counts[degree] - 1) / 2)
                labels.append(str(degree))
                start += counts[degree]
            ax.set_xticks(positions, labels)
            ax.set_yticks(positions, labels)
        else:
            ax.set_xticks(range(len(parts)), parts, rotation="vertical")
            ax.set_yticks(range(len(parts)), parts)

        ax.tick_params(labelbottom=False, labeltop=True, labelleft=True)
        ax.set_xlabel("Mu degree" if ticks == "degree" else "Mu")
        ax.xaxis.set_label_position('top')
        ax.set_ylabel("Lambda degree" if ticks == "degree" else "Lambda")
        return fig

    def PD_stability(self, resolution=float("inf"), seq_length=2, width=20, *, unique=False):
        """We investigate adding one box to the top row of each partition.

        Plots a graph of how the coefficients change when adding one box to
        the top row of each partition ``lambda_`` and ``mu``. This is
        stability in the PD-direction.

        Args:
            resolution (int): (default: ``np.inf``) Cut off any values above
                this threshold.
            seq_length (int): (default: 2) Only display sequences of length at
                least ``seq_length``.
            width (int): The width of the figure to output.
            unique (bool): Keep only the earliest eligible start of each chain.

        Returns:
            A matplotlib figure plotting of the growing coefficients.
        """

        return self._plot_stability("PD", resolution, seq_length, width, unique)

    def stability_sequences(self, direction="PD", resolution=float("inf"),
                            seq_length=2, *, unique=False):
        """Return exact trajectory data without importing Matplotlib.

        Each record contains the starting ``lambda`` and ``mu`` tuples,
        ``sizes``, and integer ``coefficients``. Only positive starting
        coefficients are considered, as in the historical plots. With
        ``unique=True``, keep the earliest start satisfying both filters
        in each chain, not every suffix of that trajectory.
        """
        if direction not in ("PD", "FI"):
            raise ValueError("direction must be 'PD' or 'FI'.")
        seq_length = _positive_degree(seq_length, "seq_length")
        if not resolution > 0:
            raise ValueError("resolution must be positive.")
        sequences, seen = [], set()
        for _, lambda_ in self.cf._partitions():
            for mu, value in self.cf.coeffs[lambda_].items():
                if value < 1:
                    continue
                if direction == "PD":
                    key = (tuple(lambda_[1:]), tuple(mu[1:]), lambda_[0] - mu[0])
                    push = self._push_diagonal
                else:
                    key = (tuple(lambda_[1:]), tuple(mu))
                    push = self._push_down
                if unique and key in seen:
                    continue
                sizes, data = push(lambda_, mu)
                if len(data) >= seq_length and max(data) <= resolution:
                    seen.add(key)
                    sequences.append({
                        "lambda": tuple(int(part) for part in lambda_),
                        "mu": tuple(int(part) for part in mu),
                        "sizes": [int(size) for size in sizes],
                        "coefficients": [int(value) for value in data],
                    })
        return sequences

    def _plot_stability(self, direction, resolution, seq_length, width, unique):
        import matplotlib.pyplot as plt

        if not 0 < width < float("inf"):
            raise ValueError("width must be positive and finite.")
        sequences = self.stability_sequences(
            direction, resolution, seq_length, unique=unique
        )
        fig, ax = plt.subplots(figsize=(width, width))
        for sequence in sequences:
            ax.plot(sequence["sizes"], sequence["coefficients"], "o-")
        if direction == "PD":
            ax.set_xlabel("Sum of sizes of lambda and mu partition")
            ax.set_xticks(range(2 * self.cf.top_degree + 1))
        else:
            ax.set_xlabel("Sizes of lambda partition")
            ax.set_xticks(range(1, self.cf.top_degree + 1))
        ax.set_ylabel("Coefficient")
        return fig

    def _push_diagonal(self, lambda_, mu):
        """Computes list of coefficients in the PD-stable direction.

        Uses ``self.cf`` to compute the coefficints of ``lambda`` and ``mu``,
        adding a box to the top row of each

        Args:
            lambda_ (partition): Base outer partition for the stability check.
            mu (partition): Base inner partition for the stability check.

        Returns:
            List of sizes of lambda_ partitions and list their coefficients.
        """
        max_  = self.cf.top_degree
        size = sum(lambda_) + sum(mu)

        sizes, data = [], []
        while sum(lambda_) <= max_:
            c = self.cf.coeff(lambda_, mu)

            data.append(c)
            sizes.append(size)

            size += 2
            lambda_ = self._add_one(lambda_)
            mu = self._add_one(mu)

        return sizes, data

    def FI_stability(self, resolution=float("inf"), seq_length=2, width=20, *, unique=False):
        """We investigate adding one box to the top row of the lambdas.

        Plots a graph of how the coefficients change when adding one box to
        the top row of the partition ``lambda_``. This is stability in the
        FI-direction.

        Args:
            resolution (int): (default: ``np.inf``) Cut off any values above
                this threshold.
            seq_length (int): (default: 2) Only display sequences of length at
                least ``seq_length``.
            width (int): The width of the figure to output.
            unique (bool): Keep only the earliest eligible start of each chain.

        Returns:
            A matplotlib figure plotting of the changing coefficients.
        """

        return self._plot_stability("FI", resolution, seq_length, width, unique)

    def _push_down(self, lambda_, mu):
        """Computes list of coefficients in the FI-stable direction.

        Uses ``self.cf`` to compute the coefficients while adding a box to
        the top row of ``lambda`` and keeping ``mu`` fixed.

        Args:
            lambda_ (partition): Base outer partition for the stability check.
            mu (partition): Base inner partition for the stability check.

        Returns:
            List of sizes of lambda_ partitions and list their coefficients.
        """
        max_  = self.cf.top_degree
        size = sum(lambda_)

        sizes, data = [], []
        while sum(lambda_) <= max_:
            c = self.cf.coeff(lambda_, mu)

            data.append(c)
            sizes.append(size)

            size += 1
            lambda_ = self._add_one(lambda_)

        return sizes, data

    @staticmethod
    def _add_one(p):
        """Add a box to the top row of the partition.

        Args:
            p (partitions): The partition to which a box is added.

        Returns:
            A partition with one box added to the top row.
        """
        p_as_list = p.to_list()
        p_as_list[0] += 1
        return Partition(p_as_list)
