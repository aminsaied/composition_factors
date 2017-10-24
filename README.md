# composition_factors

Computes the composition factors of the n-th tensor power of the free
associative algebra in terms of coefficients `cf[lambda_][mu]` indexing the
terms in the irreducible decomposition.

This file implements:

    - A fast algorithm computing the coefficients `cf[lambda_][mu]`.

    - A data-structure, `Lie`, capturing the representation theory free Lie algebra.

    - Visualisations of the composition factors.

Examples:
Given two partitions `lambda_` and `mu`, compute the coefficient `cf[lambda_][mu]` as follows.

~~~~
sage: cf = CompositionFactors(6)
sage: lambda_ = [4,2]
sage: mu = [1,1,1]
sage: cf[lambda_][mu]
2
~~~~

You can see the all coefficients up a chosen degree in with the `display` method.

~~~~

sage: cf = CompositionFactors(7)
sage: cf.display()
~~~~

[[https://github.com/aminsaied/composition_factors/blob/master/img/cf_7_example.png|alt=Composition factors of degree 7]]
