# Composition Factors

Computes the composition factors of the n-th tensor power of the free
associative algebra in terms of coefficients `cf[lambda_][mu]` indexing the
terms in the irreducible decomposition.

This file implements:

- A fast algorithm computing the coefficients `cf[lambda_][mu]` for partitions `lambda_`, `mu`.
- A data-structure `Lie` capturing the representation theory free Lie algebra.
- Visualisations of the composition factors.

Examples:
Given two partitions `lambda_` and `mu`, compute the coefficient `cf[lambda_][mu]` as follows.

~~~~
sage: cf = CompositionFactors(6)
sage: lambda_ = Partition([4,2])
sage: mu = Partition([1,1,1])
sage: cf[lambda_][mu]
2
~~~~

You can see the all coefficients up a chosen degree in with the `display` method.

~~~~

sage: cf = CompositionFactors(7)
sage: cf.display()
~~~~

![Composition Factors of degree at most 7](/img/cf_7_example.png)

We also provide a `Visualisations` class to investigate new stability phenomena. Here we investigate PD-module stability. Concretely, this is the stability that occurs when you add one box to the first row in each partition `lambda_` and `mu`. The method `PD_stability` plots how the coefficients evolve under this stability.

~~~~
sage: cf = CompositionFactors(9)
sage: vis = Visualisations(cf)
sage: vis.PD_stability()
~~~~

![PD-stability](/img/PD_stability_cf_degree_9.png)
