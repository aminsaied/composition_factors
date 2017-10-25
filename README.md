# Composition Factors

Computes the composition factors of the n-th tensor power of the free
associative algebra in terms of coefficients `cf[lambda_][mu]` indexing the
terms in the irreducible decomposition.

This file implements:

- A fast algorithm computing the coefficients `cf[lambda_][mu]` for partitions `lambda_`, `mu`.
- A data-structure `Lie` capturing the representation theory of the free Lie algebra.
- Visualisations of the composition factors and certain stability phenomena.

## Examples

Compute all coefficients for partitions of size up `n` with `cf = CompositionFactors(n)`.  

Given two partitions `lambda_` and `mu`, get at the coefficient `cf[lambda_][mu]` as follows.

~~~~
sage: cf = CompositionFactors(6) # computes all coefficients of degree <= 6
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

<img src="/img/cf_7_example.png" width="750">

We also provide a `Visualisations` class to investigate new stability phenomena. Here we investigate PD-module stability among the coefficients. Concretely, this is the stability that occurs when you add one box to the first row in each partition `lambda_` and `mu`. The method `PD_stability` plots how the coefficients evolve under this stability.

~~~~
sage: cf = CompositionFactors(10)
sage: vis = Visualisations(cf)
sage: vis.PD_stability()
~~~~

<img src="/img/PD_stability_cf_degree_10.png" width="750">
