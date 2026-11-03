# Composition Factors

Computes the composition factors of the n-th tensor power of the free
associative algebra in terms of coefficients `cf[lambda_][mu]` indexing the
terms in the irreducible decomposition.

This file implements:

- A fast algorithm computing the coefficients `cf[lambda_][mu]` for partitions `lambda_`, `mu`.
- A data-structure `Lie` capturing the representation theory of the free Lie algebra.
- Visualisations of the composition factors and certain stability phenomena.

There is a [white paper](https://arxiv.org/abs/1711.04326) describing this algorithm on the arXiv.

## Installation

This is **SageMath code**, not a standalone NumPy program. It uses Sage's
partitions, symmetric functions, exact rational arithmetic, and Littlewood-
Richardson coefficients. Run it with Sage's Python, not an unrelated system
Python interpreter.

With [Miniforge](https://github.com/conda-forge/miniforge) installed:

```sh
conda env create -f environment.yml
conda activate composition-factors
sage --version
python -c 'from composition_factors import CompositionFactors; print(CompositionFactors(6).coeff([4, 2], [1, 1, 1]))'
```

The final command prints `2`. The environment pins SageMath 10.9 and Python
3.12; run these commands from the repository root with the environment
activated. SageMath 10.9's Conda entry point does not accept the older
`sage -python` option; use the environment's `python` directly.

For the exact Linux x86_64 package set used for the CPU computations below,
replace the environment creation command with:

```sh
conda create -n composition-factors --file conda-linux-64.lock
```

For an interactive session, start `sage` and import the classes explicitly:

```python
from composition_factors import CompositionFactors, Lie, Visualisations
```

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

## Headless computations and saved results

On a CPU server without a display, select Matplotlib's noninteractive backend:

```sh
MPLBACKEND=Agg python - <<'PY'
from composition_factors import CompositionFactors, Visualisations

cf = CompositionFactors(6)
print(cf.coeff([4, 2], [1, 1, 1]))
cf.save("composition_factors_6")
cf.display(width=10).savefig("composition_factors_6.png", dpi=150)
Visualisations(cf).PD_stability(width=10).savefig("pd_stability_6.png", dpi=150)
PY
```

Reload with `CompositionFactors.load("composition_factors_6")`. Sage's `.sobj`
files use pickle: only load files from a trusted source. The nested `coeffs`
dictionary contains exact coefficients; the NumPy `matrix` is a floating-point
view intended for visualisation, not exact large-integer arithmetic.

Stability plots include the highest computed degree. `seq_length` is an
inclusive minimum; FI plots use the size of the lambda-partition, and PD plots
use the sum of the two partition sizes.

Start with degree 6 before attempting larger computations: the number of
partitions, decompositions, and assembly instructions grows rapidly. No GPU is
required. Repeated Schur plethysms are cached within each degree's Lie instance;
the cache is released with that computation rather than retained globally.

### Verified CPU example

With SageMath 10.9 and Python 3.12.16, computation through degree 10 gives
138 partitions and 11,313 stored coefficients, of which 4,668 are nonzero.
All 19,044 pairs in the full matrix agree with a separate truncated power-sum
plethysm calculation. The documented coefficients `2` and `84` are reproduced,
and saving, reloading, and all three headless plots work.

On the same two-CPU Linux devbox, a fresh-process `CompositionFactors(10)`
computation took 36.2 seconds with compatibility repairs alone and 6.1 seconds
with plethysm caching. These are single-run timings excluding imports, plots,
and the independent comparison, not a general performance guarantee. Checks
through degree 10 do not establish correctness in arbitrary degree.

## Larger CPU runs

Run larger computations without plotting first, and save the exact coefficients
before doing additional analysis. In the activated environment on Linux, this
example limits the whole command to ten minutes, the Python process's virtual
address space to 12 GiB, and BLAS/OpenMP to one thread:

```sh
timeout --signal=TERM --kill-after=20s 600 \
  env MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u - <<'PY'
import resource
from time import perf_counter

resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))

from composition_factors import CompositionFactors

degree = 12
print(f"Computing through degree {degree}...", flush=True)
start = perf_counter()
cf = CompositionFactors(degree)
seconds = perf_counter() - start
cf.save(f"composition_factors_{degree}")
print(f"Computed in {seconds:.1f} seconds; matrix shape: {cf.matrix.shape}")
print(f"Peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MiB")
PY
```

The `resource` limit is an address-space ceiling, not a reservation of RAM.
Each invocation of this example computes all degrees up to the requested degree
from scratch.
If the timeout interrupts computation before `save`, no result for that run
is saved; previously saved files remain available. Reload a completed result
to plot it without repeating the computation.

On the same two-CPU devbox and pinned environment, degree 12 completed in
60.6 seconds: 271 partitions, 43,044 stored coefficients, and 17,016 nonzero
coefficients. Peak process RSS, including save/load and comparisons, was
413 MiB. All 11,313 stored coefficients through degree 10 were unchanged.
A separate direct-plethysm comparison matched 2,981 pairs: every lambda row
against the 11 mu-partitions of sizes 1 through 4. This is a partial independent
check of degree 12, unlike the complete matrix comparison through degree 10.

A degree-14 attempt reached the ten-minute wall-time limit without a completed
result. It was stopped by the configured timeout, with no reported out-of-memory
event. This does not establish that degree 14 cannot finish with more time.

### Checkpointed computations

`compute.py` saves each completed degree before moving on. It uses the same
per-degree algorithm as `CompositionFactors`, without generating plots:

```sh
python -u compute.py --degree 12 --output results/first-run
python -u compute.py --degree 14 --resume results/first-run/degree12 \
  --output results/resumed-run
```

The resumed run computes only degrees 13 and 14, saving them under
`results/resumed-run/degree13` and `results/resumed-run/degree14`. Each checkpoint
contains exact `coefficients.json`, a reloadable `composition_factors.sobj`, and
`report.json` with counts, source/coefficient hashes, incremental timing, and
peak process RSS. **Each timing measures the new degree only**, not a fresh
computation through that degree. RSS is the peak of the current process.

Use a **new output directory for each invocation**; existing directories are
never overwritten. After an interruption, resume from the highest degree whose
`report.json` was saved, using another output directory. Work within an
unfinished degree is not resumed. Earlier checkpoints remain available even if
a later computation or optional comparison fails.

Only resume trusted checkpoints: `.sobj` uses pickle, and the hashes are
integrity checks, not a security boundary. Resumption requires the same
`composition_factors.py` source hash, matching coefficient data, and a
successful save/load record. A checkpoint with a failed independent comparison
is rejected. A pending comparison is reported as unfinished, not passed.

For a bounded Linux run in the activated environment:

```bash
set -o pipefail
mkdir -p results
timeout --signal=TERM --kill-after=20s 7200 \
  env MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  bash -ec 'ulimit -v 12582912; exec python -u compute.py \
    --degree 14 --resume results/first-run/degree12 \
    --output results/bounded-run' 2>&1 | tee results/bounded-run.log
```

This uses a two-hour total budget, a 20-second kill grace, and a 12 GiB
virtual-address-space limit. The command runs in the foreground; use your
scheduler or terminal multiplexer for an unattended job. Checkpoints and logs
are ordinary local files, with no dependency on a particular cluster, account,
or home directory.

For an optional independent comparison, add `--reference`. This checks the full
matrix at every newly computed degree and can be expensive. To compare only
mu-partitions of sizes at most 4, add both `--reference` and
`--reference-max-mu-size 4`; intermediate degrees use the smaller applicable
bound. Reports record the actual number of pairs compared. A checkpoint is
saved before this comparison, and a mismatch produces a nonzero exit.

### Completed degree-14 run

A subsequent two-hour-budget run successfully extended the saved degree-12
result through degrees 13 and 14 on the same CPU host and environment. The new
degree-13 computation took 202.5 seconds; degree 14 took 1,607.8 seconds
(26.8 minutes). The whole resumed job, including imports, serialization, and
comparisons, took 30 minutes 39 seconds. These are incremental timings, not
fresh-process computations of all degrees through 13 or 14.

The degree-14 result contains 507 partitions, **149,061 stored coefficients**
and **56,557 nonzero coefficients**. Its full visualization matrix has
**257,049 entries**, matching the thesis's reported degree-14 matrix size.
Matching that size alone is not a comparison against the original dataset.
Peak process RSS was 826 MiB. All 80,616 stored coefficients through degree 13
were preserved. The independent comparison matched 5,577 pairs: all 507 lambda
rows against the 11 mu-partitions of sizes 1 through 4, not the whole matrix.
The run used the same mathematical source as this repository and a precursor
of the portable checkpoint runner above.

## Regression checks

The regression suite covers the documented coefficient, degree-three matrix
and direct plethysm, save/load, independent instruction copies, decomposition
iteration, headless stability plots, checkpoint resumption, integrity checks,
and failed independent comparisons:

```sh
MPLBACKEND=Agg python -m unittest discover -s tests
```
