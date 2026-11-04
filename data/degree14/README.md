# Corrected degree-14 checkpoint

This is the single current degree-14 dataset. It uses algorithm version 2 and
replaces the incorrect version-1 snapshot previously stored at this path.
The older data remains available in Git/LFS history, not a second dataset
directory. The correction changes **350 coefficients**: 50 lambda partitions
across seven mu columns of size 8, all in degree 12. Coefficients in degrees
1 through 11, 13, and 14 match the original.
All degrees 1 through 14 were freshly computed for this snapshot, without
resuming a historical checkpoint or reusing its coefficient rows.
`corrections.json` lists every changed coefficient with its old and new values.
For example, the coefficient at `lambda=(7,3,2), mu=(5,3)` changes from `66` to
`65`. The report records the exact number of changes, comparison scope, and hashes.

The original algorithm sorted decomposition parts only by size and length.
Distinct partitions `(3,1)` and `(2,2)` tie on both, so both orders survived
deduplication and were assembled twice. The corrected ordering also uses the
full partition; repeated identical parts retain their factorial correction.

## Download and use

From a checkout with [Git LFS](https://git-lfs.com/) installed:

```sh
git lfs install --local
git lfs pull --include="data/degree14/coefficients.json,data/degree14/composition_factors.sobj"
(cd data/degree14 && shasum -a 256 -c SHA256SUMS)
python compute.py inspect data/degree14
python compute.py query data/degree14 --lambda 5,3,2,2,1,1 --mu 2,1,1
```

The last command prints `51877`. Inspection, queries, and export need only
standard Python and exact JSON. For JSON-only access, fetch just
`data/degree14/coefficients.json`; the full checksum command also requires
the Sage payload.

Only `coefficients.json` and `composition_factors.sobj` use LFS. The main report,
the per-degree reports in `run.json`, the correction list, and checksums are
ordinary Git files. LFS pointer text is not the payload. Storage and downloads
count against the repository owner's allowance; CI does not download the data.

The `.sobj` file uses pickle and must only be loaded from a trusted source.
To create a current-source Sage checkpoint without loading the published
pickle, use the JSON-first migration command in the activated Sage environment:

```sh
python compute.py migrate data/degree14 --output results/imported-degree14
```

Migration validates all dimension identities and compares mu sizes up to 4
by default. It does not repair or change coefficients. The independently
verified fresh-computation scope is recorded separately in this snapshot's report.

## Fresh computation and provenance

The fresh run used algorithm version 2 and the pinned Sage 10.9 environment.
It started at degree 1 with no `--resume` argument and saved a checkpoint after
each degree through 14. Every checkpoint received a full independent comparison.

The final checkpoint agrees with independent, degree-truncated power-sum
plethysm on all **257,049 entries** of the implied full matrix, including
structural zeros. All **1,756 degree/column dimension identities** and Sage
save/load equality pass. Its **149,061 stored coefficients** also match the
earlier corrected reconstruction exactly.

| Measurement | Seconds |
| --- | ---: |
| Fresh coefficient computation, degrees 1-14 | 1,796.8 |
| Degree 13 alone | 136.0 |
| Degree 14 alone | 1,631.6 |
| Final full-matrix independent comparison | 120.7 |
| Computation loop, including all comparisons and checkpoint I/O | 2,024.0 |

In this snapshot's `report.json`, `computation_seconds` is the sum of all fresh
per-degree computation times, not wall time. `degree14_computation_seconds`
records just the final increment, and `direct_plethysm_seconds` records the
final full-matrix comparison. `run.json` retains all fourteen original
per-degree reports, source hashes, and the run's start, completion, and wall
times. Its hash is linked from `report.json` and included in `SHA256SUMS`.
Wall time spans the first degree's start message through final completion,
excluding interpreter startup.
The original reports describe each incremental step; their reused earlier
degrees were computed earlier in this same fresh run, not imported from
historical data.

The report also records the original source/data hashes, the correction count,
and agreement with the earlier reconstruction. No historical timing or partial
comparison is relabelled as a fresh result.

The mathematical module used for this run has SHA-256:

```text
a56b9947d9758d58172347a9c18de7ba28af83e9b7d0a0598976e076e92e0da1
```

`SHA256SUMS` records physical file hashes. The report's `coefficient_sha256`
instead hashes canonical coefficient records without their final newline.
The corrected canonical coefficient hash is:

```text
effb854ea250df454c0fb5d20390825d0457ef5385a0e7dca7c6b000959072e3
```

Save local computations to a separate output directory. This checked-in
directory contains the current corrected snapshot; earlier data remains in
Git/LFS history.

## Reproduce the fresh computation

Use the pinned Sage environment from `conda-linux-64.lock`. From the repository
root:

```sh
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python compute.py compute --degree 14 --output results/fresh-degree14 \
    --reference --progress-interval 60
```

The output directory must not exist. No original data is required. The final
checkpoint is `results/fresh-degree14/degree14`; its canonical coefficient
hash should match the value above. Timings and serialized Sage-file bytes need
not match. Per-degree CLI reports measure individual increments; this snapshot's
summary aggregates them as described above.

The checkpoint reader rejects the incorrect version-1 snapshot from Git history
because of its dimension mismatch. Recompute historical runs with algorithm
version 2; version-1 migration is allowed only through degree 11.
