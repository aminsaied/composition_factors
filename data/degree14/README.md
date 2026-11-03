# Degree-14 computation snapshot

This immutable checkpoint contains all positive degrees through 14:

- 507 partitions and 149,061 stored coefficients, including zeros.
- 56,557 nonzero coefficients; the implied full matrix has 257,049 entries.
- Maximum coefficient 51,877, at lambda `[5,3,2,2,1,1]`, mu `[2,1,1]`.

| File | Bytes | Storage |
| --- | ---: | --- |
| `coefficients.json` | 8,621,950 | Git LFS |
| `composition_factors.sobj` | 5,068,860 | Git LFS |
| `report.json` | 894 | Git |
| `SHA256SUMS` | 253 | Git |

The JSON and Sage payloads total 13,690,810 bytes (approximately 13.1 MiB).
The report and payloads are preserved byte-for-byte from the completed run.
The report contains no private host paths or credentials.

## Download and verify

Install [Git LFS](https://git-lfs.com/) before these commands. From the repository
root:

```sh
git lfs install --local
git lfs pull --include="data/degree14/coefficients.json,data/degree14/composition_factors.sobj"
(cd data/degree14 && shasum -a 256 -c SHA256SUMS)
python compute.py inspect data/degree14
python compute.py query data/degree14 --lambda 5,3,2,2,1,1 --mu 2,1,1
```

The final command prints `51877`. Inspection and queries use only Python's
standard library and do not unpickle the Sage file.

For JSON-only access, fetch just `data/degree14/coefficients.json` with the
`--include` option. The same inspection and query commands work without the
Sage payload. The full checksum command above requires all three listed files.

LFS pointer files are not the dataset. Use a Git checkout and `git lfs pull`;
do not assume a source ZIP or a raw pointer file contains the payloads.
`SHA256SUMS` hashes physical file bytes, as do the LFS object IDs. The report's
`coefficient_sha256` instead hashes canonical coefficient JSON without its
final newline, so those two JSON digests intentionally differ.

Hashes check consistency, not the trustworthiness of rewritten data. The
`.sobj` file uses pickle and must only be loaded from a trusted source.
For a fresh Sage checkpoint without reading the published pickle, use the
JSON-first migration command in the activated Sage environment:

```sh
python compute.py migrate data/degree14 --output results/imported-degree14
```

This reads exact JSON, independently compares mu sizes at most 4 by default,
and creates a new local checkpoint. It does not recompute degree 14.

## Provenance and comparison scope

The follow-up run completed on 2026-11-03 UTC using SageMath 10.9,
Python 3.12.16, and NumPy 2.5.4. It resumed a migrated degree-12 result and
computed degrees 13 and 14. The report's 1,519.2-second timing covers only the
new degree-14 computation; earlier degrees were reused.

The run used a pre-publication working-tree snapshot. Its mathematical source
is identical to `composition_factors.py` in repository commit
`349e5c18d50993468bd17a3f8ba1945958f07f18`, with SHA-256:

```text
6a142afb60e3d2f9a5cc3224baecb42560a5d91eeaeb19bd21d664cdd4d9dbde
```

That commit supplies the compatible public implementation and
`conda-linux-64.lock`; it is not presented as the commit executed by the run.
The original `runner_sha256` is retained in the report and identifies the
earlier CLI snapshot, before the final metadata-validation changes.

All 149,061 stored coefficients match the previously saved modernized
degree-14 result exactly. The canonical coefficient hash is:

```text
c6f514a80f62e7b54ef38d0f925bc33e2c0df4f778b3b8159609faa59bf59e60
```

The separate independent plethysm comparison checked 5,577 pairs: all 507
lambda rows against the 11 mu partitions of sizes 1 through 4. That is a
partial independent comparison, not a full-matrix check or a proof in arbitrary
degree. The unavailable original thesis integer table was not compared here.

Keep this snapshot unchanged. Save later computations in new directories;
LFS stores each changed payload as a complete new version. Downloads, including
CI downloads when enabled, count against the repository owner's bandwidth
allowance. The ordinary code-checking CI deliberately does not fetch these
payloads.
