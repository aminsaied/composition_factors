# Whitney homology through degree 14

This directory stores independent signed-plethysm calculations, not a renamed
composition-factor checkpoint. It includes ordinary and extended Whitney
decompositions, both total and selected by block-size profile.

The domain is characteristic zero and positive rank: `2 <= degree <= 14` and
`1 <= blocks < degree`. There are no positive-rank cases in degree one.
A complete run covers **91 bidegrees**, **493 block-size profiles**, and
**1,168 decompositions**: ordinary and extended for each total and each type.
All irreducible labels are included, including zero multiplicities.

The completed payload contains **637,564 coefficient records**, of which
**414,782 are nonzero**, in 35,729,238 bytes. The full calculation and its
validation took **46.47 seconds** in Sage 10.9; degree 14 took 28.30 seconds.
All 5,342 ordinary and 107,988 extended total coefficients agree with the
corrected composition-factor table. The run also passed 986 type dimension
checks and 40,566 ordinary invariant-component coefficient checks.

## Download and inspect

```sh
git lfs pull --include="data/whitney-degree14/decompositions.jsonl"
(cd data/whitney-degree14 && shasum -a 256 -c SHA256SUMS)
```

Only `decompositions.jsonl` uses Git LFS. The report, generator, documentation,
and checksums are ordinary Git files. Storage/downloads count against the
repository owner's LFS allowance. See the main README's TLS guidance if your
environment overrides public certificate trust; never disable TLS verification.

The payload is UTF-8 JSON Lines, with one complete decomposition per line.
It uses the same derived-output envelope as `whitney compute`:

- `degree` is the underlying set size; `blocks` counts blocks; `rank` is their difference.
- `kind` is `ordinary` or `extended`.
- `block_type` is `null` for a total, otherwise the decreasing list of block sizes.
- `lambda` labels the underlying-set representation; extended records also
  have `mu`, labeling the block representation.
- `multiplicity` is an exact nonnegative integer.
- `calculation` is `signed_plethysm`; `source` identifies the calculator's
  source hash, algorithm version, and Sage version.

For example, this reads the extended contribution of block sizes three, two,
and one, without Sage or pickle:

```python
import json
from pathlib import Path

with Path("data/whitney-degree14/decompositions.jsonl").open() as stream:
    for line in stream:
        result = json.loads(line)
        if result["kind"] == "extended" and result["block_type"] == [3, 2, 1]:
            print(json.dumps(result, indent=2))
            break
```

The JSONL payload is not accepted by the composition-factor checkpoint
reader, resume, or migration commands.

## Validation and provenance

`report.json` must have `status: "passed"`. The generator:

- Computes every requested total and type directly with `WhitneyCalculator`.
- Compares every total coefficient, including zeros, with the corrected
  composition-factor checkpoint via the independent partition-conjugation view.
- Checks that type-selected contributions sum to each total.
- Checks ordinary invariants against the trivial block-action component.
- Checks each type's ordinary dimension `n! / product(j^m_j * m_j!)` and its
  extended dimension, which is `blocks!` times that value.
- Requires complete bidegree/type/ordinary/extended coverage before publishing
  the payload and a passing report.

The report records source hashes, the calculator commit, reference coefficient
hash, per-degree timings, exact coverage counts, and the payload's physical
SHA-256 and byte size. The generator is separately identified by its source
hash; the calculator commit identifies the code used before the data was added.
These checks establish numerical consistency, not a Lean proof or a
chain-level homology calculation.

## Reproduce

Use the pinned Sage environment described in the main README. The reference
checkpoint is used for verification, not as input to the Whitney calculation.
The output directory must not already exist:

```sh
mkdir -p results
python data/whitney-degree14/generate.py --degree 14 \
  --reference data/degree14 --output results/whitney14 \
  --calculator-commit "$(git rev-parse HEAD)"
```

Progress is written to `report.json`. The payload retains a `.tmp` suffix
until computation finishes; failed runs are marked failed and must not be
published. Timings and commit provenance may differ between reproductions.
