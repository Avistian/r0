# Bounded standalone-network CPU measurements

These measurements cover nine fixed configurations on a shared WSL2 ARM64 host,
with one PyTorch thread per fresh worker. They qualify these particular sampled
functions and sizes, not every seed, the Cartesian product of maxima, training
throughput, GPU operation or 100 million task execution.

The generator source was `343f3f117a2aec5b92a5aeb50e409f9d4c8e2258`, with no
source-module diff. Each case samples a three-block network of widths (8, 8, 1),
using named seed 28 and raw seed 29, then generates a regression task with seed 28
and task ID 9007199254740993. The root is Gaussian and the context contains half
the rows. The complete request and sampled function are in the retained archive.
Python 3.12.3, Linux 6.6.87.2 WSL2, twelve visible CPUs; concurrent verification
and agent work were active. Timing variation therefore includes shared-host load.

## Measured cases

All nine attempted cases passed; none was filtered or retried. Generation
produced finite inputs, scores and targets. Saved arrays and recipe records
matched after loading. Separately decoding the retained recipe and generating
again reproduced the original arrays, partitions and exact recipe JSON.

| Family | Rows | Features | Generate seconds | Regenerate seconds | Process peak MiB | Admission estimate MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 1024 | 8 | 0.0179 | 0.0694 | 250.20 | 8.16 |
| individual-sparse | 1024 | 8 | 0.0155 | 0.0626 | 250.47 | 8.16 |
| block-sparse | 1024 | 8 | 0.0152 | 0.0562 | 250.32 | 8.16 |
| skip | 1024 | 8 | 0.0167 | 0.0178 | 250.35 | 8.22 |
| residual | 1024 | 8 | 0.0611 | 0.0208 | 250.76 | 9.26 |
| irregular | 1024 | 8 | 0.0181 | 0.0593 | 252.48 | 8.66 |
| gated | 1024 | 8 | 0.0641 | 0.0305 | 253.44 | 11.82 |
| dense | 20000 | 8 | 0.2403 | 0.1827 | 288.44 | 157.85 |
| dense | 16 | 256 | 0.1545 | 0.2281 | 257.31 | 26.88 |

Generation timing includes replay validation of the sampled construction but
excludes imports and initial construction. Regeneration timing includes recipe
decoding and generation. The raw receipt additionally separates construction,
save and load time. Process peak memory is Linux ru_maxrss for the entire worker:
imports, construction, generation, multiple retained tasks, archive round-trip,
regeneration and evidence copying. It is **not** the conservative tensor working
estimate shown in the final column. Admission estimates are not process-memory
caps; these results do not justify lowering them or accepting wider combinations.

## Reproduction and retention

Run from the checkout with its existing environment:

```sh
python3 scripts/run_check.py python scripts/measure_network_resources.py   docs/evidence/network-resources-fresh
```

The runner refuses an existing output directory, executes every fixed case once
in a separate process with a 180-second deadline, and retains failures/timeouts
without retry. A complete summary requires both successful case assertions and
successful worker exit. Interrupted runs retain their started record and completed
case records; a missing summary is not success.

Raw measurements are ignored under
`docs/evidence/issue-28-resource-measurements/run-02/`. Each archive is retained in
ordered chunks of at most 900000bytes, with per-chunk and original-archive SHA256
in its case result. Concatenate the listed chunks in order to reconstruct task.zip;
copy task-receipt.json beside it as task.json. The original archive is validated
before chunking. Each summary records generator head, source diff and runner hash.

The first retained pass, run-01, checked archive save/load parity only. Run-02 adds
actual regeneration and fixes aggregate child-exit handling. A separate adversarial
runner probe supplies PASS receipts from nonzero child exits and confirms overall
failure; it is not generator evidence. Historical harness failure is retained.
No source-module behavior changed in this measurement slice.

## Remaining qualification

The 20,000-row case has only eight input features; the 256-feature case has only
sixteen rows. Neither establishes a 20,000-by256 workload or maximal network
complexity. One seed per family does not establish across-seed performance or
diversity. Root laws, native vectors, fitted lifecycles, typed converters, graph
topologies, sustained bulk throughput and hosted/GPU execution need their own
measurements. No predictive-quality claim follows from these checks.


## Combined size probe (2026-11-28)

At source head `35c6c95ae799ed0da5f8daec20795593bdf41901`, a separate
CPU probe attempted 20,000 rows, 256 features and 100 classes together for four
causal representations. Each request used a Gaussian 256-channel root, a frozen
100-output linear target, seed 28, 10,000 context rows, a 3 GiB declared working
estimate limit and a 120-second worker deadline. This is a simple graph, not a
maximum-complexity qualification. The existing full verification suite was running
concurrently.

| Representation | Result | Worker seconds | Process peak MiB |
| --- | --- | ---: | ---: |
| Functional | PASS: generated, saved, reloaded and checked | 5.849 | 884.75 |
| Native | Rejected before generation: working estimate exceeds 3 GiB | Not run | Not measured |
| Native vector | Rejected before generation: working estimate exceeds 3 GiB | Not run | Not measured |
| Converted | Rejected before generation: converter workspace exceeds 3 GiB | Not run | Not measured |

The functional worker reported 4.538 generation seconds and saved a 75,041,587-byte
archive. All saved inputs and scores were finite; labels were int64 in 0..99 and
the declared vocabulary contained 100 classes. An independent NumPy matrix product
and softmax calculation agreed with saved probabilities to maximum absolute error
`1.6653345369377348e-16`. This checks probabilities, not a separate oracle for the
random label draws. The worker reported aarch64, Python 3.12.3, NumPy 2.5.5 and
CPU PyTorch 2.15.0.

All four outcomes are retained under ignored
`docs/evidence/issue-28-typed-size-envelope/`, including requests where construction
succeeded, the functional archive, script and incremental `results.json`. An initial
harness failure used NumPy scalar weights where immutable records require Python
numbers; its log is retained separately. After that harness correction, each case
was attempted once. The three admission failures were not retried with larger
limits. Native estimates include conservative retained emission and JSON-copy costs;
the converted estimate includes vocabulary-by-channel workspace. These rejections
do not measure actual peak memory and are not generation passes. The aggregate
probe failed because only one of four cases passed. Typed combined-size support
under this cap remains unqualified; no GPU, sustained-throughput or maximal-network
claim follows from this probe.


## Bounded 100-class typed configurations

At source head `8a1ac39db28da9734ba05d7b1350b84574b3263b`, a separate
four-case probe used 512 rows, eight observed Gaussian features and
100 classes, with 384 context rows and the same frozen linear logits, seed,
3 GiB estimate cap and 120-second deadline pattern as the combined-size probe.
All four representations generated, saved and reloaded one dataset successfully.
This establishes a supported many-class configuration for each representation;
it does not reverse the larger workload's admission failures.

| Representation | Worker seconds | Process peak MiB | Probability oracle max absolute error |
| --- | ---: | ---: | ---: |
| functional | 1.358 | 261.23 | 1.110e-16 |
| native | 1.354 | 264.65 | 2.220e-16 |
| native-vector | 1.344 | 264.84 | 8.327e-17 |
| converted | 1.448 | 267.59 | 1.110e-16 |

All arrays were finite, labels were int64 in 0..99, and every schema retained its
100 declared classes. Independent NumPy matrix multiplication and softmax checked
all saved probabilities. This does not require each random sample to contain every
class and does not independently verify the random categorical draw. The functional
case uses a probabilistic target adapter; native, native-vector and converted cases
retain their respective typed target semantics. No classes were merged or discarded.
The script, full requests, four archives and results remain ignored under
`docs/evidence/issue-28-many-class/`. No GPU or sustained-throughput claim is made.
