# XGBoost vector-leaf backend adapter — implementation in progress

The accepted #28 specification, decision 17, explicitly includes XGBoost structural
mechanisms. This satisfies the repository's condition that dependencies must be
required by the ticket. Earlier planning language saying no dependency authority
exists is superseded by that accepted requirement; sklearn GBDT remains distinct.

This initial internal adapter exports numerical joint vector trees from the pinned
CPU XGBoost 3.4.1 native JSON model and evaluates their shared splits, vector leaves,
base score, default missing routes and regression/logistic/softmax objectives.
Float32 input routing uses strict less-than; each additive stage rounds to float32.
Native vector leaf offsets are decoded separately from internal child offsets.
Categorical splits and scalar-leaf layouts explicitly reject in this adapter.
They are not claimed implemented by this numerical vector slice.

Primary references:
- [Official multi-output documentation](https://xgboost.readthedocs.io/en/stable/tutorials/multioutput.html).
- [Official vector-leaf release description](https://xgboost.ai/2026/08/25/introducing-the-xgboost-vector-leaf-model).
- [Official model IO documentation](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html).

An isolated `xgboost-cpu==3.4.1` target directory was used for backend verification;
the shared project environment was not changed. The online docs reported 3.4.2,
but the package index offered 3.4.1, which is the actually tested version.
Repository dependency/lock integration is still pending.

Author evidence: 12 literal routing/rounding/link/admission/rejection tests pass.
An actual backend comparison covers three objectives, widths 2/3/7, five rounds,
missing values and float32 neighbors around every split: 18 margin/output groups
pass. All margins match exactly in these cases; largest probability discrepancy
is 5.960464477539063e-08. Raw scripts/results are ignored under
`docs/evidence/issue-28-xgboost-vector/`. Focused Pyright and Ruff pass.

This is not complete XGBoost delivery. Remaining: strict frozen archive decoder,
fit recipe and exact synthetic-data/RNG custody, aggregate resource admission,
shared graph/evaluation integration, population lifecycle variants, public exports,
UI/Colab, wider backend cases, independent review and full integration checks.
The numerical exporter is an internal native-model boundary, not the public
untrusted recipe decoder. Full #28 completion is unchanged.

## Shared frozen graph adapter

The follow-on adapter registers the exact frozen vector model with the common
function protocol, parent composition, archive decoder, bounded channel
certificates and per-call aggregate admission. Torch evaluation uses float32
strict-less routing and stage accumulation on the input device, then returns
float64 channels. Shared graph inputs retain the existing finite-input contract;
the lower-level numerical backend oracle separately verifies NaN default routes.
Class objectives expose their probability vectors as explicit continuous outputs;
they do not silently turn them into native categorical node outcomes.

Public checks cover complete strict frozen records, caller limits, composed
parents, functional/native/vector/converted generation, repeated-row traces and
save/reopen with task IDs above 2^53. Nine fitted backend models (three objectives,
three widths) also compare the public Torch evaluator to actual CPU backend
predictions around split thresholds. CPU evidence is measured; GPU is NOT_RUN.
Synthetic fitting provenance, backend dependency lock and UI remain pending.

## Independent synthetic fitting and retained fit archive

`XGBoostFitPlan` and `fit_xgboost` implement an explicitly named independent
construction lifecycle. Three distinct injected NumPy streams draw standard-normal
fit inputs, normal regression targets or uniform discrete labels, and one backend
seed. All raw values, initial/final snapshots, exact CPU parameters and native model
JSON are retained with the exported function. Backend failures roll back all three
streams. Labels and inputs are synthetic only; this API accepts no external rows.

Versioned controls cover dimensions, rows, rounds, depth, histogram bins, objective,
learning rate, row/column subsampling, child-weight threshold and L1/L2/split-gain
penalties. Admission charges the entire raw construction and a conservative
histogram/record workspace before RNG copying or backend import. These bounds are
not measured Cartesian-product capacity or bulk throughput.

The pinned `xgboost-cpu==3.4.1` dependency is now declared and locked; the shared
verification environment has not been synced. Tests use an isolated backend target.
One-output regression/logistic fits explicitly export the backend's degenerate
scalar storage into one-coordinate leaves; multiclass scalar forests remain a
separate layout, not mislabeled as shared vector trees.

`xgboost_fit_from_dict` requires complete nested fields, applies caller limits,
replays raw RNG custody, and binds the exported function to its retained native
model. Archive reading does not import or refit XGBoost. This verifies record
consistency; it is not an independent refit proving that arbitrary externally
supplied model bytes were trained on the supplied draws. The author tests perform
actual independent backend refits for all five engines and all three objectives.

Current fitting tests also cover missing nested fields, exact caller state, failed
backend rollback and admission before backend import. Fit-provenance integration
into generated graph task archives, source-specific population/support/batch
lifecycles, categorical backend splits, exploration UI/Colab and independent
acceptance remain pending. Prior completed checkpoints remain separate.
