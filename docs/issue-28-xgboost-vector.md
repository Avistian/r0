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
