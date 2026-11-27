# Independent XGBoost construction workbench

The guarded workbench exposes three CPU histogram shared-vector fitting presets:
regression values, independent binary probabilities, and multiclass softmax vectors.
Each preset includes the complete editable fit plan: geometry, rows, estimator count,
depth, histogram bins, learning rate, row and column subsampling, minimum child weight,
L1/L2 penalties and split penalty. Root and nonroot equation replacement is supported;
all selected indices and dimensions are checked before any backend fitting begins.
Known aggregate fitting workspace is checked before construction streams are created.
The fitted envelope subsequently admits actual retained model and replay storage.

These presets fit independent iid normal inputs and independently drawn targets once.
They do not use ancestor populations or context/query rows. Structural probability
outputs feed the enclosing task adapter and are not native categorical task labels.
Reconstruction uses the construction seed; Next and replay retain the fitted model.
Exact native model bytes, parameters, raw synthetic draws and RNG states are archived.
Backend internal draws are not individually captured. Mechanism edits create linked
authored equations without misrepresenting the edited equation as its original fit.

This is one required fitting lifecycle, not completion of issue #28 or reproduction
of an entire paper. Ancestor/support/batch fitting, source parameter laws, broader
categorical input variants and fitted provenance across native/converted envelopes
remain separate implementation work. Hosted Colab and GPU execution are not claimed.

Retained development evidence is gitignored under
`docs/evidence/issue-28-xgboost-workbench/`. It includes genuine missing-construction
RED, corrected JSON-list test oracle, focused fit/graph checks, guarded session and
local notebook callback checks, static checks and browser receipts when completed.
Full-suite acceptance and independent workbench review remain pending.

## Authored edit transport

Browser JSON has one numeric type and serializes integral real values as integer
tokens. The guarded authored-edit path explicitly adapts XGBoost initial margins,
thresholds and leaf values to schema floats, including intentionally edited integer
values. Boolean/nonfinite values and noninteger child/feature indices still reject.
Metadata and model geometry are bounded before conversion. Stored graph/fit decoders
remain strict; this adaptation never runs during archive reopening or fit replay.
The original fitted archive remains intact when a linked authored edit is saved.

The first workbench review found this transport failure on a noise-only edit;
its REJECT and counterfactual evidence are retained in the original frozen tree.
The repair also routes tests through public module entry points and keeps
construction tests under `tests/generation/`.

## Causal response references

Fitted XGBoost and authored functional graph curves use an explicit causal reference,
not an assumed function of observational features. All observed channels are clamped
at both state and readout after their equations/pipelines. One varies across the
query grid; the others stay at the chosen row. Hidden root draws stay at that row.
Structural innovations are either fixed to their original-row draws or explicitly
zeroed. The curve returns raw target readouts without task normalization, observation
noise, separation or enclosing-task class probabilities. A structural equation may
itself output probabilities. This reference is not a conditional mean.

The guarded worker decodes the complete frozen fit archive under caller limits,
admits original-population plus query workspace before finite masks/latent arrays,
and evaluates retained functions without fitting a backend. Invalid limits or a
timeout leave the active archive unchanged. The public
`evaluate_functional_reference` API provides the same reference evaluation; support
population fits still require a separate partition-aware replay implementation.
