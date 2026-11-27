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
