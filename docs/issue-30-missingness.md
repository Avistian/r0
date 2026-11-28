# Explicit missingness mechanisms (r0 extensions)

`draw_missingness(plan, inputs, rng=..., target=None, latent=None, limits=...)`
accepts a finite float64 pre-observation matrix. It returns full-width boolean
masks, conditional missingness probabilities, an input/driver/plan binding hash,
and initial/final NumPy generator snapshots. It does not modify inputs or a
structural equation. These are named r0 policies, not claims of reproducing a
particular paper/checkpoint.

| Mode | Probability and draw dependence |
| --- | --- |
| `mcar` | Explicit constant probability; independent uniform draw for each selected cell. |
| `row` | Explicit constant probability; one uniform draw per row, shared by selected columns. Rows remain present, possibly entirely missing; this does not delete observations. |
| `block` | Explicit constant probability; one draw per contiguous row block and selected column, repeated over that block. Final partial blocks are retained. Different selected columns draw independently. |
| `mar` | `sigmoid(intercept + sum(coefficients[j] * inputs[drivers[j]]))`; independent cell draws conditional on named observed drivers. Driver columns cannot also be selected for masking in this plan. |
| `mnar-self` | `sigmoid(intercept + slope * inputs[row, column])` per selected cell; independent cell draws. |
| `mnar-target` | `sigmoid(intercept + slope * target[row])`; independent selected-cell draws conditional on the privileged numeric target driver. |
| `mnar-latent` | Same logistic law using a separately supplied privileged latent row driver. |

Fixed-probability modes require `probability`; logistic modes require it to be
`None`. MAR uses explicit coefficients, while the other logistic modes use slope.
No standardization, fitted intercept or realized missing-rate correction is hidden
inside these laws. In particular, a logistic intercept is not an expected table-wide
missingness percentage. Invalid/inactive control combinations are rejected.

Uniforms are drawn row-major in selected-column order; row mode draws rows×1,
and block mode draws ceil(rows/block_rows)×selected-columns. Probabilities zero
and one still consume their declared draws. Empty/full realized masks, saturated
probabilities and all-missing rows/columns are retained. No repair, retries or
minimum observed-count condition is applied. Stable sigmoid evaluation avoids
exponential overflow; nonfinite linear logits fail rather than silently clipping.
Preflight and late failure preserve the caller-owned random stream.

The matrix is bounded to 20,000 rows and 256 columns with an explicit workspace
check. Target/latent drivers must be finite float64 vectors with one value per row.
Unused privileged drivers are rejected. The binding hash includes the complete
plan, input shape, input bytes and applicable driver bytes; changing values even
in currently unselected columns changes the binding. Generator snapshots also
retain engine/environment identity.

Independent checks cover probability formulas, exact uniform draws, row/block
sharing, untouched columns, initial-state replay, input binding, zero/full masks,
invalid conditioning and rollback on resource rejection or arithmetic overflow.

The kernel alone does not define a task archive. The single-stage observation
wrapper below retains raw inputs, driver provenance and schema; CLI integration
and ordered transformation composition remain pending. In a composed pipeline,
MAR conditioning columns must remain observed; checking only one plan is not enough
to claim that an arbitrary composition is MAR. Target-driven masks describe a
privileged generative assumption and must not expose query labels or driver arrays
to the learner. Categorical/tail-censoring/near-empty/latent structured families and
other required #30 variants remain open beyond these initial modes.

## Observation tasks and privileged archives

`observe_missingness(task, plan, seed=..., latent=None,
latent_provenance=None, limits=...)` now applies one mask to an existing CPU
float64 task. It retains original inputs, conditional mask probabilities and
any numeric target/latent driver in `ObservationTruth`. Target-driven policies
use the original target encoding; latent policies require an explicit caller
provenance label. This label records an assertion, not independent proof of the
latent driver's scientific origin.

The observed task keeps targets, scores, context/query indices and feature schema.
Its source/episode identities change to include the observation specification and
value binding. Its recipe record retains the plan, separate observation seed,
source identities, RNG snapshots and binding. PCG64 is keyed by a separate domain,
observation seed and original episode seed/task ID. Structural equations and their
mechanism record remain unchanged. Already-masked/observed tasks are rejected by
this initial wrapper; composing stages needs a separate ordered contract.

`save_group` places raw inputs, probabilities and optional driver in `truth.npz`,
with provenance in the privileged task card. `predictor.npz` contains only observed
inputs, context targets and their missing masks. Loading reconstructs and validates
the observation: masks, probabilities, source-value binding, RNG replay, present
values and identities must agree. Removing required truth while retaining the
observation declaration is rejected. This validation checks the observation layer
against retained source values; it does not by itself re-run every source generator.

`restore_observation_source(task)` returns retained complete source inputs and
original identities/recipe for separate source replay and `explain_generation`.
Tests compare restored source episodes and structural traces with the originals.
Direct structural inspection of the masked task is not yet an integrated trace;
use the explicit source-restoration helper. The higher-level CLI request transport
for observation plans is still pending.
