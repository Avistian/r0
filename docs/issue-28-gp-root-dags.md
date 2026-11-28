# Finite GP inputs for causal DAGs

This increment binds retained finite GP feature columns to explicitly named root
inputs of authored functional, scalar-native, vector-native and converted DAGs.
Each binding preserves original row identity, selected column order and aliases,
including one GP column shared between roots. Unbound roots retain their existing
laws. The DAG readout defines the target; the GP source's marginal target remains
source provenance.

The public GPRootDAGRecipe participates in ordinary recipe decoding, generation,
saved groups, resume and original-row inspection. The complete source, graph and
bindings are admitted before covariance reconstruction. Shared source columns
participate in functional dependency diagnostics. Saved tasks retain the full
envelope and runtime mechanism card.

GPRootCalibration retains both the original DAG population and an explicitly
separately keyed population with the same graph and GP law. It fits fixed target
centers/scales on the independent population, with identity scaling for constant
channels. Functional regression/classification use the existing noise/label
adapter after scaling; continuous typed targets retain their numeric semantics.
Queries never fit the scaling constants. Fixed-location GP calibration accepts an
explicit matching calibration-location matrix.

Both envelopes support JSON replay, public shared generation, task archives and
traces. Run requests require exactly one retained source identity; malformed
transport controls, execution headers and source/episode linkage reject before
replay. Producer validation checks the full retained task, not just its recipe.
Calibration traces show the original DAG, retained fit statistics, normalized
signal and final adapter steps.

## Verification and remaining scope

Author evidence is ignored under docs/evidence/issue-28-gp-root-dags. The focused
checks include literal downstream equations across all four runtime families,
independent calibration arithmetic, aliases, constant/binary adapter semantics,
malformed-source preflight with forbidden covariance reconstruction, saved-task
custody, JSON/ZIP and repeated-row traces. Genuine failures and fixture errors are
retained separately. This is an implementation increment, not whole issue #28
acceptance or scientific replication.

Remaining required work includes independent review; response
curves; matched/fresh graph interventions and fixed-fit query lifecycles; broader
sampled/fitted mechanism wrappers and categorical calibration policies; mixed nested layouts and policies; measured resource/diversity
qualification; and local/Colab/CPU-GPU integration evidence. Calibrated output
currently uses explicit CPU replay. Independent fit populations use the current
finite GP cap of 32..1024 rows. These are current implementation bounds, not
exclusions from the accepted broad issue scope. The original ordinary DAG APIs
and their default root laws remain supported.

Calibration construction admits the complete future original/fit/calibration
envelope before replaying either population. GP receipt costs are symbolic and
exact under the existing metadata accounting law; no dummy covariance arrays are
allocated. The only value-dependent metadata length, the constant-channel list,
is bounded by the target width. Both GP and graph runtimes are checked against
caller and retained nested caps. Declared graph caps must fit their retained
envelope caps, which must fit the calibration caller caps; records are never
silently rewritten to make a lower cap appear compatible. These are conservative
admission accounts, not measured allocator or RSS figures.

## GP-root workbench

The construction catalog includes 27 authored GP-root examples across functional,
scalar-native, vector-native and converted graphs, with linear, convolution and
network root mechanisms. Regression and multiclass cover all four families;
functional binary examples retain the ordinary adapter. Complete GP request,
runtime and binding controls remain editable JSON. Construction admits the future
source metadata and runtime before drawing a population. The construction seed
does not modify authored equations; the explicitly retained episode identity
selects the GP population.

Load controls, exact replay and raw-population Next preserve large integer IDs.
Traces run through the guarded worker, retain requested row order and duplicates,
and show full source covariance, factors, locations, innovations and values.
Matrix downloads include root-input bindings; graph traces show the downstream
mechanisms. Failed construction preserves the active dataset. Calibrated
fixed-fit Next, graph edits and response/intervention controls still
require their own implementations; unavailable actions report this explicitly.

Author Chromium evidence covers six family/mechanism/target combinations, matrix
views, downloads, ZIP exports, replay, Next and invalid-draft recovery. Literal
checks on the saved browser artifacts reconstruct RBF matrices, GP draws, root
and descendant equations and classification probabilities. Evidence is ignored
under docs/evidence/issue-28-gp-root-workbench. These are bounded author checks;
independent review and a full check of the combined change remain required.

## Guarded independent GP fitting

The explorer can fit fixed target centers/scales on a separately keyed GP
population, preserving the active query graph, its source identity and both full
populations in the saved calibration envelope. Fit rows, seed, task identifier
and optional fixed-location matrix are explicit controls. Typed continuous
targets and functional regression/binary/multiclass adapters retain their
semantics. Direct categorical scaling remains unavailable until a
category-preserving policy is implemented. Fitting never uses the query rows.

Raw construction admits the complete future original/fit/calibration metadata and
runtimes before decoding the retained source. Known producer mismatches bind to
the requested original, independent fit identity and geometry before any GP work;
full canonical construction and tensor validation still follow. Transport
requires one exact retained query identity and CPU execution. Large fit IDs use
exact decimal-string browser transport. Failure keeps the active dataset; replay
retains the fixed fit. Fit controls disable on already calibrated datasets so a
second fitting action requires opening the original explicitly.

Author checks cover four continuous graph families, separate fit arithmetic,
invalid controls, coherent aggregate rejection before covariance reconstruction,
producer metadata and guarded replay. Six Chromium workflows cover four-family
regression plus functional binary/multiclass, invalid-fit rollback, exact IDs,
repeated-row traces, ZIP downloads and replay. Literal checks reconstruct fit
means/scales and query scores/probabilities from the downloaded populations.
Evidence remains ignored under docs/evidence/issue-28-gp-calibration-workbench.
Independent review and a full check of this child are still required. Fixed-fit
Next, matched/fresh interventions and broader
calibration policies remain required work.

GP-root workbench controls explicitly mark causal interventions, sampled policies,
causal-reference curves and generic task-only comparisons unavailable while their
retained-population integration remains required. Calibrated GP-root Next is also
unavailable until fixed-fit continuation is implemented; replay remains available.
These capability guards prevent unsupported workflows, including after busy-state
reset, and clear when returning to an ordinary supported dataset. Disabling them
is an interface correction, not completion or exclusion of those requirements.

## Independent fitting-population inspection

The generation graph has a population selector for calibrated GP-root datasets.
Query remains the default. Selecting the independent fitting population traces
its own retained row indices, including rows beyond the query population size,
with repeated indices and order preserved. Its covariance, factors, locations,
innovations and values use the complete fitting population and identity in the
matrix views and downloads. Returning to query restores the original population
matrices; neither operation changes the active dataset or calibration.

Public explain_gp_calibration_fit validates the entire active query envelope and
tensors, then replays the saved fitting graph under its retained identity. Invalid
fit row selections reject before covariance reconstruction. The guarded graph
worker carries the population choice and output limits. Large fit identities use
exact browser transport. Calibration statistics in both graph views have explicit
center/scale row labels and global axes, even when the user selects exactly two
rows; their statistics always refer to all independent fitting rows.

Author checks cover all four continuous graph families, literal fitting inputs
and aggregate statistics, repeated/out-of-query-range rows, malformed selections,
forged active tensors, and guarded session preservation. Independent acceptance
and the combined full check remain pending. This inspection increment does not
implement fixed-fit query continuation or the remaining intervention/reference
workflows.
