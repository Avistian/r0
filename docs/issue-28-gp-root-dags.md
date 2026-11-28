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

Remaining required work includes independent review; the workbench and response
curves; matched/fresh graph interventions and fixed-fit query lifecycles; broader
sampled/fitted mechanism wrappers and categorical calibration policies; fit-graph
inspection; mixed nested layouts and policies; measured resource/diversity
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
