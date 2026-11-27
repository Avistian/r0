# Spec review: SCOPED REJECT at 47630e8

**One P2 finding: probability-view underflow discards a valid extreme categorical law.**
The authoritative accepted spec says: “Preserve valid extreme draws; expose domain restrictions and numerical failures without silently clipping values” (`docs/specs/0028-causal-network-generators.md:86-87`).

`categorical_preferences.py:84` unconditionally requests the binary64 `probabilities` view during construction. The finite retained weights `(1.2247448713915892, 5e-324, 1.2247448713915892)` have positive mass and an exact inverse-CDF answer: unit `0.5` selects middle category 1. Constructor and JSON decoder both reject this law because dividing its smallest weight by the total rounds the displayed probability to zero. This is avoidable: the existing exact comparison needs no normalized floating-point probability. The increment document's declared failure at lines 69-72 conflicts with the accepted preservation requirement; documenting the extra rejection does not authorize it.

Preserve the record and exact choice. Keep an explicit error for an unrepresentable floating-point probability view, and expose finite log probabilities or exact masses for inspection. Retained independent reproduction: `preserve_extreme.py`, `extreme-reject/input.json`, `extreme-reject/result.json`.

No other implementation/scope-creep finding was established before stopping at this defect. Earlier scoped-document checks passed: 33 categorical/admission/malformed probes; four exact-boundary probes plus the documented-underflow rejection; 96 matrix configurations; nine binding attacks; three specialized-input guards; twelve profile replays; four backend save/reopen workflows. All 504 old-profile cases match the frozen baseline exactly, including four failed draws on both revisions. These passes do not override the accepted-spec rejection.

The remaining whole-issue features and browser/Colab/GPU/scale evidence stay pending; this review does not claim complete issue #28 delivery.
