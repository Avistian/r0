# Finite nearconstant task summaries

Frozen53ed topology target-reference review exposed an inherited valid-draw
inspection failure:20 histogram bins cannot fit into a range only a few floating
point steps wide. The same guarded authored multiclass construction fails on
baseline460c88b. Entropy is finite; no dataset retry or parameter change is needed.
The exact failing controls and both counterfactual traces remain in
`/tmp/r0-28-topology-target-reference-review/docs/evidence/issue-28-topology-target-independent/`.

Inspection now checks whether the proposed20 bin boundaries remain distinct in
scaled and displayed coordinates. For an unresolvable range, it deduplicates the
representable display boundaries, explicitly includes the original extrema, and
counts untouched values with explicit bins. Constant/scaling-collapsed summaries
retain the actual minimum and maximum. Ordinary resolvable ranges keep the prior
20-bin path. This changes summary construction only, not generated tasks, class
probabilities, references, draws or calibration.

Public verification covers the original guarded multiclass construction and
literal constant, zero, adjacent positive/negative float, two-ULP, subnormal and
extreme-finite counts. Outside-test evidence checks finite JSON and conservation
of all rows without changing inputs. Ordinary regression/binary/multiclass full
views and task artifacts match the frozen baseline exactly. Actual Chromium
completes both topology families and all three task types, including all class
probability channels, output views, downloads and replay. Raw evidence lives in
ignored `docs/evidence/issue-28-finite-summary-histograms/`.

This is author repair evidence. Independent acceptance of this histogram delta
remains separate from the previously reviewed target adapter and original06e
workbench. No full suite, cloud invocation or distributional retry was performed.
