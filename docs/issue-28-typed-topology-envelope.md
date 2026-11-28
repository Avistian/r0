# Typed topology construction for issue 28

`TypedTopologyRequest` composes the explicit fixed-size or random-size redirection
hyperprior with functional, native scalar, native vector, and converted mechanism
families. It retains the sampled parent graph and passes it to the existing
mechanism samplers; geometry construction does not draw another graph.

The request exposes width, root and function priors, supported numerical pipeline
priors, calibration size, named sampling policies, and caller-owned resource limits.
Native roles independently specify roots, hidden variables, observed features and
the target. Readout roles override structural roles. Placement can be a retained
permutation or prefix features with the last node as target. Minimum topology size
must accommodate distinct native readouts. Native scalar width means continuous
channels; native vector width means variable count, with hidden-role domains for
additional variables. Converted roles retain explicit feature and target vocabularies
and converter priors. Task declarations must agree with guaranteed target semantics.

The five input streams belong to topology, named choices, raw mechanism draws,
calibration and fit. Failed construction rolls them back together. The archive
retains original/final snapshots, topology, geometry and the typed child record.
Decoding admits metadata and checks outer/child ownership before restoring streams,
then compares the complete replay. Public task generation, saved groups and generation
traces dispatch through the envelope.

This is a compositional extension, not a claim of exact paper reproduction. The
existing named redirection rounding and Gamma probability transforms remain explicit
r0 reconstruction choices. The broad option inventory remains authoritative; this
slice does not complete per-parent compositions, additional topology families,
workbench controls, population-aware policies, nested environments, or whole-issue
resource and scientific acceptance. Full-suite and independent acceptance of this
new envelope remain pending.
