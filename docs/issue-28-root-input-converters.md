# Issue 28: independent root inputs and standard activation converters

This increment extends authored causal recipes and pipelines with a root input
draw before activation, separate from the node's structural innovation. It also
allows independently fitted standard activations in numerical converters.
The accepted [issue 28 scope](specs/0028-causal-network-generators.md) remains open;
this is a component implementation, not full paper-option acceptance.

## Recorded root equations

`RootInputCausalNode` is a parentless stochastic node with immutable
`root_exogenous`, `root_scale`, and `root_version="root-input-v1"` fields.
Its pre-noise affine response is `bias + root_scale * root_input`. Signed and zero
root scales are allowed. Root and later innovation laws independently support
Gaussian, uniform, Student-t, mixture, explicit truncation, ball geometry and
square linear factors through the existing `ExogenousDistribution` record.

```text
root input draw → bias + root_scale * draw → output activation
  → ordered pipeline stages → channel importance → L2 → converters
  → add separate structural innovation → node importance → descendants
```

The plain causal recipe additionally supports the existing `before_output` noise
placement. Pipelines retain their explicit `after_output` requirement. Root input
uses the additional `root-input-v1` stream prefix; changing context size, requesting
original-row inspection, or changing later noise scales does not refit the root
law or shift its random stream. Independent target calibration adds its existing
calibration prefix to both root and innovation draws.

The [pinned source-law audit](research/0028-node-pipeline-sampling-laws.md)
distinguishes source RandomPoints input from later Gaussian noise. This authored
surface supplies those separate positions. It does **not** yet sample or reproduce
the source RandomPoints random-function catalog, its per-call behavior, or source
node-construction preferences. The fixed laws here are explicit r0 compositions.

Signal-dependent noise uses the actual pre-noise root affine response, including
root scale. A zero root factor/scale removes its potential ancestry; an independent
later innovation retains a distinct source identity. Original-row traces expose
`root-input-i`, `exogenous-i`, `noise-scale-i`, and `innovation-i`, including the
root scale needed to reconstruct the noise response. The saved node record keeps
the complete frozen law, and malformed parented roots or mismatched widths fail.

## Converter choices and moment contracts

`NumericalConverter.transform` accepts identity, a one-channel frozen Kumaraswamy
transform, or a one-channel `CalibratedActivation`. The latter applies the existing
independent standardize/rescale/activate/standardize function, with its recorded
means, inverse scales, shift rows, scale, calibration identity and sample digest.
It is not merely a primitive activation with a different name.

All three existing explicit converter variants apply unchanged:

| Variant | State before later noise | Captured readout |
| --- | --- | --- |
| `historical-tuple` | transformed input | original input |
| `historical-graph` | transformed input | transformed input |
| `readout-only` | original input | transformed input |

The transform is evaluated even when its transformed value only feeds descendants
or only feeds the readout. Both its input and transformed output need certified
finite second moments for fitted standardization. A finite calibration sample is
not population moment evidence. For example, Student-t with df=1.5 fails the
input gate; df=3 followed by a square activation fails the output gate. A preceding
bounded tanh can establish both certificates. No clipping or replacement draw is
introduced. Root moment certificates also propagate through signal-dependent
noise, so an affine root with df=3 and squared noise response has certified order
below 1.5 rather than all moments.

Constant standard activations, such as one-channel softmax, remove potential
ancestry only on their declared state/readout branches. Later innovation remains
separate. Captured readouts exclude both later innovation and node importance.

## Admission and scope limits

Root records count their extra exogenous law and scale constants. The declared
working estimate adds root sampling/evaluation buffers and honors the existing
256-row floor for truncated root draws. These are array admission estimates,
not measured process RSS. Existing recipes acquire no new serialized fields or
root draws. Default root input is Gaussian; default later noise retains the
inherited stochastic-node value. Callers should set both scales explicitly when
choosing a particular root equation.

Authored calibration still cannot prove caller-supplied sample independence from
a digest alone. Sampled ancestral fitting, shared sampling preferences, standalone
source Standardize, complete source profiles and their census remain outstanding.
Actual browser/Colab interaction, GPU execution, throughput and scientific
comparison are not established by this component's API checks.

## Verification status

The source and public-workflow checks, independent numerical review, full suite,
and retained evidence are recorded below when completed. This section currently
marks component acceptance **PENDING**; it makes no unrun verification claim.
