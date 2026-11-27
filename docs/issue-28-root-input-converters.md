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

## Option-level coverage

The authored activation surface advances `I2.converter.03` and
`I2.converter.03.graph`, together with the explicit corrected readout-only variant.
The root input stage supplies compositions needed for `I2.root.01`–`.07` and the
accepted broader root-law family. Source random-covariance/function sampling and
`I2.root.08`–`.11` mixture parameter laws remain separate requirements: the existing
two-mode input mixture is not the source Gaussian-mixture catalog. The inventory
remains 583 source/extension/disposition records plus 140 specification anchors;
these are traceability entries, not counts of completed algorithms.

## Verification at the frozen source

Source commit: `c997bc2dc98db7611d7aa78552ee95467093301e`.
Baseline: `6ba0bc2813b777749c9944d93b404ef0348168c3`.
The full `make check` passes **797 tests, one skipped**, all static gates, in
427.52 seconds. [Raw output](evidence/issue-28/legacy-research/0028-root-converter-check-c997bc2.txt).
The nine newly added saved workflows cover every converter variant and task type;
the existing nine pipeline workflows also remain green. The existing CUDA skip
is retained. The [CLI help](evidence/issue-28/legacy-research/0028-root-converter-cli-help.txt) still exposes
no train/eval commands, so smoke/eval-smoke are NOT_AVAILABLE.

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv make check
```

[Development history](evidence/issue-28/legacy-research/0028-root-converter-development.json) preserves
raw RED/GREEN and exploratory failures, including fixture/assertion mistakes.
Intermediate snapshots were not retained; those logs are development history,
not exact-revision acceptance evidence. The committed tests and full check above
provide the reproducible author check.

## Standards

Independent review finds **zero hard violations** and one nonblocking heuristic:
the root-law and innovation-law decoder blocks duplicate their conversion shape.
[Unedited report](evidence/issue-28/legacy-research/0028-root-converter-standards.txt). The advisory remains
visible; it is not a numerical or contract rejection.

## Spec

Independent component review finds **no actionable defects**: 306 checks pass,
including 209 numerical comparisons at predeclared rtol=atol=2e-11, maximum absolute
error 8.88e-16, and 17 expected contract rejections. Seven root laws under both noise
placements are reconstructed using independent NumPy equations and the documented
stream identity. Standard activation fitting, all converter branches, moments,
ancestry, frozen controls, and nine saved/reopened/replayed workflows are covered.
Exact stream replay is not statistical proof of independence or upstream RNG parity.
[Unedited report](evidence/issue-28/legacy-research/0028-root-converter-spec-review.txt),
[totals](evidence/issue-28/legacy-research/0028-root-converter-spec-totals.json).

Independent resource arithmetic matches **21 parameters / 162,432 estimated bytes**,
including the 256-row truncated-draw floor. Exact limits admit; either limit one
lower rejects. This remains an array estimate, not an RSS measurement.

All **168 legacy proposals** match the baseline byte for byte: 164 successes and
four identical retained numerical failures. The canonical record SHA256 is
`03a5e4bd1334a6d0721b342491a8bcf124bd5bc4998c41395265608cbcd67969`.
An audit checks 260 imported-source receipts, covering 100 distinct module names,
against both pinned Git bytes and the executed files. Eight published programs
match the exact executed bytes after an independent publication follow-up.
Ruff classified imports differently inside the repository, requiring six import
separator blank lines; the reviewer re-executed the actual published scripts and
retained a [separate report](evidence/issue-28/legacy-research/0028-root-converter-publication-review.txt).
The original package stayed unchanged. The initial verifier NumPy-scalar input mistake
is retained as HARNESS_ERROR, separate from product findings.

[Artifact custody](evidence/issue-28/legacy-research/0028-root-converter-spec-artifacts.json) identifies the
split ZIP, original manifest, duplicate-file reconstruction and Git-reconstructable
baseline source. [Recovery instructions](evidence/issue-28/legacy-research/0028-root-converter-artifact-recovery.txt)
verify both original manifests after restoring the archive. [Reproduction commands](evidence/issue-28/legacy-research/0028-root-converter-reproduction.txt)
use the existing environment and the published programs in
`scripts/verify_root_converters/`. The archive retains original attempts at its
root and the independently re-executed publication supplement under `publication/`.
Its `executed-programs/` snapshots match published bytes. Use fresh output paths
to preserve earlier attempts.

Scoped review totals: Standards zero hard / one advisory; Spec zero actionable
findings. Full issue #28 acceptance, actual browser/Colab interaction, GPU, supported
workload measurements and scientific comparisons remain outstanding.
