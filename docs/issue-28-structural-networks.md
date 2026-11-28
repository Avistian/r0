# Broad standalone frozen network architectures

`NetworkRecipe` generates a standalone random function, with input features as
roots and network outputs as regression or classification scores. Its architecture
is distinct from an SCM topology or a parent-function neural reducer. No network
weights are trained on labels. The old `FrozenMLPFunction` and eight-family joint
selector keep their original records and behavior.

## Architecture variety and combinations

`FrozenNetworkFunction` stores an ordered graph of `NetworkBlock` objects. State 0
is the input; block i produces state i+1. Each block concatenates its declared
parent states in order. Widths may differ between blocks. Each `NetworkAffine`
retains weights, bias, an optional Boolean mask, explicit scaling, and an activation
before or after the affine map. Masks are fixed structural connectivity, not new
random masks per row or a training-mode dropout switch.

| Sampled family | Realized equation |
| --- | --- |
| Dense | One affine/activation branch per layer. |
| Individual sparse | Independent frozen edge masks. An all-dropped draw stays valid. |
| Block sparse | Diagonal blocks with explicitly zero remainder rows/columns. |
| Skip | Sampled projections from earlier states, added to the nonlinear branch. |
| Residual | An independent projection from the immediately preceding state, supporting dimension changes. |
| Irregular feedforward | Sampled subsets of all earlier states; no chain edge is forced. Empty selections retain a zero-masked input placeholder and a bias-only function. |
| Gated | Several expert branches share the same input; input-dependent affine logits are softmaxed and weight the experts. |

Authored blocks can combine these mechanisms: sparse expert branches, projected
skips, irregular parent orders and gating can coexist in one network. The sampler
chooses one of the requested family presets per construction. The family catalog,
explicit width sequence/depth, activation catalog and placement, homogeneous or
independent activation selection, weight laws, biases, scales, mask density, block
ceiling, skip probability and expert count are configurable. Gaussian, uniform,
Student-t(df=3) and Cauchy weight draws are available; valid extremes and disconnected
networks are retained. Numerical failures raise rather than being clipped or retried.
Default fan-in scaling is a named r0 convention; it is not substituted for source
mask scaling. `block_limit` is a ceiling, bounded by available matrix dimensions.

The output activation is independently explicit. Activations here are primitive
frozen transformations and do not fit batch statistics. More learned/calibrated
architecture variants remain required elsewhere in the #28 inventory.

## Source mappings and explicit adaptations

[TabPFN-v1 §4.4, Appendix C and Table 5](https://arxiv.org/html/2207.01848v6)
motivate standalone BNNs and structural masks. `dropout_profile="p1-beta"` retains
`p=0.9*Beta(a,b)`, with independent `a,b ~ Uniform(0.1,5)`. The Beta draw uses two
independent unit-scale Gamma draws and their ratio. This preserves the stated law,
not the original implementation's seed sequence. The profile applies to individual
edge masks; block masks have their own block-count law.

The [pinned P1 sampler](https://github.com/automl/tabpfn-v1-prior/blob/778f0d0eb10f35ca85964fbc27ecb8baee9146a4/tabpfn_prior/priors/mlp.py)
provides two separate `mask_profile` adaptations. `p1-individual` exempts the first
layer and uses weight multiplier `init_std/(1-p**e)`, where e is 1/2 or 1. Its
explicit source cap is `p<=0.99`. `p1-block` samples a block count uniformly from
1 through ceil(sqrt(min(input_width,output_width))), keeps equal diagonal blocks,
and uses `init_std/keep_fraction**e`. Both sample affine biases uniformly within
±1/sqrt(fan_in). Source profiles require their matching sparse family, Gaussian
weights and disabled generic fan-in scaling. In this implementation raw weights
are sampled for the full matrix and then masked; RNG order/backend differs from
the source's block-only initialization. These are source-law adaptations, not
bitwise reproductions or complete P1 priors.

[MITRA-v2 §2.3](https://arxiv.org/html/2609.04540v1#S2.SS3) motivates direct projected
paths alongside nonlinear transformations. The standalone residual/skip/gated
architectures here are r0 extensions. They do not claim equivalence to MITRA's
SCM aggregation law, which is implemented separately as parent composition.

| Inventory IDs | This slice | Still required |
| --- | --- | --- |
| SPEC.N02.01–07 | All seven architecture families, authored combinations, explicit sampled presets. | Workbench/Colab controls, saved UI examples and broad measured diversity. |
| SPEC.N01.01 | Separate standalone network task and architecture records. | Full source-prior reproduction coverage. |
| SPEC.N01.02 | Explicit architecture, weight, bias and sparsity controls. | Broad architecture/TNLU hyperpriors and preference variants. |
| SPEC.N03.01–02 | Before/after and independently chosen layer/expert activations. | Full calibrated/source wrapper combinations. |
| P1.net.01–03,10 | Standalone neural function, individual/block masks, named Beta dropout law. | Complete source lifecycle and acceptance evidence. |
| P1.code.07–12 | Named mask scaling, first-layer exemption, bias law, homogeneous/mixed activation controls. | Complete original noise/activation order and source task sampler. |

P1 input importance, shared/per-node internal-noise statistics, TNLU architecture,
weight/noise hyperpriors and row-stochastic network innovations remain required.
They are not silently equated to fixed structural masks or the separate output
adapter noise. Published and extension options beyond this slice remain required;
these mappings are bounded implementation statements, not closure of all N02/P1.

## Replay, inspection and admission

`sample_network(request, named_rng=..., raw_rng=...)` accepts two distinct injected
streams. One records architecture/hyperparameter choices and the other raw weights,
biases and masks. Both initial/final snapshots, every raw draw and the realized
function are retained in `SampledNetwork`. Complete replay compares every retained
field. Failure publishes neither RNG stream nor a partial ledger.

Construction checks conservative worst-case compute, retained metadata and raw-draw
budgets before numerical allocation. Public evaluation and enclosing causal/native
recipes separately admit their actual row count. These estimates are not measured
peak RSS. Larger caller limits are explicit; a saved request cannot silently raise
the decoder's limits. No new dependency or backend fitting is required.

Generation uses the shared target adapters and preserves row-prefix behavior and
large integer identities. `recipe_from_dict` and the established generation archive
paths retain the construction; generated evidence is ignored. `explain_generation`
shows ordered inputs, expert affine/activation stages, gate probabilities, skip
projections, and final states on original rows. Native, authored causal, parent
member and intervention paths accept the frozen function. Intervention inspection
expands evaluated network nodes, while a fully bypassed node has no evaluated
internal trace. The new standalone recipe is not itself an intervention wrapper;
its explicit `.runtime` supplies the equivalent authored two-node graph when needed.

Example construction:

```python
request = priors.NetworkSamplingRequest(
    input_width=4,
    widths=(8, 5, 1),
    families=("residual", "gated", "irregular"),
    weight_laws=("normal", "uniform", "student_t"),
    experts=3,
)
network = priors.sample_network(request, named_rng=named_rng, raw_rng=raw_rng)
recipe = priors.NetworkRecipe(task, network)
training_task = priors.generate_task(recipe, seed=17, task_id=3)
```

CPU arithmetic, source mask rules, malformed replay, rollback and legacy-byte
comparisons are covered by the scoped verification. The shared explorer now exposes
these architectures; see [network workbench](issue-28-network-workbench.md). Hosted
Colab, GPU agreement, maximum-workload/resource measurements and scientific
comparisons remain pending. No individual slice alone closes #28.
