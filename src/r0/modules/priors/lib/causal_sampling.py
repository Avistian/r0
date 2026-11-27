"""Sample fixed structural mechanisms on an explicitly selected graph prior."""

import math
from dataclasses import asdict, replace

import numpy as np

from r0.modules.priors.lib.activation_sampling import (
    ActivationParameterPrior,
    CausalActivationPrior,
)
from r0.modules.priors.lib.causal import ActivatedCausalNode, CausalNode, CausalRecipe
from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.causal_exogenous_sampling import sample_exogenous_profile
from r0.modules.priors.lib.causal_graph import CausalGraphPrior, sample_causal_graph
from r0.modules.priors.lib.causal_importance_sampling import sample_importance_profile
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.causal_matrix_sampling import sample_matrix_profile
from r0.modules.priors.lib.causal_sampled import with_activation_prior
from r0.modules.priors.lib.causal_weight_sampling import sample_factorized_profile
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.matrix_preference_prior import PREFERRED_MATRIX_PROFILES
from r0.modules.priors.lib.preferred_matrix_sampling import sample_preferred_matrices
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.source_causal_prior import SOURCE_ACTIVATION_PROFILES
from r0.modules.priors.lib.source_causal_sampling import with_source_activations

_EXPLORATION_ACTIVATIONS = (
    "identity",
    "tanh",
    "sigmoid",
    "relu",
    "leaky_relu",
    "elu",
    "selu",
    "silu",
    "softplus",
    "relu6",
    "hardtanh",
    "sign",
    "gaussian_bump",
    "sqrt_abs",
    "interval_abs_le_1",
    "sin",
    "square",
    "abs",
    "log_clip_abs",
    "step",
    "modulo",
    "gelu",
    "signed_power",
)


def sample_causal_profile(
    family: str, *, task: LinearRecipe, seed: int, task_id: int, profile: str = "causal-affine-v1"
) -> CausalRecipe:
    """An r0 exploration profile, not a complete published neural/causal prior."""
    if profile not in (
        "causal-affine-v1",
        "causal-activated-v1",
        "causal-preferences-v1",
        "causal-exogenous-v1",
        "causal-factorized-v1",
        "causal-importance-v1",
        "causal-matrices-v1",
        "causal-gaussian-matrices-v1",
        *PREFERRED_MATRIX_PROFILES,
        *SOURCE_ACTIVATION_PROFILES,
    ):
        raise ValueError("unsupported causal sampling profile")
    outputs = task.classes if task.task_type == "multiclass" else 1
    rng = np.random.default_rng(int(record_fingerprint((profile, family, seed, task_id)), 16))
    nodes = task.features + outputs + int(rng.integers(2, 17))
    components = int(rng.integers(1, 4))
    layers: tuple[int, ...] = ()
    if family == "layered":
        size = math.ceil(nodes / components)
        depth = int(rng.integers(2, min(size, 5) + 1))
        cuts = sorted(int(i) for i in rng.choice(np.arange(1, size), depth - 1, replace=False))
        boundaries = [0, *cuts, size]
        layers = tuple(b - a for a, b in zip(boundaries[:-1], boundaries[1:], strict=True))
        nodes = size * components
    prior = CausalGraphPrior(
        family=family,
        nodes=nodes,
        components=components,
        layer_widths=layers,
        edge_probability=0.25
        if family == "redirection"
        else float(rng.choice((0, 0.05, 0.15, 0.35, 0.65, 1))),
        redirection_probability=float(rng.choice((0, 0.25, 0.5, 0.75, 1)))
        if family == "redirection"
        else 0.5,
    )
    widths = (
        tuple(int(rng.choice((1, 2, 4))) for _ in range(nodes))
        if profile
        in (
            "causal-exogenous-v1",
            "causal-factorized-v1",
            "causal-importance-v1",
            "causal-matrices-v1",
            "causal-gaussian-matrices-v1",
            *PREFERRED_MATRIX_PROFILES,
            *SOURCE_ACTIVATION_PROFILES,
        )
        else ()
    )
    recipe = sample_causal_recipe(prior, task=task, seed=seed, task_id=task_id, widths=widths)
    if profile in SOURCE_ACTIVATION_PROFILES:
        source_prior = replace(
            SOURCE_ACTIVATION_PROFILES[profile],
            placement=str(rng.choice(("input", "output", "both"))),
            noise_after_probability=float(rng.choice((0, 0.5, 1))),
        )
        return with_source_activations(recipe, prior=source_prior, seed=seed, task_id=task_id)
    if profile in (
        "causal-preferences-v1",
        "causal-exogenous-v1",
        "causal-factorized-v1",
        "causal-importance-v1",
        "causal-matrices-v1",
        "causal-gaussian-matrices-v1",
        *PREFERRED_MATRIX_PROFILES,
    ):
        activated = with_activation_prior(
            recipe, _preference_profile(rng), seed=seed, task_id=task_id
        )
        if profile in PREFERRED_MATRIX_PROFILES:
            return sample_preferred_matrices(
                activated, rng=rng, prior=PREFERRED_MATRIX_PROFILES[profile]
            )
        if profile in ("causal-matrices-v1", "causal-gaussian-matrices-v1"):
            return sample_matrix_profile(
                activated, rng, gaussian_only=profile == "causal-gaussian-matrices-v1"
            )
        if profile == "causal-importance-v1":
            return sample_importance_profile(activated, rng)
        if profile == "causal-factorized-v1":
            return sample_factorized_profile(activated, rng)
        return (
            sample_exogenous_profile(activated, rng)
            if profile == "causal-exogenous-v1"
            else activated
        )
    if profile == "causal-affine-v1":
        return recipe
    activated_nodes: list[CausalNode] = []
    for node in recipe.nodes:
        input_activation = CausalActivation()
        if node.parents and rng.random() < 0.5:
            input_activation = CausalActivation(str(rng.choice(_EXPLORATION_ACTIVATIONS)))
        activated_nodes.append(
            ActivatedCausalNode(
                **asdict(node),
                input_activation=input_activation,
                output_activation=CausalActivation(str(rng.choice(_EXPLORATION_ACTIVATIONS))),
                noise_position=str(rng.choice(("before_output", "after_output"))),
            )
        )
    return replace(
        recipe,
        nodes=tuple(activated_nodes),
        version="causal-v2",
        construction="gaussian-activated-v1",
    )


def sample_causal_recipe(
    graph: CausalGraphPrior,
    *,
    task: LinearRecipe,
    seed: int,
    task_id: int,
    widths: tuple[int, ...] = (),
    limits: CausalLimits | None = None,
    activations: CausalActivationPrior | None = None,
) -> CausalRecipe:
    """Return a frozen editable recipe; sampling never reads generated task rows."""
    limits = CausalLimits() if limits is None else limits
    topology = sample_causal_graph(graph, seed=seed, task_id=task_id, limits=limits)
    if type(widths) is not tuple or any(type(w) is not int or w < 1 for w in widths):
        raise ValueError("node widths must be a tuple of positive integers")
    widths = widths or (1,) * graph.nodes
    if len(widths) != graph.nodes:
        raise ValueError("one width is required per causal node")
    outputs = task.classes if task.task_type == "multiclass" else 1
    if sum(widths) < task.features + outputs:
        raise ValueError("not enough distinct channels for observed features and targets")
    limits.admit_dimensions(
        widths,
        topology.parents,
        rows=task.rows,
        calibration_rows=0,
        features=task.features,
        outputs=outputs,
    )

    def rng(*keys: object) -> np.random.Generator:
        return np.random.default_rng(
            int(record_fingerprint(("gaussian-affine-v1", seed, task_id, keys)), 16)
        )

    nodes: list[CausalNode] = []
    for i, (width, parents) in enumerate(zip(widths, topology.parents, strict=True)):
        incoming = sum(widths[p] for p in parents)
        weights = rng("weights", i).normal(size=(incoming, width)) / math.sqrt(max(1, incoming))
        nodes.append(
            CausalNode(
                parents=parents,
                weights=tuple(tuple(float(v) for v in row) for row in weights),
                bias=tuple(float(v) for v in rng("bias", i).normal(0, 0.25, size=width)),
                noise_scale=(0.1 if parents else 1.0,) * width,
            )
        )
    channels = tuple((i, j) for i, width in enumerate(widths) for j in range(width))
    chosen = [channels[int(i)] for i in rng("readouts").permutation(len(channels))]
    recipe = CausalRecipe(
        task=task,
        nodes=tuple(nodes),
        observed=tuple(chosen[: task.features]),
        target=tuple(chosen[task.features : task.features + outputs]),
        limits=limits,
        topology=topology,
        construction="gaussian-affine-v1",
    )

    return (
        recipe
        if activations is None
        else with_activation_prior(recipe, activations, seed=seed, task_id=task_id)
    )


def _preference_profile(rng: np.random.Generator) -> CausalActivationPrior:
    """Versioned r0 exploration ranges; not a reproduction of a complete source prior."""
    names = (*_EXPLORATION_ACTIVATIONS, "relu_power", "integer_power", "reciprocal_abs_power")
    mode = str(rng.choice(("independent", "dirichlet", "homogeneous")))
    power = (ActivationParameterPrior("exponent", 0.1, 10, "log_uniform"),)
    ranges = {
        "signed_power": power,
        "relu_power": power,
        "reciprocal_abs_power": power,
        "integer_power": (ActivationParameterPrior("exponent", 2, 6, "randint"),),
        "leaky_relu": (ActivationParameterPrior("negative_slope", 0, 1),),
        "elu": (ActivationParameterPrior("alpha", 0.1, 10, "log_uniform"),),
        "softplus": (ActivationParameterPrior("beta", 0.1, 10, "log_uniform"),),
        "modulo": (ActivationParameterPrior("period", 0.1, 10, "log_uniform"),),
    }
    return CausalActivationPrior(
        choices=tuple(CausalActivation(name) for name in names),
        mode=mode,
        concentration=float(math.exp(rng.uniform(math.log(0.5), math.log(50))))
        if mode == "dirichlet"
        else 1,
        placement=str(rng.choice(("input", "output", "both"))),
        noise_after_probability=float(rng.choice((0, 0.5, 1))),
        parameter_priors=tuple(ranges.get(name, ()) for name in names),
    )
