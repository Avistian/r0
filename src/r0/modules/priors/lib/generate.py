"""One task entrypoint and versioned decoder for all implemented training families."""

from typing import Any, cast

from r0.modules.priors.lib.activation_sampling import activation_prior_from_dict
from r0.modules.priors.lib.categorical_preferences import categorical_preferences_from_dict
from r0.modules.priors.lib.causal import CausalRecipe, causal_node_from_dict, generate_causal
from r0.modules.priors.lib.causal_graph import CausalGraph, CausalGraphPrior
from r0.modules.priors.lib.causal_intervention_records import CausalInterventionRecipe
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.causal_pipeline import CausalPipelineRecipe
from r0.modules.priors.lib.causal_preferred import PreferredCausalRecipe
from r0.modules.priors.lib.causal_sampled import SampledCausalRecipe
from r0.modules.priors.lib.compositions import (
    Combination,
    CompositeRecipe,
    FunctionComponent,
    generate_composite,
)
from r0.modules.priors.lib.exact_gp import GaussianProcessRecipe
from r0.modules.priors.lib.fourier import FourierRecipe
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_decoding import functional_recipe_from_dict
from r0.modules.priors.lib.functional_causal_evaluation import generate_functional_causal
from r0.modules.priors.lib.geometry_graph_decoding import geometry_graph_from_dict
from r0.modules.priors.lib.grouped_linear import GroupedLinearRecipe
from r0.modules.priors.lib.inputs import InputDistribution
from r0.modules.priors.lib.kernels import (
    GibbsKernel,
    Kernel,
    RationalQuadraticKernel,
    SpectralKernel,
    StationaryKernel,
)
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.matrix_preference_prior import CausalMatrixPrior
from r0.modules.priors.lib.native_causal import NativeCausalRecipe
from r0.modules.priors.lib.native_decoding import native_recipe_from_dict
from r0.modules.priors.lib.native_evaluation import generate_native_causal
from r0.modules.priors.lib.native_graph_decoding import native_graph_from_dict
from r0.modules.priors.lib.node_pipeline import pipeline_from_dict
from r0.modules.priors.lib.parameter_preferences import parameter_preferences_from_dict
from r0.modules.priors.lib.polynomial import PolynomialRecipe
from r0.modules.priors.lib.preference_binding import PreferenceBinding
from r0.modules.priors.lib.quadratic import QuadraticRecipe
from r0.modules.priors.lib.sampled_functional_graph import SampledFunctionalGraph
from r0.modules.priors.lib.sampled_geometry_graph import SampledGeometryGraph
from r0.modules.priors.lib.sampled_graph_decoding import sampled_graph_from_dict
from r0.modules.priors.lib.sampled_native_graph import SampledNativeGraph
from r0.modules.priors.lib.sampled_native_output import generate_sampled_native
from r0.modules.priors.lib.sampled_parent_graph import SampledParentGraph
from r0.modules.priors.lib.sampled_topology_graph import SampledTopologyGraph
from r0.modules.priors.lib.sampling import BaseRecipe, generate_base_task
from r0.modules.priors.lib.smooth import SmoothRecipe
from r0.modules.priors.lib.source_activation_sampling import source_activation_selections_from_dict
from r0.modules.priors.lib.source_causal import SourceCausalRecipe
from r0.modules.priors.lib.source_causal_prior import SourceActivationPrior, SourceStageBinding
from r0.modules.priors.lib.source_matrix import SourceMatrixCausalRecipe
from r0.modules.priors.lib.topology_graph_decoding import topology_graph_from_dict
from r0.modules.priors.lib.training_task import TrainingTask

type Recipe = (
    CausalInterventionRecipe
    | BaseRecipe
    | CompositeRecipe
    | CausalRecipe
    | FunctionalCausalRecipe
    | SampledFunctionalGraph
    | SampledParentGraph
    | SampledTopologyGraph
    | SampledGeometryGraph
    | NativeCausalRecipe
    | SampledNativeGraph
)


def _kernel_from_dict(data: dict[str, Any]) -> Kernel:
    fields = dict(data)
    for key in (
        "axes",
        "feature_scales",
        "frequencies",
        "centers",
        "transition_scales",
        "length_slopes",
        "amplitude_slopes",
    ):
        if key in fields:
            fields[key] = tuple(fields[key])
    if fields.get("kind") == "gibbs":
        return GibbsKernel(**fields)
    if fields.get("kind") == "rational_quadratic":
        return RationalQuadraticKernel(**fields)
    if fields.get("kind") == "spectral":
        return SpectralKernel(**fields)
    return StationaryKernel(**fields)


def _expression_from_dict(value: Any) -> int | Combination:
    if type(value) is int:
        return value
    fields = dict(value)
    fields["children"] = tuple(_expression_from_dict(child) for child in fields["children"])
    return Combination(**fields)


def recipe_from_dict(data: object, *, limits: CausalLimits | None = None) -> Recipe:
    """Decode JSON/TOML fields; caller limits bound frozen functional graph decoding."""
    limits = CausalLimits() if limits is None else limits
    if type(limits) is not CausalLimits:
        raise ValueError("recipe decoding requires caller-owned causal limits")
    if not isinstance(data, dict):
        raise ValueError("recipe must be an object")
    fields = dict(cast(dict[str, Any], data))
    version = fields.get("version", "linear-v1")
    if version == "causal-intervention-v1":
        from r0.modules.priors.lib.causal_intervention_decoding import intervention_from_dict

        return intervention_from_dict(fields, limits=limits)
    if version == "sampled-native-graph-v1":
        return native_graph_from_dict(fields, limits=limits)
    if version == "native-causal-v1":
        return native_recipe_from_dict(fields, limits=limits)
    if version == "sampled-geometry-graph-v1":
        return geometry_graph_from_dict(fields, limits=limits)
    if version == "sampled-topology-graph-v1":
        return topology_graph_from_dict(fields, limits=limits)
    if version == "sampled-parent-graph-v1":
        from r0.modules.priors.lib.parent_graph_decoding import parent_graph_from_dict

        return parent_graph_from_dict(fields, limits=limits)
    if version == "sampled-functional-graph-v1":
        return sampled_graph_from_dict(fields, limits=limits)
    if version in ("functional-causal-v1", "functional-causal-v2", "fitted-functional-causal-v1"):
        return functional_recipe_from_dict(fields, limits=limits)
    if version in ("causal-v1", "causal-v2"):
        task = recipe_from_dict(fields.get("task", {}))
        if not isinstance(task, LinearRecipe):
            raise ValueError("task controls must use linear-v1 settings")
        fields["task"] = task
        fields["limits"] = CausalLimits(**fields.get("limits", {}))
        if fields.get("topology") is not None:
            topology = dict(fields["topology"])
            prior = dict(topology["prior"])
            prior["layer_widths"] = tuple(prior.get("layer_widths", ()))
            topology["prior"] = CausalGraphPrior(**prior)
            topology["parents"] = tuple(tuple(row) for row in topology["parents"])
            fields["topology"] = CausalGraph(**topology)
        fields["nodes"] = tuple(causal_node_from_dict(item) for item in fields["nodes"])
        for key in ("observed", "target"):
            fields[key] = tuple(tuple(ref) for ref in fields[key])
        if "matrix_prior" in fields:
            matrix_prior = dict(fields["matrix_prior"])
            for key in ("families", "simple_families", "point_laws", "covariance_bases"):
                if key in matrix_prior:
                    matrix_prior[key] = tuple(matrix_prior[key])
            fields["matrix_prior"] = CausalMatrixPrior(**matrix_prior)
            fields["numerical_preferences"] = parameter_preferences_from_dict(
                fields["numerical_preferences"]
            )
            fields["categorical_preferences"] = categorical_preferences_from_dict(
                fields["categorical_preferences"]
            )
            bindings: list[PreferenceBinding] = []
            for raw in fields["preference_bindings"]:
                binding = dict(raw)
                binding["path"] = tuple(binding["path"])
                bindings.append(PreferenceBinding(**binding))
            fields["preference_bindings"] = tuple(bindings)
        if "source_prior" in fields:
            fields["source_prior"] = SourceActivationPrior(**fields["source_prior"])
            fields["source_selections"] = source_activation_selections_from_dict(
                fields["source_selections"]
            )
            fields["rescale_preferences"] = parameter_preferences_from_dict(
                fields["rescale_preferences"]
            )
            fields["source_bindings"] = tuple(
                SourceStageBinding(**b) for b in fields["source_bindings"]
            )
            fields["noise_units"] = tuple(fields["noise_units"])
            if "matrix_prior" in fields:
                fields["matrix_activation_nodes"] = tuple(fields["matrix_activation_nodes"])
                return SourceMatrixCausalRecipe(**fields)
            return SourceCausalRecipe(**fields)
        if "activation_prior" in fields:
            fields["activation_prior"] = activation_prior_from_dict(fields["activation_prior"])
            fields["activation_probabilities"] = tuple(fields["activation_probabilities"])
            if "matrix_prior" in fields:
                return PreferredCausalRecipe(**fields)
            return SampledCausalRecipe(**fields)
        if "pipelines" in fields:
            fields["pipelines"] = tuple(pipeline_from_dict(p) for p in fields["pipelines"])
            return CausalPipelineRecipe(**fields)
        return CausalRecipe(**fields)
    if version == "composite-v1":
        task = recipe_from_dict(fields.get("task", {}))
        if not isinstance(task, LinearRecipe):
            raise ValueError("task controls must use linear-v1 settings")
        fields["task"] = task
        fields["inputs"] = InputDistribution(**fields.get("inputs", {}))
        components: list[FunctionComponent] = []
        for item in fields["components"]:
            component = dict(item)
            child = recipe_from_dict(component.pop("recipe"))
            if isinstance(
                child,
                (
                    CompositeRecipe,
                    CausalRecipe,
                    FunctionalCausalRecipe,
                    SampledParentGraph,
                    SampledFunctionalGraph,
                    SampledTopologyGraph,
                    SampledGeometryGraph,
                    NativeCausalRecipe,
                    SampledNativeGraph,
                ),
            ):
                raise ValueError("nest combinations in expression, not in component recipes")
            component["recipe"] = child
            if "axes" in component:
                component["axes"] = tuple(component["axes"])
            components.append(FunctionComponent(**component))
        fields["components"] = tuple(components)
        fields["expression"] = _expression_from_dict(fields.get("expression", 0))
        return CompositeRecipe(**fields)
    if version == "linear-v1":
        if "class_offsets" in fields:
            fields["class_offsets"] = tuple(fields["class_offsets"])
        return LinearRecipe(**fields)
    if version in (
        "fourier-v1",
        "polynomial-v1",
        "quadratic-v1",
        "grouped-linear-v1",
        "smooth-v1",
        "exact-gp-v1",
    ):
        task = recipe_from_dict(fields.get("task", {}))
        if not isinstance(task, LinearRecipe):
            raise ValueError("task controls must use linear-v1 settings")
        fields["task"] = task
        if version in (
            "polynomial-v1",
            "quadratic-v1",
            "grouped-linear-v1",
            "smooth-v1",
            "exact-gp-v1",
        ):
            fields["inputs"] = InputDistribution(**fields.get("inputs", {}))
            if version == "exact-gp-v1":
                if "kernels" in fields:
                    fields["kernels"] = tuple(_kernel_from_dict(k) for k in fields["kernels"])
                return GaussianProcessRecipe(**fields)
            if version == "smooth-v1":
                return SmoothRecipe(**fields)
            if version == "grouped-linear-v1":
                return GroupedLinearRecipe(**fields)
            return (
                PolynomialRecipe(**fields)
                if version == "polynomial-v1"
                else QuadraticRecipe(**fields)
            )
        return FourierRecipe(**fields)
    raise ValueError(f"unsupported recipe version: {version!r}")


def generate_task(recipe: Recipe, *, seed: int, task_id: int, device: str = "cpu") -> TrainingTask:
    """Generate the selected family without substituting another mechanism."""
    if isinstance(recipe, CausalInterventionRecipe):
        from r0.modules.priors.lib.causal_intervention_generation import generate_intervened

        return generate_intervened(recipe, seed=seed, task_id=task_id, device=device)
    if isinstance(recipe, SampledNativeGraph):
        return generate_sampled_native(recipe, seed=seed, task_id=task_id, device=device)
    if isinstance(recipe, NativeCausalRecipe):
        return generate_native_causal(recipe, seed=seed, task_id=task_id, device=device)
    if isinstance(
        recipe,
        (
            FunctionalCausalRecipe,
            SampledParentGraph,
            SampledFunctionalGraph,
            SampledTopologyGraph,
            SampledGeometryGraph,
        ),
    ):
        return generate_functional_causal(recipe, seed=seed, task_id=task_id, device=device)
    if isinstance(recipe, CausalRecipe):
        return generate_causal(recipe, seed=seed, task_id=task_id, device=device)
    if isinstance(recipe, CompositeRecipe):
        return generate_composite(recipe, seed=seed, task_id=task_id, device=device)
    return generate_base_task(recipe, seed=seed, task_id=task_id, device=device)
