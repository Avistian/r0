"""Tiny inspectable source tasks for the episode walkthrough."""

from dataclasses import dataclass

import numpy as np

from r0.modules.episodes import Feature, Row, Target, TaskType
from r0.modules.priors.lib.activation_sampling import (
    ActivationParameterPrior as ActivationParameterPrior,
)
from r0.modules.priors.lib.activation_sampling import CausalActivationPrior as CausalActivationPrior
from r0.modules.priors.lib.authored import authored_episode as authored_episode
from r0.modules.priors.lib.calibrated_activations import (
    CalibratedActivation as CalibratedActivation,
)
from r0.modules.priors.lib.calibrated_activations import (
    CalibratedStandardization as CalibratedStandardization,
)
from r0.modules.priors.lib.calibrated_activations import (
    calibrate_source_activation as calibrate_source_activation,
)
from r0.modules.priors.lib.calibrated_activations import (
    calibrate_standardization as calibrate_standardization,
)
from r0.modules.priors.lib.calibrated_transforms import CalibratedTransform as CalibratedTransform
from r0.modules.priors.lib.calibrated_transforms import (
    calibrate_kumaraswamy as calibrate_kumaraswamy,
)
from r0.modules.priors.lib.calibrated_transforms import (
    calibrate_l2_normalization as calibrate_l2_normalization,
)
from r0.modules.priors.lib.categorical_preferences import CategoricalDraw as CategoricalDraw
from r0.modules.priors.lib.categorical_preferences import (
    CategoricalParameter as CategoricalParameter,
)
from r0.modules.priors.lib.categorical_preferences import (
    CategoricalPreference as CategoricalPreference,
)
from r0.modules.priors.lib.categorical_preferences import (
    CategoricalPreferences as CategoricalPreferences,
)
from r0.modules.priors.lib.categorical_preferences import (
    categorical_preferences_from_dict as categorical_preferences_from_dict,
)
from r0.modules.priors.lib.categorical_sampling import CategoricalSampler as CategoricalSampler
from r0.modules.priors.lib.causal import ActivatedCausalNode as ActivatedCausalNode
from r0.modules.priors.lib.causal import CausalNode as CausalNode
from r0.modules.priors.lib.causal import CausalRecipe as CausalRecipe
from r0.modules.priors.lib.causal import MatrixCausalNode as MatrixCausalNode
from r0.modules.priors.lib.causal import StochasticCausalNode as StochasticCausalNode
from r0.modules.priors.lib.causal_activations import CausalActivation as CausalActivation
from r0.modules.priors.lib.causal_graph import CausalGraph as CausalGraph
from r0.modules.priors.lib.causal_graph import CausalGraphPrior as CausalGraphPrior
from r0.modules.priors.lib.causal_graph import sample_causal_graph as sample_causal_graph
from r0.modules.priors.lib.causal_limits import CausalLimits as CausalLimits
from r0.modules.priors.lib.causal_nodes import RootInputCausalNode as RootInputCausalNode
from r0.modules.priors.lib.causal_pipeline import CausalPipelineRecipe as CausalPipelineRecipe
from r0.modules.priors.lib.causal_preferred import PreferredCausalRecipe as PreferredCausalRecipe
from r0.modules.priors.lib.causal_sampled import SampledCausalRecipe as SampledCausalRecipe
from r0.modules.priors.lib.causal_sampling import sample_causal_recipe as sample_causal_recipe
from r0.modules.priors.lib.compositions import Combination as Combination
from r0.modules.priors.lib.compositions import CompositeRecipe as CompositeRecipe
from r0.modules.priors.lib.compositions import FunctionComponent as FunctionComponent
from r0.modules.priors.lib.exact_gp import GaussianProcessRecipe as GaussianProcessRecipe
from r0.modules.priors.lib.exogenous import ExogenousDistribution as ExogenousDistribution
from r0.modules.priors.lib.exogenous import NoiseResponse as NoiseResponse
from r0.modules.priors.lib.fourier import FourierRecipe as FourierRecipe
from r0.modules.priors.lib.fourier import generate_fourier as generate_fourier
from r0.modules.priors.lib.frozen_function_composites import (
    FrozenDiscretizationFunction as FrozenDiscretizationFunction,
)
from r0.modules.priors.lib.frozen_function_composites import FrozenEMFunction as FrozenEMFunction
from r0.modules.priors.lib.frozen_function_composites import FrozenMLPFunction as FrozenMLPFunction
from r0.modules.priors.lib.frozen_function_composites import (
    FrozenProductFunction as FrozenProductFunction,
)
from r0.modules.priors.lib.frozen_function_decoding import (
    frozen_function_from_dict as frozen_function_from_dict,
)
from r0.modules.priors.lib.frozen_function_evaluation import FrozenFunction as FrozenFunction
from r0.modules.priors.lib.frozen_function_evaluation import (
    evaluate_frozen_function as evaluate_frozen_function,
)
from r0.modules.priors.lib.frozen_function_primitives import (
    FrozenLinearFunction as FrozenLinearFunction,
)
from r0.modules.priors.lib.frozen_function_primitives import (
    FrozenQuadraticFunction as FrozenQuadraticFunction,
)
from r0.modules.priors.lib.frozen_function_primitives import FrozenRFFFunction as FrozenRFFFunction
from r0.modules.priors.lib.frozen_function_primitives import (
    FrozenTreeFunction as FrozenTreeFunction,
)
from r0.modules.priors.lib.generate import Recipe as Recipe
from r0.modules.priors.lib.generate import generate_task as generate_task
from r0.modules.priors.lib.generate import recipe_from_dict as recipe_from_dict
from r0.modules.priors.lib.generation_graph import explain_generation as explain_generation
from r0.modules.priors.lib.gp_conditioning import condition_gp as condition_gp
from r0.modules.priors.lib.grouped_linear import GroupedLinearRecipe as GroupedLinearRecipe
from r0.modules.priors.lib.inputs import InputDistribution as InputDistribution
from r0.modules.priors.lib.joint_activation_plans import JointActivationPlan as JointActivationPlan
from r0.modules.priors.lib.joint_activation_plans import (
    joint_activation_plan_from_dict as joint_activation_plan_from_dict,
)
from r0.modules.priors.lib.joint_activation_plans import (
    sample_joint_activation as sample_joint_activation,
)
from r0.modules.priors.lib.joint_converter_plans import (
    JointConverterPlan as JointConverterPlan,
)
from r0.modules.priors.lib.joint_converter_plans import (
    JointNumericalConverterPrior as JointNumericalConverterPrior,
)
from r0.modules.priors.lib.joint_converter_plans import (
    joint_converter_plan_from_dict as joint_converter_plan_from_dict,
)
from r0.modules.priors.lib.joint_converter_plans import (
    sample_joint_converter as sample_joint_converter,
)
from r0.modules.priors.lib.joint_discretization_realization import (
    JointDiscretizationFunctionRealization as JointDiscretizationFunctionRealization,
)
from r0.modules.priors.lib.joint_function_calibration import (
    FunctionCalibrationSample as FunctionCalibrationSample,
)
from r0.modules.priors.lib.joint_function_decoding import (
    joint_function_plan_from_dict as joint_function_plan_from_dict,
)
from r0.modules.priors.lib.joint_function_plans import JointFunctionPlan as JointFunctionPlan
from r0.modules.priors.lib.joint_function_prior import JointFunctionPrior as JointFunctionPrior
from r0.modules.priors.lib.joint_function_realization import (
    JointMatrixFunctionRealization as JointMatrixFunctionRealization,
)
from r0.modules.priors.lib.joint_function_realization import (
    materialize_joint_function as materialize_joint_function,
)
from r0.modules.priors.lib.joint_function_realization_decoding import (
    joint_function_realization_from_dict as joint_function_realization_from_dict,
)
from r0.modules.priors.lib.joint_function_sampling import (
    sample_joint_function as sample_joint_function,
)
from r0.modules.priors.lib.joint_matrix_decoding import (
    joint_matrix_realization_from_dict as joint_matrix_realization_from_dict,
)
from r0.modules.priors.lib.joint_matrix_plans import JointMatrixPlan as JointMatrixPlan
from r0.modules.priors.lib.joint_matrix_plans import JointMatrixPrior as JointMatrixPrior
from r0.modules.priors.lib.joint_matrix_plans import (
    joint_matrix_plan_from_dict as joint_matrix_plan_from_dict,
)
from r0.modules.priors.lib.joint_matrix_plans import sample_joint_matrix as sample_joint_matrix
from r0.modules.priors.lib.joint_matrix_realization import (
    JointMatrixRealization as JointMatrixRealization,
)
from r0.modules.priors.lib.joint_matrix_realization import (
    materialize_joint_matrix as materialize_joint_matrix,
)
from r0.modules.priors.lib.joint_plan_calibration import (
    fit_joint_activation as fit_joint_activation,
)
from r0.modules.priors.lib.joint_plan_calibration import fit_joint_converter as fit_joint_converter
from r0.modules.priors.lib.joint_point_plans import JointPointPlan as JointPointPlan
from r0.modules.priors.lib.joint_point_plans import (
    joint_point_plan_from_dict as joint_point_plan_from_dict,
)
from r0.modules.priors.lib.joint_point_plans import sample_joint_points as sample_joint_points
from r0.modules.priors.lib.joint_point_realization import (
    materialize_joint_points as materialize_joint_points,
)
from r0.modules.priors.lib.joint_rff_realization import (
    JointRFFFunctionRealization as JointRFFFunctionRealization,
)
from r0.modules.priors.lib.joint_root_decoding import (
    joint_root_plan_from_dict as joint_root_plan_from_dict,
)
from r0.modules.priors.lib.joint_root_plans import JointRootPlan as JointRootPlan
from r0.modules.priors.lib.joint_root_plans import JointRootPrior as JointRootPrior
from r0.modules.priors.lib.joint_root_plans import sample_joint_root as sample_joint_root
from r0.modules.priors.lib.joint_sampling import JointSamplingContext as JointSamplingContext
from r0.modules.priors.lib.joint_sampling_records import JointDecision as JointDecision
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord as JointSamplingRecord
from r0.modules.priors.lib.joint_sampling_records import (
    joint_sampling_from_dict as joint_sampling_from_dict,
)
from r0.modules.priors.lib.joint_tree_realization import (
    JointTreeFunctionRealization as JointTreeFunctionRealization,
)
from r0.modules.priors.lib.joint_weight_plans import (
    JointWeightPlan as JointWeightPlan,
)
from r0.modules.priors.lib.joint_weight_plans import (
    joint_weight_plan_from_dict as joint_weight_plan_from_dict,
)
from r0.modules.priors.lib.joint_weight_plans import (
    materialize_joint_weights as materialize_joint_weights,
)
from r0.modules.priors.lib.joint_weight_plans import (
    sample_joint_weights as sample_joint_weights,
)
from r0.modules.priors.lib.kernel_points import KernelPointCloud as KernelPointCloud
from r0.modules.priors.lib.kernels import GibbsKernel as GibbsKernel
from r0.modules.priors.lib.kernels import RationalQuadraticKernel as RationalQuadraticKernel
from r0.modules.priors.lib.kernels import SpectralKernel as SpectralKernel
from r0.modules.priors.lib.kernels import StationaryKernel as StationaryKernel
from r0.modules.priors.lib.linear import LinearRecipe as LinearRecipe
from r0.modules.priors.lib.linear import generate_linear as generate_linear
from r0.modules.priors.lib.matrix_preference_prior import CausalMatrixPrior as CausalMatrixPrior
from r0.modules.priors.lib.node_pipeline import CausalNodePipeline as CausalNodePipeline
from r0.modules.priors.lib.node_pipeline import NumericalConverter as NumericalConverter
from r0.modules.priors.lib.pairs import TaskPair as TaskPair
from r0.modules.priors.lib.pairs import compare_pair as compare_pair
from r0.modules.priors.lib.pairs import generate_pair as generate_pair
from r0.modules.priors.lib.pairs import paired_recipe as paired_recipe
from r0.modules.priors.lib.parameter_preferences import NumericalDraw as NumericalDraw
from r0.modules.priors.lib.parameter_preferences import NumericalParameter as NumericalParameter
from r0.modules.priors.lib.parameter_preferences import NumericalPreference as NumericalPreference
from r0.modules.priors.lib.parameter_preferences import ParameterPreferences as ParameterPreferences
from r0.modules.priors.lib.parameter_preferences import ParameterSampler as ParameterSampler
from r0.modules.priors.lib.parameter_preferences import (
    parameter_preferences_from_dict as parameter_preferences_from_dict,
)
from r0.modules.priors.lib.polynomial import PolynomialRecipe as PolynomialRecipe
from r0.modules.priors.lib.preference_binding import PreferenceBinding as PreferenceBinding
from r0.modules.priors.lib.preference_limits import PreferenceLimits as PreferenceLimits
from r0.modules.priors.lib.preferred_matrix_sampling import (
    sample_preferred_matrices as sample_preferred_matrices,
)
from r0.modules.priors.lib.profile import diagnostic_profile as diagnostic_profile
from r0.modules.priors.lib.quadratic import QuadraticRecipe as QuadraticRecipe
from r0.modules.priors.lib.recipe_sampling import sample_recipe as sample_recipe
from r0.modules.priors.lib.representatives import FAMILIES as FAMILIES
from r0.modules.priors.lib.representatives import GeneratedTask as GeneratedTask
from r0.modules.priors.lib.representatives import Proposal as Proposal
from r0.modules.priors.lib.representatives import record_fingerprint as record_fingerprint
from r0.modules.priors.lib.representatives import sample_proposal as sample_proposal
from r0.modules.priors.lib.signals import evaluate_signal as evaluate_signal
from r0.modules.priors.lib.smooth import SmoothRecipe as SmoothRecipe
from r0.modules.priors.lib.source_activation_law import (
    SourceActivationSelection as SourceActivationSelection,
)
from r0.modules.priors.lib.source_activation_sampling import (
    SourceActivationSampler as SourceActivationSampler,
)
from r0.modules.priors.lib.source_activation_sampling import (
    SourceActivationSelections as SourceActivationSelections,
)
from r0.modules.priors.lib.source_activation_sampling import (
    source_activation_selections_from_dict as source_activation_selections_from_dict,
)
from r0.modules.priors.lib.source_causal import SourceCausalRecipe as SourceCausalRecipe
from r0.modules.priors.lib.source_causal_prior import SourceActivationPrior as SourceActivationPrior
from r0.modules.priors.lib.source_causal_prior import SourceStageBinding as SourceStageBinding
from r0.modules.priors.lib.source_causal_sampling import (
    with_source_activations as with_source_activations,
)
from r0.modules.priors.lib.source_matrix import SourceMatrixCausalRecipe as SourceMatrixCausalRecipe
from r0.modules.priors.lib.source_matrix_sampling import (
    with_source_matrices as with_source_matrices,
)
from r0.modules.priors.lib.training_task import TrainingTask as TrainingTask
from r0.modules.priors.lib.transformed_weights import GaussianWeights as GaussianWeights
from r0.modules.priors.lib.transformed_weights import NormalizedWeights as NormalizedWeights
from r0.modules.priors.lib.transformed_weights import TransformedWeights as TransformedWeights
from r0.modules.priors.lib.weight_matrices import FactorizedWeights as FactorizedWeights
from r0.modules.priors.lib.weight_matrices import (
    ImportanceFactorizedWeights as ImportanceFactorizedWeights,
)
from r0.modules.priors.lib.weight_matrices import ImportanceWeights as ImportanceWeights
from r0.modules.priors.lib.weight_matrices import LaplaceKernelWeights as LaplaceKernelWeights
from r0.modules.priors.lib.weight_vectors import PolynomialWeightVector as PolynomialWeightVector
from r0.modules.priors.lib.weight_vectors import UniformWeightVector as UniformWeightVector


@dataclass(frozen=True)
class SourceTask:
    features: tuple[Feature, ...]
    rows: tuple[Row, ...]
    targets: tuple[Target, ...]
    classes: tuple[str, ...]
    class_provenance: str | None
    mechanism: str


def sample_fixture(task_type: TaskType, *, rng: np.random.Generator) -> SourceTask:
    """Draw a small offset; all other values are deliberately readable."""
    if task_type not in ("regression", "binary", "multiclass"):
        raise ValueError("unsupported fixture task type")
    offset = int(rng.integers(0, 4))
    features = (
        Feature("measurement", "continuous"),
        Feature("count", "count"),
        Feature("flag", "binary"),
        Feature("color", "categorical"),
        Feature("grade", "ordinal", order=("low", "medium", "high")),
        Feature("time", "timestamp", unit="s", timezone="UTC"),
        Feature("constant", "continuous"),
        Feature("unobserved", "continuous"),
    )
    colors = ("red", "blue", "red", "blue", None, "green")
    grades = ("low", "medium", "high", "low", "medium", "high")
    rows = tuple(
        Row(
            f"row-{i}",
            (
                float(offset + 2 * i),
                i,
                i % 2,
                colors[i],
                grades[i],
                1_700_000_000 + 60 * i,
                7.0,
                None,
            ),
        )
        for i in range(6)
    )
    if task_type == "regression":
        targets: tuple[Target, ...] = tuple(float(2 * i + 1) for i in range(6))
        classes: tuple[str, ...] = ()
    else:
        classes = ("A", "B") if task_type == "binary" else ("A", "B", "C")
        targets = ("A", "B", "A", "B", "A", classes[-1])
    return SourceTask(
        features,
        rows,
        targets,
        classes,
        "declared by fixture-v1 task design" if classes else None,
        f"fixture-v1: measurement offset {offset}; regression y=2*row_index+1; "
        "classification labels assigned by declared fixture design",
    )
