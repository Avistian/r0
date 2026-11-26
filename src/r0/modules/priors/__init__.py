"""Tiny inspectable source tasks for the episode walkthrough."""

from dataclasses import dataclass

import numpy as np

from r0.modules.episodes import Feature, Row, Target, TaskType
from r0.modules.priors.lib.activation_sampling import (
    ActivationParameterPrior as ActivationParameterPrior,
)
from r0.modules.priors.lib.activation_sampling import CausalActivationPrior as CausalActivationPrior
from r0.modules.priors.lib.authored import authored_episode as authored_episode
from r0.modules.priors.lib.causal import ActivatedCausalNode as ActivatedCausalNode
from r0.modules.priors.lib.causal import CausalNode as CausalNode
from r0.modules.priors.lib.causal import CausalRecipe as CausalRecipe
from r0.modules.priors.lib.causal_activations import CausalActivation as CausalActivation
from r0.modules.priors.lib.causal_graph import CausalGraph as CausalGraph
from r0.modules.priors.lib.causal_graph import CausalGraphPrior as CausalGraphPrior
from r0.modules.priors.lib.causal_graph import sample_causal_graph as sample_causal_graph
from r0.modules.priors.lib.causal_limits import CausalLimits as CausalLimits
from r0.modules.priors.lib.causal_sampled import SampledCausalRecipe as SampledCausalRecipe
from r0.modules.priors.lib.causal_sampling import sample_causal_recipe as sample_causal_recipe
from r0.modules.priors.lib.compositions import Combination as Combination
from r0.modules.priors.lib.compositions import CompositeRecipe as CompositeRecipe
from r0.modules.priors.lib.compositions import FunctionComponent as FunctionComponent
from r0.modules.priors.lib.exact_gp import GaussianProcessRecipe as GaussianProcessRecipe
from r0.modules.priors.lib.fourier import FourierRecipe as FourierRecipe
from r0.modules.priors.lib.fourier import generate_fourier as generate_fourier
from r0.modules.priors.lib.generate import Recipe as Recipe
from r0.modules.priors.lib.generate import generate_task as generate_task
from r0.modules.priors.lib.generate import recipe_from_dict as recipe_from_dict
from r0.modules.priors.lib.generation_graph import explain_generation as explain_generation
from r0.modules.priors.lib.gp_conditioning import condition_gp as condition_gp
from r0.modules.priors.lib.grouped_linear import GroupedLinearRecipe as GroupedLinearRecipe
from r0.modules.priors.lib.inputs import InputDistribution as InputDistribution
from r0.modules.priors.lib.kernels import GibbsKernel as GibbsKernel
from r0.modules.priors.lib.kernels import RationalQuadraticKernel as RationalQuadraticKernel
from r0.modules.priors.lib.kernels import SpectralKernel as SpectralKernel
from r0.modules.priors.lib.kernels import StationaryKernel as StationaryKernel
from r0.modules.priors.lib.linear import LinearRecipe as LinearRecipe
from r0.modules.priors.lib.linear import generate_linear as generate_linear
from r0.modules.priors.lib.pairs import TaskPair as TaskPair
from r0.modules.priors.lib.pairs import compare_pair as compare_pair
from r0.modules.priors.lib.pairs import generate_pair as generate_pair
from r0.modules.priors.lib.pairs import paired_recipe as paired_recipe
from r0.modules.priors.lib.polynomial import PolynomialRecipe as PolynomialRecipe
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
from r0.modules.priors.lib.training_task import TrainingTask as TrainingTask


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
