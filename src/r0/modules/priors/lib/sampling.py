"""Dispatch standalone recipe sampling for public tasks and composition leaves."""

from r0.modules.priors.lib.exact_gp import GaussianProcessRecipe, generate_exact_gp
from r0.modules.priors.lib.fourier import FourierRecipe, generate_fourier
from r0.modules.priors.lib.grouped_linear import GroupedLinearRecipe, generate_grouped_linear
from r0.modules.priors.lib.linear import LinearRecipe, generate_linear
from r0.modules.priors.lib.polynomial import PolynomialRecipe, generate_polynomial
from r0.modules.priors.lib.quadratic import QuadraticRecipe, generate_quadratic
from r0.modules.priors.lib.smooth import SmoothRecipe, generate_smooth
from r0.modules.priors.lib.training_task import TrainingTask

type BaseRecipe = (
    LinearRecipe
    | FourierRecipe
    | PolynomialRecipe
    | QuadraticRecipe
    | GroupedLinearRecipe
    | SmoothRecipe
    | GaussianProcessRecipe
)


def generate_base_task(
    recipe: BaseRecipe, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    """Generate the selected family without substituting another mechanism."""
    if isinstance(recipe, FourierRecipe):
        return generate_fourier(recipe, seed=seed, task_id=task_id, device=device)
    if isinstance(recipe, PolynomialRecipe):
        return generate_polynomial(recipe, seed=seed, task_id=task_id, device=device)
    if isinstance(recipe, QuadraticRecipe):
        return generate_quadratic(recipe, seed=seed, task_id=task_id, device=device)
    if isinstance(recipe, GroupedLinearRecipe):
        return generate_grouped_linear(recipe, seed=seed, task_id=task_id, device=device)
    if isinstance(recipe, SmoothRecipe):
        return generate_smooth(recipe, seed=seed, task_id=task_id, device=device)
    if isinstance(recipe, GaussianProcessRecipe):
        return generate_exact_gp(recipe, seed=seed, task_id=task_id, device=device)
    return generate_linear(recipe, seed=seed, task_id=task_id, device=device)
