"""Authored mechanism edit policy, enforced after decoding inside the worker."""

from r0.modules import priors
from r0.modules.generation.lib.requests import RunRequest


def validate_edit_request(request: RunRequest, policy: object) -> None:
    if (
        policy != "authored-mechanism-v1"
        or request.sampling is not None
        or request.construction is not None
    ):
        raise ValueError("mechanism edit requires the explicit authored recipe policy")
    recipe = request.recipe
    if not isinstance(
        recipe,
        (
            priors.FunctionalCausalRecipe,
            priors.NativeCausalRecipe,
            priors.NativeVectorRecipe,
            priors.NetworkRecipe,
            priors.NetworkInputView,
            priors.ConvertedCausalRecipe,
        ),
    ):
        raise ValueError(
            "mechanism edits require a frozen authored functional graph, native graph or network"
        )
    base = recipe.base if isinstance(recipe, priors.NetworkInputView) else recipe
    if isinstance(base, priors.NetworkRecipe) and not isinstance(
        base.network, priors.FrozenNetworkFunction
    ):
        raise ValueError(
            "network edits require an authored frozen function, not sampled construction"
        )
