"""Sample fixed structural mechanisms on an explicitly selected graph prior."""

from r0.modules.priors.lib.causal import CausalRecipe
from r0.modules.priors.lib.causal_graph import CausalGraphPrior
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.linear import LinearRecipe


def sample_causal_recipe(
    graph: CausalGraphPrior,
    *,
    task: LinearRecipe,
    seed: int,
    task_id: int,
    widths: tuple[int, ...] = (),
    limits: CausalLimits | None = None,
) -> CausalRecipe:
    """Return a frozen editable recipe; sampling never reads generated task rows."""
    raise NotImplementedError("causal mechanism sampling is not implemented")
