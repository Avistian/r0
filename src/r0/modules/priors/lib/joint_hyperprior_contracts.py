"""Known shared categorical-name contracts checked before topology or named draws."""

from r0.modules.priors.lib.joint_function_prior import JointFunctionPrior
from r0.modules.priors.lib.joint_root_plans import JointRootPrior


def validate_hyperprior_catalogs(
    root: JointRootPrior,
    edge: JointFunctionPrior,
    *,
    has_edges: bool,
    mode: str,
    global_policy: str,
) -> None:
    contracts: dict[str, tuple[str, ...]] = {}

    def add(name: str, choices: tuple[str, ...]) -> None:
        if name in contracts and contracts[name] != choices:
            raise ValueError("joint hyperprior categorical choice contract differs for shared name")
        contracts[name] = choices

    def matrix(prior: JointFunctionPrior, *, simple: bool = False) -> None:
        request = prior.matrix.request(simple=simple)
        add(request.name, request.choices)
        families = prior.matrix.simple_families if simple else prior.matrix.families
        if "transformed" in families:
            matrix(prior, simple=True)
        if any(f in ("importance", "factorized", "kernel") for f in families):
            weights(prior)

    def weights(prior: JointFunctionPrior) -> None:
        if (
            mode == "global"
            and global_policy == "scalar-quantile-v1"
            and prior.matrix.vector_family == "polynomial"
            and prior.matrix.scalarization == "source-weight-batches-v1"
        ):
            raise ValueError("joint global quantile policy requires scalarized weight requests")

    def function(prior: JointFunctionPrior, *, cheap: bool = False) -> None:
        request = prior.request(cheap=cheap)
        add(request.name, request.choices)
        for slot in request.choices:
            family = prior.family(slot, cheap=cheap)
            if family == "prod":
                function(prior.child_prior(slot), cheap=True)
            elif family in ("lin", "quad", "disc", "em", "mlp"):
                matrix(prior)
            elif family == "gp":
                weights(prior)

    base = root.request()
    add(base.name, base.choices)
    function(root.function)
    if "covariance" in root.families:
        weights(root.function)
    if has_edges:
        function(edge)
