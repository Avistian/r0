"""Verify retained synthetic draws and fit metadata without rerunning a backend."""

import json
from typing import TYPE_CHECKING, Any, cast

from r0.modules.priors.lib.fitted_backend_contract import fitted_backend_contract
from r0.modules.priors.lib.generator_snapshot import snapshot_generator

if TYPE_CHECKING:
    from r0.modules.priors.lib.fitted_tree_fitting import FittedTreeRealization
    from r0.modules.priors.lib.fitted_tree_plan import FittedTreePlan
    from r0.modules.priors.lib.fitted_tree_records import FrozenFittedTreeFunction
    from r0.modules.priors.lib.weight_geometry import Matrix


def validate_fitted_custody(fit: "FittedTreeRealization") -> None:
    streams = {
        name: snapshot.restore()
        for name, snapshot in zip(("input", "target", "backend"), fit.initial_rng, strict=True)
    }
    expected = ["synthetic-targets", "backend-seeds"]
    if fit.input_law == "iid-standard-normal":
        expected.insert(0, "synthetic-inputs")
    if [draw.role for draw in fit.raw_draws] != expected:
        raise ValueError("fitted raw draw roles differ from synthetic fit law")
    for sequence, draw in enumerate(fit.raw_draws):
        name = {
            "synthetic-inputs": "input",
            "synthetic-targets": "target",
            "backend-seeds": "backend",
        }[draw.role]
        if (
            draw.stream != name
            or draw.sequence != sequence
            or draw.owner != ("fitted-tree", "independent-fit")
        ):
            raise ValueError("fitted raw draw ownership differs")
        draw.replay(streams[name])
        if name == "input":
            retained = tuple(v for row in fit.inputs for v in row)
            shape, distribution, parameters = (fit.plan.rows, fit.plan.input_width), "normal", ()
        elif name == "target":
            retained = tuple(v for row in fit.targets for v in row)
            shape = (fit.plan.rows, fit.plan.output_width)
            distribution = "integers" if fit.plan.task == "classification" else "normal"
            parameters = (("high", fit.plan.classes),) if distribution == "integers" else ()
        else:
            retained = fit.backend_seeds
            shape, distribution, parameters = (
                (fit.plan.output_width,),
                "integers",
                (("high", 2**31),),
            )
        if (draw.values, draw.shape, draw.distribution, draw.parameters) != (
            retained,
            shape,
            distribution,
            parameters,
        ):
            raise ValueError("retained fitted inputs, targets or seeds differ from their raw draws")
    if tuple(snapshot_generator(rng) for rng in streams.values()) != fit.final_rng:
        raise ValueError("fitted final RNG custody differs")
    validate_backend_fit(
        fit.plan, fit.function, fit.backend_seeds, fit.backend_parameters, fit.targets
    )


def validate_backend_fit(
    plan: "FittedTreePlan",
    function: "FrozenFittedTreeFunction",
    backend_seeds: tuple[int, ...],
    backend_parameters: tuple[str, ...],
    targets: "Matrix",
) -> None:
    """Bind exported tables to declared backend options without importing sklearn."""
    for channel, (output, seed, payload) in enumerate(
        zip(function.outputs, backend_seeds, backend_parameters, strict=True)
    ):
        try:
            parameters = json.loads(payload)
        except (ValueError, TypeError) as exc:
            raise ValueError("invalid fitted backend parameters") from exc
        if type(parameters) is not dict or any(
            key not in parameters
            or cast(dict[str, Any], parameters)[key] != value
            or type(cast(dict[str, Any], parameters)[key]) is not type(value)
            for key, value in fitted_backend_contract(plan, seed).items()
        ):
            raise ValueError("fitted backend parameters differ from plan and seeds")
        classifier = plan.task == "classification"
        observed = tuple(sorted({int(row[channel]) for row in targets})) if classifier else ()
        if output.classes != observed:
            raise ValueError("fitted class map differs from fit task")
        boosted = plan.family == "gbdt"
        expected_combiner = (
            (
                "boost_softmax"
                if len(output.classes) > 2
                else "boost_sigmoid"
                if classifier
                else "boost_identity"
            )
            if boosted
            else plan.classification_rule
            if classifier
            else "mean"
        )
        trees = (1 if plan.family == "dt" else plan.estimators) * (
            len(output.classes) if boosted and len(output.classes) > 2 else 1
        )
        if output.combiner != expected_combiner or len(output.trees) != trees:
            raise ValueError("fitted ensemble layout differs from fit plan")
        if boosted and output.learning_rate != plan.learning_rate:
            raise ValueError("fitted learning rate differs from fit plan")
        for tree in output.trees:
            depths = [0] * len(tree.features)
            for index, feature in enumerate(tree.features):
                if feature >= 0:
                    depths[tree.left[index]] = depths[tree.right[index]] = depths[index] + 1
            if max(depths) > plan.depth:
                raise ValueError("fitted tree depth exceeds declared fit plan")
