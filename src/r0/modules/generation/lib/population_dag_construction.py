"""Admit complete original-row population construction before initializing its stream."""

from typing import Any

import numpy as np

from r0.modules import priors as p


def construct_population_dag(
    fields: dict[str, Any],
    *,
    construction_seed: int,
    seed: int,
    task_id: int,
    limits: p.CausalLimits,
) -> p.RecursivePopulationGraph:
    if (
        set(fields) != {"topology", "request", "engine"}
        or fields["topology"] != "recursive-population-dag"
    ):
        raise ValueError(
            "population DAG construction requires complete request and explicit engine"
        )
    engines = {
        name: getattr(np.random, name)
        for name in ("PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937")
    }
    if type(fields["engine"]) is not str or fields["engine"] not in engines:
        raise ValueError("unsupported population construction engine")
    for value in (construction_seed, seed, task_id):
        if type(value) is not int or not 0 <= value < 2**63:
            raise ValueError("population construction identities require integers below 2**63")
    request = p.recursive_population_graph_request_from_dict(fields["request"], limits=limits)
    request.admit(limits=limits)
    rng = np.random.Generator(
        engines[fields["engine"]](np.random.SeedSequence([construction_seed, seed, task_id]))
    )
    return p.realize_recursive_population_graph(request, seed=seed, task_id=task_id, rng=rng)
