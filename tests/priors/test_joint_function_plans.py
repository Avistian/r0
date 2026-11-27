"""Source function catalogs retain ordered slots, including duplicated families."""

import pytest

from r0.modules import priors


def test_joint_function_catalog_preserves_preset_expansion_and_duplicate_slots() -> None:
    prior = priors.JointFunctionPrior(functions="tabpfnv2,lin,lin,default")
    expected = (
        "mlp",
        "tree",
        "disc",
        "lin",
        "lin",
        "mlp",
        "tree",
        "disc",
        "lin",
        "quad",
        "gp",
        "em",
        "prod",
    )
    request = prior.request()
    assert request.name == "random_function_fct_type"
    assert len(set(request.choices)) == len(expected)
    assert tuple(prior.family(slot) for slot in request.choices) == expected
    cheap = prior.request(cheap=True)
    assert cheap.name == "cheap_random_function_fct_type"
    assert tuple(prior.family(slot, cheap=True) for slot in cheap.choices) == (
        "gp",
        "tree",
        "disc",
        "lin",
        "quad",
    )
    with pytest.raises(ValueError, match="function"):
        priors.JointFunctionPrior(functions="lin, tree")
    with pytest.raises(ValueError, match="function"):
        priors.JointFunctionPrior(functions="unknown")
