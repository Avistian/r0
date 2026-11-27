"""Construction labels for independently fitted structural graph archives."""


def fitted_card(version: str) -> dict[str, str]:
    if version != "fitted-functional-causal-v1":
        return {}
    return {
        "construction": "independent-fitted-structural-trees-v1",
        "sampling_provenance": "retained-synthetic-fit-and-ancestor-custody-v1",
        "fit_semantics": (
            "independent per-output sklearn fits; internal backend randomness represented "
            "by seeds and version, not a raw draw ledger"
        ),
    }
