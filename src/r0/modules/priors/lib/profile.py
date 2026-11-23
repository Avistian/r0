"""Versioned parameter distributions for bounded representatives, not production defaults."""


def diagnostic_profile() -> dict[str, object]:
    return {
        "version": "diagnostic-v1",
        "continuous_inputs": "independent normal(0,1) unless overridden",
        "linear": {
            "relevant_count": "discrete uniform 1..min(4,features)",
            "relevant_indices": "uniform subset without replacement",
            "coefficient_magnitude": "uniform[0.5,1.5)",
            "coefficient_sign": "uniform {-1,+1}",
        },
        "gp": {
            "domain": (
                "eight sorted uniform[-2,2) support coordinates; observations resample uniformly"
            ),
            "mean": 0.0,
            "kernel": "RBF",
            "signal_variance": 1.0,
            "length_scale": 0.6,
            "diagonal_component_variance": 1e-4,
            "innovations": "independent normal(0,1)",
            "feature_count": 1,
        },
        "mlp": {
            "depth": "discrete uniform {1,2}",
            "widths": "discrete uniform 4..8",
            "activation": "uniform {tanh,relu}",
            "weights": "normal(0,1/sqrt(fan_in))",
            "bias": "none",
        },
        "scm": {
            "motif": "uniform {chain,fork,collider}",
            "node_counts": [3, 3, 4],
            "weights": "uniform[-1.5,1.5)",
            "root_noise_std": 1.0,
            "internal_noise_std": 0.1,
            "equation": "tanh(weighted parent sum)+node noise; roots equal noise",
            "observed_nodes": "chain/fork: node1; collider: roots0,1; trim to requested features",
            "nuisance_features": "independent normal(0,1)",
        },
        "tree": {
            "depth": "discrete uniform 1..3",
            "shape": "full binary tree",
            "split_features": "independent uniform feature indices",
            "thresholds": "uniform[-1,1)",
            "leaves": "normal(0,1)",
            "left": "x <= threshold",
        },
        "interaction": {
            "degree": "discrete uniform {2,3}",
            "features": "first degree columns",
            "rule": "cycle product/parity/gated by draw plus task-type index",
            "parity": "parity of positive input signs",
            "gated": "product of remaining inputs if x0>0 else -1",
        },
        "targets": {
            "binary_thresholds": [0.0],
            "multiclass_thresholds": [-0.5, 0.5],
            "threshold_equality": "upper class",
            "mapping": "uniform class permutation per mechanism",
            "regression_added_noise_std": {"none": 0.0, "added": 1.0},
            "label_replacement_probability": {"none": 0.0, "added": 0.2},
            "replacement": "uniform declared class; replacement may leave label unchanged",
        },
        "selection": (
            "none or demonstration drop_added_noise; never resample to fill acceptance quotas"
        ),
    }
