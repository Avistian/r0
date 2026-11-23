"""Bounded diagnostic proposals; mechanism, observations and splits replay separately."""

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from r0.modules.episodes import Episode, Feature, Row, Target, TaskType
from r0.modules.priors.lib.smooth_neural import gp_draw, mlp_draw
from r0.modules.priors.lib.structured import scm_draw, tree_draw

Family = Literal["linear", "gp", "mlp", "scm", "tree", "interaction"]
FAMILIES: tuple[Family, ...] = ("linear", "gp", "mlp", "scm", "tree", "interaction")


def encode(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def record_fingerprint(value: object) -> str:
    return hashlib.sha256(encode(value).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class Proposal:
    family: Family
    seed: int = 11
    draw: int = 0
    noise: Literal["none", "added"] = "none"
    task_type: TaskType = "regression"
    rows: int = 64
    features: int = 8
    context_rows: int = 48
    observation: int = 0
    split: int = 0
    version: str = "diagnostic-v1"

    def __post_init__(self) -> None:
        if self.family not in FAMILIES or self.version != "diagnostic-v1":
            raise ValueError("unsupported diagnostic family/version")
        if self.noise not in ("none", "added") or self.task_type not in (
            "regression",
            "binary",
            "multiclass",
        ):
            raise ValueError("unsupported noise regime or task type")
        for value in (self.seed, self.draw, self.observation, self.split):
            if type(value) is not int or not 0 <= value < 2**63:
                raise ValueError("stream keys must be nonnegative integers below 2**63")
        if any(type(v) is not int for v in (self.rows, self.features, self.context_rows)) or not (
            2 <= self.rows <= 128
            and 1 <= self.features <= 16
            and 1 <= self.context_rows < self.rows
        ):
            raise ValueError(
                "diagnostic limit: 2..128 rows, 1..16 features and nonempty partitions"
            )

    @property
    def id(self) -> str:
        return record_fingerprint(asdict(self))

    def stream(self, kind: str) -> tuple[np.random.Generator, str]:
        # Noise is excluded so clean/noisy arms share rules, features and partitions.
        key: tuple[object, ...] = (
            self.version,
            self.family,
            self.seed,
            self.draw,
            self.features,
            self.task_type,
            kind,
        )
        if kind != "mechanism":
            key += (self.observation, self.rows)
        if kind == "split":
            key += (self.split, self.context_rows)
        identity = record_fingerprint(key)
        return np.random.default_rng(int(identity, 16)), identity


@dataclass(frozen=True, slots=True)
class GeneratedTask:
    proposal: Proposal
    episode: Episode
    source_rows: tuple[Row, ...]
    source_targets: tuple[Target, ...]
    mechanism_json: str
    trace_json: str
    streams: tuple[tuple[str, str], ...]

    @property
    def mechanism_id(self) -> str:
        return record_fingerprint(json.loads(self.mechanism_json))


def _linear(p: Proposal, rng: np.random.Generator) -> dict[str, object]:
    relevant = int(rng.integers(1, min(4, p.features) + 1))
    weights = np.zeros(p.features)
    selected = rng.choice(p.features, relevant, replace=False)
    weights[selected] = rng.uniform(0.5, 1.5, relevant) * rng.choice([-1.0, 1.0], relevant)
    return {"weights": weights.tolist(), "relevant": selected.tolist(), "input": "normal(0,1)"}


def sample_proposal(p: Proposal) -> GeneratedTask:
    mechanism_rng, mechanism_stream = p.stream("mechanism")
    observation_rng, observation_stream = p.stream("observations")
    noise_rng, noise_stream = p.stream("noise")
    split_rng, split_stream = p.stream("split")
    construction: dict[str, object] = {}
    x = observation_rng.normal(size=(p.rows, p.features))
    if p.family == "linear":
        parameters = _linear(p, mechanism_rng)
        latent = x @ np.asarray(parameters["weights"], dtype=np.float64)
    elif p.family == "gp":
        x, latent, parameters, construction = gp_draw(
            p.rows, rng=mechanism_rng, observations=observation_rng
        )
    elif p.family == "mlp":
        latent, parameters = mlp_draw(x, rng=mechanism_rng)
    elif p.family == "scm":
        x, latent, parameters, construction = scm_draw(
            x, rng=mechanism_rng, observations=observation_rng
        )
    elif p.family == "tree":
        latent, parameters, construction = tree_draw(x, rng=mechanism_rng)
    elif p.family == "interaction":
        if p.features < 3:
            raise ValueError("interaction representative requires at least 3 features")
        degree = int(mechanism_rng.integers(2, 4))
        rule = ("product", "parity", "gated")[
            (p.draw + ("regression", "binary", "multiclass").index(p.task_type)) % 3
        ]
        parameters = {
            "degree": degree,
            "rule": rule,
            "relevant": list(range(degree)),
            "input": "normal(0,1); parity uses signs",
        }
        if rule == "product":
            latent = np.prod(x[:, :degree], axis=1)
        elif rule == "parity":
            latent = np.asarray(np.sum(x[:, :degree] > 0, axis=1) % 2, dtype=np.float64)
        else:
            latent = np.where(x[:, 0] > 0, np.prod(x[:, 1:degree], axis=1), -1.0)
    else:
        raise ValueError(f"representative not implemented: {p.family}")
    mechanism: dict[str, object] = {
        "family": p.family,
        "version": p.version,
        "parameters": parameters,
        "target_type": p.task_type,
    }
    classes: tuple[str, ...] = ()
    trace: dict[str, object] = {"latent": latent.tolist(), **construction}
    if p.task_type == "regression":
        realized_noise = noise_rng.normal(size=p.rows) * (1.0 if p.noise == "added" else 0.0)
        targets: tuple[Target, ...] = tuple(float(v) for v in latent + realized_noise)
        trace["target_noise"] = realized_noise.tolist()
        mechanism["target_adapter"] = "identity; original units"
    else:
        count = 2 if p.task_type == "binary" else 3
        classes = tuple(f"class-{i}" for i in range(count))
        thresholds = [0.0] if count == 2 else [-0.5, 0.5]
        mapping = [int(v) for v in mechanism_rng.permutation(count)]
        base = np.searchsorted(thresholds, latent, side="right")
        event = noise_rng.uniform(size=p.rows) < (0.2 if p.noise == "added" else 0.0)
        replacement = noise_rng.integers(0, count, size=p.rows)
        mapped = np.asarray(mapping)[base]
        final = np.where(event, replacement, mapped)
        targets = tuple(classes[int(v)] for v in final)
        mechanism["target_adapter"] = {"thresholds": thresholds, "mapping": mapping}
        trace.update(
            {
                "base_class": mapped.tolist(),
                "replacement_event": event.tolist(),
                "replacement_class": replacement.tolist(),
                "label_changed": (final != mapped).tolist(),
            }
        )
    features = tuple(Feature(f"x{i}", "continuous") for i in range(x.shape[1]))
    rows = tuple(Row(f"row-{i}", tuple(float(v) for v in row)) for i, row in enumerate(x))
    indices = [int(i) for i in split_rng.permutation(p.rows)]
    context, queries = indices[: p.context_rows], indices[p.context_rows :]
    source_id = record_fingerprint(
        (tuple(asdict(f) for f in features), tuple(r.values for r in rows), targets)
    )
    episode = Episode(
        source_id,
        record_fingerprint((source_id, indices)),
        p.task_type,
        features,
        tuple(rows[i] for i in context),
        tuple(targets[i] for i in context),
        tuple(rows[i] for i in queries),
        tuple(targets[i] for i in queries),
        classes,
        "declared diagnostic target adapter" if classes else None,
    )
    return GeneratedTask(
        p,
        episode,
        rows,
        targets,
        encode(mechanism),
        encode(trace),
        (
            ("mechanism", mechanism_stream),
            ("observations", observation_stream),
            ("noise", noise_stream),
            ("split", split_stream),
        ),
    )
