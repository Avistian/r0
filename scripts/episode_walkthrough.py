"""Inspect issue 21's fixture reports through public interfaces, without pytest."""

import json
from dataclasses import asdict, replace
from pathlib import Path

from r0.modules.episodes import Row
from r0.modules.featurize import transform
from r0.modules.replay import build_episode, parse_config, predict, replay, score


def main() -> None:
    config = parse_config(Path("configs/fixtures/regression.toml").read_text())
    episode = build_episode(config)
    view = transform(episode.predictor_view())
    predictions = predict(view)
    print("Regression context", [r.id for r in episode.context], episode.context_targets)
    print("Regression query terms", score(episode, predictions))
    changed = replace(episode, query_targets=(0.0,) * len(episode.queries))
    print(
        "Changed targets preserve predictor inputs",
        changed.predictor_view() == episode.predictor_view(),
    )
    print(
        "Changed targets preserve predictions",
        predict(transform(changed.predictor_view())) == predictions,
    )
    print("Changed targets score", score(changed, predictions)["value"])
    first = replay(config)
    replay(replace(config, episode_key="another"))
    print("Reordered requests reproduce report", replay(config) == first)
    split = build_episode(replace(config, split_seed=1))
    print("Changed split preserves source", episode.source_task_id == split.source_task_id)
    print("Changed split changes partition", episode.context != split.context)
    print("First raw query", asdict(episode.queries[0]))
    print("First transformed query", asdict(view.queries[0]))
    try:
        parse_config('version=1\nfixture="regression"\nunknown=1')
    except ValueError as error:
        print("Invalid configuration", str(error))
    try:
        replace(
            episode,
            queries=(Row(episode.context[0].id, episode.queries[0].values),) + episode.queries[1:],
        )
    except ValueError as error:
        print("Overlapping identity", str(error))
    for name in ("binary", "multiclass"):
        report = replay(parse_config(Path(f"configs/fixtures/{name}.toml").read_text()))
        print(name, json.dumps(report["scoring"], sort_keys=True))


if __name__ == "__main__":
    main()
