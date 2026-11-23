"""Explicitly authored observations, with no sampling claim or random state."""

from r0.modules.episodes import Episode, Feature, Row, Target, TaskType


def authored_episode(
    *,
    source_id: str,
    episode_id: str,
    task_type: TaskType,
    context_x: tuple[float, ...],
    context_y: tuple[Target, ...],
    query_x: tuple[float, ...],
    query_targets: tuple[Target, ...],
    classes: tuple[str, ...] = (),
) -> Episode:
    return Episode(
        source_id,
        episode_id,
        task_type,
        (Feature("x", "continuous"),),
        tuple(Row(f"context-{i}", (x,)) for i, x in enumerate(context_x)),
        context_y,
        tuple(Row(f"query-{i}", (x,)) for i, x in enumerate(query_x)),
        query_targets,
        classes,
        "declared in authored reference specification" if classes else None,
    )
