"""Dataset-first summaries; classification and target units never get pooled."""

import math
from collections import defaultdict

from .storage import Document


def summarize_draws(scores: list[float]) -> Document:
    if len(scores) != 3 or not all(math.isfinite(x) for x in scores):
        raise ValueError("exactly three finite draw scores required")
    return {"draws": scores, "mean": sum(scores) / 3, "min": min(scores), "max": max(scores)}


def comparison_gaps(
    task: str, learner: list[float], tabpfn: list[float], simple: list[float]
) -> Document:
    for values in (learner, tabpfn, simple):
        summarize_draws(values)
    raw = [a - b for a, b in zip(learner, tabpfn, strict=True)]
    quality = sum(learner) < sum(simple)
    if task == "regression" and 0 in simple:
        return {
            "status": "INCOMPLETE",
            "raw_gaps": summarize_draws(raw),
            "gap": None,
            "quality_target_met": quality,
            "reason": "zero context-mean RMSE denominator",
        }
    gaps = [r / s for r, s in zip(raw, simple, strict=True)] if task == "regression" else raw
    return {
        "status": "COMPLETE",
        "raw_gaps": summarize_draws(raw),
        "gap": summarize_draws(gaps),
        "quality_target_met": quality,
    }


def aggregate_datasets(datasets: list[Document]) -> Document:
    """Aggregate descriptive gaps equally by dataset within task and partition.

    Inputs retain each dataset's own metrics. Regression raw RMSE is never averaged
    across datasets. Missing gaps make the affected group incomplete.
    """
    groups: dict[tuple[str, str], list[Document]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for dataset in datasets:
        key = (dataset["partition"], dataset["task"], dataset["dataset"])
        if key in seen:
            raise ValueError("duplicate dataset summary")
        seen.add(key)
        groups[key[:2]].append(dataset)
    output: list[Document] = []
    for (partition, task), rows in sorted(groups.items()):
        complete = all(r["comparison"]["status"] == "COMPLETE" for r in rows)
        output.append(
            {
                "partition": partition,
                "task": task,
                "datasets": rows,
                "status": "COMPLETE" if complete else "INCOMPLETE",
                "mean_gap": sum(r["comparison"]["gap"]["mean"] for r in rows) / len(rows)
                if complete
                else None,
            }
        )
    return {"version": 1, "groups": output}
