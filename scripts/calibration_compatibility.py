"""Print exact task digests to compare successful records across source revisions."""

import hashlib
import json
from dataclasses import asdict

from r0.modules.episodes import TaskType
from r0.modules.priors import (
    GroupedLinearRecipe,
    LinearRecipe,
    PolynomialRecipe,
    QuadraticRecipe,
    generate_task,
)


def main() -> None:
    records = []
    kinds: tuple[tuple[TaskType, int], ...] = (("regression", 0), ("binary", 2), ("multiclass", 3))
    for kind, classes in kinds:
        controls = LinearRecipe(task_type=kind, classes=classes)
        for recipe in (
            PolynomialRecipe(task=controls),
            QuadraticRecipe(task=controls),
            GroupedLinearRecipe(task=controls),
        ):
            task = generate_task(recipe, seed=27, task_id=0)
            digest = hashlib.sha256()
            digest.update(json.dumps((task.source_task_id, task.episode_id, task.classes)).encode())
            digest.update(task.recipe_json.encode())
            digest.update(task.mechanism_json.encode())
            for tensor in (
                task.inputs,
                task.targets,
                task.scores,
                task.probabilities,
                task.context_indices,
                task.query_indices,
            ):
                if tensor is not None:
                    digest.update(str((tensor.shape, tensor.dtype)).encode())
                    digest.update(tensor.numpy().tobytes())
                else:
                    digest.update(b"none")
            records.append({"recipe": asdict(recipe), "sha256": digest.hexdigest()})
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
