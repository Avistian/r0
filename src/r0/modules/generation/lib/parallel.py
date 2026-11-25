"""A bounded number of in-flight tasks, yielded in requested identity order."""

import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor

from r0.modules.priors import TrainingTask


def parallel_tasks(
    ids: list[int], produce: Callable[[int], TrainingTask], *, workers: int, deadline: float
) -> Iterator[tuple[int, TrainingTask | Exception]]:
    iterator = iter(ids)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        waiting: deque[tuple[int, Future[TrainingTask]]] = deque()

        def admit() -> None:
            if time.monotonic() < deadline:
                task_id = next(iterator, None)
                if task_id is not None:
                    waiting.append((task_id, pool.submit(produce, task_id)))

        for _ in range(workers):
            admit()
        try:
            while waiting:
                task_id, future = waiting.popleft()
                try:
                    result: TrainingTask | Exception = future.result()
                except Exception as error:
                    result = error
                yield task_id, result
                # The writer handles/persists this result before the next admission.
                admit()
        finally:
            for _, future in waiting:
                future.cancel()
