"""A bounded number of in-flight tasks, yielded in requested identity order."""

import sys
import time
from collections import deque
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor

from r0.modules.priors import TrainingTask


def parallel_tasks(
    ids: list[int],
    produce: Callable[[int], TrainingTask],
    *,
    workers: int,
    deadline: float,
    observe: Callable[[int, TrainingTask | Exception], TrainingTask | None],
) -> Generator[tuple[int, TrainingTask], None, None]:
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

        def finish(task_id: int, future: Future[TrainingTask]) -> TrainingTask | None:
            try:
                result: TrainingTask | Exception = future.result()
            except Exception as error:
                result = error
            return observe(task_id, result)

        try:
            while waiting:
                task_id, future = waiting.popleft()
                result = finish(task_id, future)
                if result is not None:
                    yield task_id, result
                # The writer handles/persists this result before the next admission.
                admit()
        finally:
            original_error = sys.exception()
            cleanup_error: BaseException | None = None
            for task_id, future in waiting:
                if not future.cancel():
                    try:
                        finish(task_id, future)
                    except BaseException as error:
                        cleanup_error = cleanup_error or error
            if cleanup_error is not None and (
                original_error is None or isinstance(original_error, GeneratorExit)
            ):
                raise cleanup_error
