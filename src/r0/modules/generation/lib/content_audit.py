"""Disk-backed exact observed-task signatures; no mechanism-equivalence claim."""

import fcntl
import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from r0.modules.generation.lib.coverage import CoveragePlan, coverage_shard
from r0.modules.generation.lib.coverage_report import inspect_coverage_shard
from r0.modules.generation.lib.loading import load_group
from r0.modules.priors import TrainingTask

_VERSION = "observed-task-signature-v1"


def task_content_signature(task: TrainingTask) -> str:
    """Hash observed X/y and split/schema, excluding IDs and privileged mechanisms.

    Includes query labels for offline auditing only. Row/column order and dtypes
    matter. Floating zero signs and NaN payloads are normalized, not rounded.
    """
    digest = hashlib.sha256()

    def add(value: bytes) -> None:
        digest.update(len(value).to_bytes(8, "little"))
        digest.update(value)

    metadata = {
        "version": _VERSION,
        "task_type": task.task_type,
        "classes": task.classes,
        "feature_schema": [asdict(feature) for feature in task.feature_schema],
    }
    add(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    tensors: tuple[torch.Tensor | None, ...] = (
        task.inputs,
        task.targets,
        task.context_indices,
        task.query_indices,
        task.missing_mask,
    )
    for tensor in tensors:
        if tensor is None:
            add(b"null")
            continue
        array = tensor.detach().cpu().numpy().copy()
        if np.issubdtype(array.dtype, np.floating):
            array[array == 0] = 0
            array[np.isnan(array)] = np.nan
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
        add(json.dumps({"dtype": array.dtype.str, "shape": array.shape}, sort_keys=True).encode())
        add(array.tobytes(order="C"))
    return digest.hexdigest()


def index_coverage_shard(
    plan: CoveragePlan, index: int, out: Path, database: Path, *, summarize: bool = True
) -> dict[str, Any]:
    """Index one stopped shard transactionally into a plan-bound SQLite database.

    Existing IDs must retain their signatures. RAM is bounded by one task plus
    the existing shard inspection sets and SQLite cache; the index grows on disk.
    Set summarize=False for intermediate shards to avoid rescanning all signatures.
    Reports describe indexed snapshots, not unvisited or subsequently removed files.
    """
    if type(summarize) is not bool:
        raise ValueError("summarize must be boolean")
    shard = coverage_shard(plan, index)
    root = out / "shards" / f"{index:08d}"
    lock_path = root / ".writer.lock"
    if not lock_path.exists():
        raise ValueError("content audit requires an initialized shard")
    database.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("cannot audit an actively written shard") from error
        report = inspect_coverage_shard(plan, index, out)
        with closing(sqlite3.connect(database, timeout=0)) as connection, connection:
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-8192")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            expected = {"plan_id": plan.identity, "version": _VERSION}
            if metadata and metadata != expected:
                raise ValueError("content index belongs to a different plan or signature version")
            if not metadata:
                if (
                    connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name!='metadata'"
                    ).fetchone()
                    is not None
                ):
                    raise ValueError("refusing an unbound existing content database")
                connection.executemany("INSERT INTO metadata VALUES (?,?)", expected.items())
            connection.execute(
                "CREATE TABLE IF NOT EXISTS tasks (task_id INTEGER PRIMARY KEY, "
                "shard INTEGER NOT NULL, "
                "stratum TEXT NOT NULL, signature TEXT NOT NULL, "
                "rows INTEGER NOT NULL, queries INTEGER NOT NULL)"
            )
            connection.execute("CREATE INDEX IF NOT EXISTS signatures ON tasks(signature)")
            connection.execute("CREATE INDEX IF NOT EXISTS shards ON tasks(shard)")
            seen = 0
            for path in root.glob("groups/*/*/data-*.zip"):
                if not path.with_suffix(".json").exists():
                    continue
                for task in load_group(path):
                    task_id = json.loads(task.recipe_json)["task_id"]
                    row = (
                        task_id,
                        index,
                        shard.stratum,
                        task_content_signature(task),
                        task.inputs.shape[0],
                        len(task.query_indices),
                    )
                    existing = connection.execute(
                        "SELECT * FROM tasks WHERE task_id=?", (task_id,)
                    ).fetchone()
                    if existing is not None and existing != row:
                        raise ValueError("indexed task content changed")
                    connection.execute("INSERT OR IGNORE INTO tasks VALUES (?,?,?,?,?,?)", row)
                    seen += 1
            indexed = connection.execute(
                "SELECT COUNT(*) FROM tasks WHERE shard=?", (index,)
            ).fetchone()[0]
            if indexed != seen or seen != report["committed"]:
                raise ValueError("indexed shard lost or duplicated committed tasks")
            if not summarize:
                return {
                    "plan_id": plan.identity,
                    "shard": index,
                    "indexed_tasks": seen,
                    "scope": "this shard only; aggregate summary not requested",
                }
            return _summary(connection, plan.identity)


def _summary(connection: sqlite3.Connection, plan_id: str) -> dict[str, Any]:
    count, unique, rows, queries = connection.execute(
        "SELECT COUNT(*),COUNT(DISTINCT signature),COALESCE(SUM(rows),0),"
        "COALESCE(SUM(queries),0) FROM tasks"
    ).fetchone()
    groups, largest = connection.execute(
        "SELECT COUNT(*),COALESCE(MAX(n),0) FROM "
        "(SELECT COUNT(*) n FROM tasks GROUP BY signature HAVING n>1)"
    ).fetchone()
    strata = [
        {"stratum": key, "tasks": tasks, "rows": row_count, "query_targets": query_count}
        for key, tasks, row_count, query_count in connection.execute(
            "SELECT stratum,COUNT(*),SUM(rows),SUM(queries) FROM tasks "
            "GROUP BY stratum ORDER BY stratum"
        )
    ]
    return {
        "plan_id": plan_id,
        "signature_version": _VERSION,
        "scope": "indexed committed snapshots only; unvisited shards excluded",
        "tasks": count,
        "unique_observed_tasks": unique,
        "duplicate_excess": count - unique,
        "duplicate_excess_fraction": (count - unique) / count if count else None,
        "duplicate_groups": groups,
        "largest_duplicate_group": largest,
        "rows": rows,
        "query_targets": queries,
        "strata": strata,
        "mechanism_equivalence": "NOT_MEASURED",
        "near_duplicates": "NOT_MEASURED",
    }
