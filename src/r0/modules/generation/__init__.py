"""Persistent generation outputs shared by local and hosted run entrypoints."""

from r0.modules.generation.lib.causal_pairs import load_causal_pair as load_causal_pair
from r0.modules.generation.lib.causal_pairs import save_causal_pair as save_causal_pair
from r0.modules.generation.lib.config import load_request as load_request
from r0.modules.generation.lib.config import load_request_document as load_request_document
from r0.modules.generation.lib.construction import GraphConstruction as GraphConstruction
from r0.modules.generation.lib.content_audit import index_coverage_shard as index_coverage_shard
from r0.modules.generation.lib.content_audit import task_content_signature as task_content_signature
from r0.modules.generation.lib.coverage import (
    CoverageAllocation as CoverageAllocation,
)
from r0.modules.generation.lib.coverage import (
    CoveragePlan as CoveragePlan,
)
from r0.modules.generation.lib.coverage import (
    CoverageShard as CoverageShard,
)
from r0.modules.generation.lib.coverage import (
    CoverageStratum as CoverageStratum,
)
from r0.modules.generation.lib.coverage import (
    coverage_allocations as coverage_allocations,
)
from r0.modules.generation.lib.coverage import (
    coverage_plan_from_json as coverage_plan_from_json,
)
from r0.modules.generation.lib.coverage import (
    coverage_shard as coverage_shard,
)
from r0.modules.generation.lib.coverage import (
    planned_coverage as planned_coverage,
)
from r0.modules.generation.lib.coverage import (
    run_coverage_shard as run_coverage_shard,
)
from r0.modules.generation.lib.coverage_hierarchy import CoverageHierarchy as CoverageHierarchy
from r0.modules.generation.lib.coverage_report import inspect_coverage as inspect_coverage
from r0.modules.generation.lib.coverage_report import (
    inspect_coverage_shard as inspect_coverage_shard,
)
from r0.modules.generation.lib.groups import GroupReceipt as GroupReceipt
from r0.modules.generation.lib.groups import save_group as save_group
from r0.modules.generation.lib.groups import verify_group as verify_group
from r0.modules.generation.lib.job import JobResult as JobResult
from r0.modules.generation.lib.job import run_guarded as run_guarded
from r0.modules.generation.lib.job import run_guarded_document as run_guarded_document
from r0.modules.generation.lib.loading import load_group as load_group
from r0.modules.generation.lib.missingness_request import ObservationRequest as ObservationRequest
from r0.modules.generation.lib.requests import RecipeSampling as RecipeSampling
from r0.modules.generation.lib.requests import RunRequest as RunRequest
from r0.modules.generation.lib.requests import request_from_dict as request_from_dict
from r0.modules.generation.lib.runs import RunSummary as RunSummary
from r0.modules.generation.lib.runs import run_generation as run_generation
