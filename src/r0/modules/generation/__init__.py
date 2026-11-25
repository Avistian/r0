"""Persistent generation outputs shared by local and hosted run entrypoints."""

from r0.modules.generation.lib.groups import GroupReceipt as GroupReceipt
from r0.modules.generation.lib.groups import save_group as save_group
from r0.modules.generation.lib.groups import verify_group as verify_group
from r0.modules.generation.lib.runs import RunRequest as RunRequest
from r0.modules.generation.lib.runs import RunSummary as RunSummary
from r0.modules.generation.lib.runs import run_generation as run_generation
