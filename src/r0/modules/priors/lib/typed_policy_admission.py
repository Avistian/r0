"""Shared metadata-only policy admission precedes every construction replay."""

from dataclasses import asdict

from r0.modules.priors.lib.typed_policy_planning import admit_typed_record
from r0.modules.priors.lib.typed_policy_request import TypedInterventionPolicyRequest


def admit_typed_policy(request: TypedInterventionPolicyRequest) -> None:
    admit_typed_record(asdict(request), request.limits)
