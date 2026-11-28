"""Versioned pipeline controls preserve legacy records without new default fields."""

from typing import Any, cast

from r0.modules.priors.lib.functional_causal_decoding import functional_array, functional_record
from r0.modules.priors.lib.joint_converter_plans import JointNumericalConverterPrior
from r0.modules.priors.lib.joint_innovation_plans import JointInnovationPrior
from r0.modules.priors.lib.joint_pipeline_plans import (
    JointInnovationPipelinePrior,
    JointPipelinePrior,
)


def pipeline_prior_from_dict(data: object) -> JointPipelinePrior:
    try:
        mapping = cast(dict[str, Any], data) if type(data) is dict else {}
        extended = mapping.get("version") == "r0-innovation-pipeline-prior-v2"
        kind = JointInnovationPipelinePrior if extended else JointPipelinePrior
        args = functional_record(data, kind)
        args["converter"] = JointNumericalConverterPrior(
            **functional_record(args["converter"], JointNumericalConverterPrior)
        )
        if extended:
            law = functional_record(args["innovation"], JointInnovationPrior)
            for key in (
                "families",
                "factors",
                "truncation",
                "degrees",
                "strengths",
                "powers",
                "location",
                "input_scale",
                "mixture_gap",
                "structural_scale",
                "offset",
            ):
                if key in law:
                    law[key] = functional_array(law[key])
            args["innovation"] = JointInnovationPrior(**law)
        return kind(**args)
    except (TypeError, KeyError, OverflowError, RecursionError) as error:
        raise ValueError("malformed sampled pipeline prior") from error
