"""Named r0 innovation profiles expose complete sampled laws through shared controls."""

from dataclasses import asdict
from typing import Any

from r0.modules import priors


def innovation_examples(geometry: dict[str, Any]) -> dict[str, Any]:
    return {
        f"Pipeline innovations: {family} (r0 extension)": {
            **geometry,
            "topology": "attachment-geometry",
            "topology_prior": asdict(priors.AttachmentGraphPrior(nodes=3)),
            "pipeline_prior": asdict(
                priors.JointInnovationPipelinePrior(
                    channel_importance=False,
                    l2_normalization=False,
                    converter=priors.JointNumericalConverterPrior(
                        allow_activation=False,
                        allow_kumaraswamy=False,
                    ),
                    innovation=priors.JointInnovationPrior(families=(family,)),
                )
            ),
        }
        for family in ("gaussian", "uniform", "student_t", "mixture", "ball")
    }
