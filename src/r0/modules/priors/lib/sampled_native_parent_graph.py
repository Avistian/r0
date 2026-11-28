"""Native parent construction retains every independent ancestor-fit interval."""

from dataclasses import dataclass

from r0.modules.priors.lib.native_parent_request import NativeParentGraphRequest
from r0.modules.priors.lib.parent_function_realization import ParentCalibrationReceipt
from r0.modules.priors.lib.sampled_native_graph import SampledNativeGraph


@dataclass(frozen=True, slots=True)
class SampledNativeParentGraph(SampledNativeGraph):
    calibrations: tuple[ParentCalibrationReceipt, ...] = ()

    def validate_replay(self) -> None:
        if type(self.request) is not NativeParentGraphRequest:
            raise ValueError("native parent archive requires its versioned request")
        previous = 0
        for receipt in self.calibrations:
            if (
                type(receipt) is not ParentCalibrationReceipt
                or receipt.version != "parent-calibration-receipt-v1"
                or not 0 <= receipt.node < len(self.mechanisms)
                or receipt.parents != self.request.parents[receipt.node]
                or not previous <= receipt.raw_start < receipt.raw_stop <= len(self.raw)
            ):
                raise ValueError("native parent calibration receipt differs")
            owner = ("native", f"receiver:{receipt.node}", receipt.role)
            if any(
                r.stream != "calibration" or r.owner[:3] != owner
                for r in self.raw[receipt.raw_start : receipt.raw_stop]
            ):
                raise ValueError("native parent raw fit provenance differs")
            previous = receipt.raw_stop
        SampledNativeGraph.validate_replay(self)
