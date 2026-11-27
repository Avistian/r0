"""Actual converter trace dependencies, including independent forced captures."""

from r0.modules.priors.lib.converter_intervention import ConverterHardIntervention
from r0.modules.priors.lib.converter_intervention_execution import ConverterInterventionExecution


def converted_trace_order(role: str) -> tuple[int, str]:
    order = {
        "root": 0,
        "inputs": 1,
        "function": 2,
        "exogenous": 3,
        "scale": 4,
        "uniform": 5,
        "scores": 6,
        "probabilities": 7,
        "capture": 8,
        "representation": 9,
        "converter-state": 10,
        "state": 11,
        "readout": 12,
    }
    return order.get(role.split(":")[0], 13), role


def converted_trace_parents(
    execution: ConverterInterventionExecution,
    index: int,
    role: str,
) -> list[str]:
    trace = execution.traces[index]
    node = execution.nodes[index]
    prefix = f"intervention-{index}-"
    if index in execution.bypassed or role in ("root", "exogenous") or role.startswith("uniform:"):
        return []
    if role == "inputs":
        return (
            [f"intervention-{p}-state" for p in node.equation.parents]
            if node.equation.parents
            else [prefix + "root"]
        )
    if role == "function":
        return [prefix + "inputs"]
    if role == "scale":
        return (
            [prefix + "function"]
            if node.equation.noise_response.strength and "function" in trace
            else []
        )
    kind, _, name = role.partition(":")
    clamp = execution.clamp(index, name)
    if kind == "scores":
        return [] if clamp else [prefix + "function"]
    if kind == "probabilities":
        return [prefix + "scores:" + name]
    if kind == "capture":
        return [prefix + "probabilities:" + name] + (
            [prefix + "uniform:" + name] if not clamp and "uniform:" + name in trace else []
        )
    if kind == "representation":
        converter = next(c for c in node.converters if c.name == name)
        if (
            isinstance(clamp, ConverterHardIntervention)
            and clamp.location == "representation_and_capture"
        ):
            return [prefix + "capture:" + name]
        # Capture-only cuts leave the original representation mechanism unchanged.
        if clamp or converter.mode.endswith("_id"):
            return [prefix + "function"]
        return [prefix + "capture:" + name]
    if role == "converter-state":
        return [prefix + key for key in trace if key.startswith("representation:")]
    if role == "state":
        overwritten = {
            channel
            for c in execution.changes(index)
            if c.location != "readout"
            for channel in c.channels
        }
        if len(overwritten) == node.equation.function.output_width:
            return []
        return [
            prefix + key
            for key in trace
            if key in ("function", "exogenous", "scale") or key.startswith("representation:")
        ]
    if role == "readout":
        return [prefix + "state"] + (
            [prefix + "function"]
            if node.equation.pipeline and node.equation.pipeline.converters
            else []
        )
    return []
