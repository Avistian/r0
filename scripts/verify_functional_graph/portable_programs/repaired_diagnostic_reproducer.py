"""Independent authored-slice diagnostic oracle."""

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from r0.modules import priors as p

ROOT = Path(os.environ["R0_VERIFY_OUT"])
identity = p.FrozenLinearFunction(((1.0,), (2.0,)))
# Explicitly constant internal function: output = 0 * parent, independently of rows.
constant = p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((0.0, 0.0),)))
recipe = p.FunctionalCausalRecipe(
    p.LinearRecipe(rows=16, features=1, signal_features=1, active_features=1, context_rows=8),
    (p.FunctionalCausalNode((), identity, root=p.JointRootBase("uniform", 1)), constant),
    ((0, 0),),
    ((0, 1),),
)
(ROOT / "recipe.json").write_text(json.dumps(asdict(recipe), indent=2) + "\n")
task = p.generate_task(recipe, seed=7, task_id=11)
card = json.loads(task.mechanism_json)
(ROOT / "mechanism.json").write_text(json.dumps(card, indent=2) + "\n")
trace = p.explain_generation(task, [15, 0, 15])
(ROOT / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
node = next(n for n in trace["nodes"] if n["id"] == "causal-1")
print(
    json.dumps(
        {
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "literal_constant_equation": "node_1 = 0 * node_0[0] + 0 * node_0[1] = 0",
            "retained_internal_values": node["values"],
            "mechanism_flags": card["flags"],
            "mechanism_diagnostics": card.get("node_diagnostics", "ABSENT"),
            "constant_node_diagnostic_present": "constant"
            in json.dumps(card.get("flags", []) + card.get("node_diagnostics", [])),
            "signal_max_error": trace["signal_max_error"],
        },
        indent=2,
    )
)
assert all(row == [0.0] for row in node["values"]), "Independent literal oracle failed"
assert {"constant_function_sample", "constant_state_sample"} <= set(card["flags"])
assert any(e["node"] == 1 and e["constant_channels"] == [0] for e in card["node_diagnostics"])
print("PASS: original constant-internal-node defect repaired; sample diagnostics retained.")
