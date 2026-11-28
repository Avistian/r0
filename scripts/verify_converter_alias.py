"""Independently reproduce the tensor-view premise of the source converter audit."""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch


def run(output: Path) -> None:
    rows = []
    for variant in ("historical-view", "historical-copy", "corrected-readout"):
        state = torch.tensor([[2.0], [4.0]], dtype=torch.float64)
        selected = state[:, :1]
        warped = selected / 4
        if variant == "historical-view":
            propagated, readout = warped, selected
        elif variant == "historical-copy":
            propagated, readout = warped, selected.clone()
        else:
            propagated, readout = selected, warped
        state[:, :1] = propagated
        expected_state = [[2.0], [4.0]] if variant == "corrected-readout" else [[0.5], [1.0]]
        expected_readout = [[2.0], [4.0]] if variant == "historical-copy" else [[0.5], [1.0]]
        assert state.tolist() == expected_state
        assert readout.tolist() == expected_readout
        assert (state.data_ptr() == readout.data_ptr()) == (variant == "historical-view")
        rows.append(
            {
                "variant": variant,
                "state": state.tolist(),
                "readout": readout.tolist(),
                "shared_storage": state.data_ptr() == readout.data_ptr(),
            }
        )
    record = {
        "status": "PASS",
        "scope": "alias premise only; no upstream or r0 generator execution",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "dtype": "float64",
        "cases": rows,
    }
    with output.open("x") as handle:
        handle.write(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
