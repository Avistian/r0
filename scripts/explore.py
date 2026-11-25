"""Start the local dataset explorer; generates only on explicit browser actions."""

import argparse
from pathlib import Path

from r0.modules.explorer import serve_explorer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/explorer"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve_explorer(args.out, port=args.port)
