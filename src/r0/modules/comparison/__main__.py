"""Separate offline comparison and TabPFN commands."""

import argparse
from pathlib import Path

from .lib.execution import environment
from .lib.external import ExternalPredictor
from .lib.prediction import prediction_worker
from .lib.predictors import Unsupported
from .lib.prepare import prepare
from .lib.runner import attempt, compare
from .lib.storage import read_json, write_json
from .lib.tabpfn import tabpfn_status
from .lib.tabpfn_backend import tabpfn_worker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preparation = sub.add_parser("prepare")
    preparation.add_argument("--source", type=Path, required=True)
    preparation.add_argument("--out", type=Path, required=True)
    preparation.add_argument("--queries", type=int, default=4)
    preparation.add_argument("--context", type=int, default=8)
    runner = sub.add_parser("run")
    runner.add_argument("--prepared", type=Path, required=True)
    runner.add_argument("--out", type=Path, required=True)
    runner.add_argument("--fixture-diagnostic", action="store_true")
    runner.add_argument("--fixture-predictors", action="store_true")
    runner.add_argument("--cgroup-root", type=Path)
    runner.add_argument("--external", type=Path, action="append", default=[])
    runner.add_argument("--saved", type=Path, help="JSON map of role to three attempt paths")
    runtime = sub.add_parser("environment")
    runtime.add_argument("--out", type=Path, required=True)
    tabworker = sub.add_parser("tabpfn-worker")
    tabworker.add_argument("--config", type=Path, required=True)
    tabworker.add_argument("--input", type=Path, required=True)
    tabworker.add_argument("--out", type=Path, required=True)
    child = sub.add_parser("worker")
    child.add_argument("--input", type=Path, required=True)
    child.add_argument("--out", type=Path, required=True)
    child.add_argument("--predictor", required=True)
    tabpfn = sub.add_parser("tabpfn")
    tabpfn.add_argument("--out", type=Path, required=True)
    tabpfn.add_argument("--prepared", type=Path)
    tabpfn.add_argument("--fixture-diagnostic", action="store_true")
    tabpfn.add_argument("--draw", type=int, default=1)
    tabpfn.add_argument("--descriptor", type=Path)
    tabpfn.add_argument("--cgroup-root", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "environment":
            write_json(args.out, environment())
        elif args.command == "tabpfn-worker":
            tabpfn_worker(args.input, args.out, args.config)
        elif args.command == "prepare":
            prepare(args.source, args.out, query_count=args.queries, context_count=args.context)
        elif args.command == "run":
            report = compare(
                args.prepared,
                args.out,
                diagnostic=args.fixture_diagnostic,
                cgroup_root=args.cgroup_root,
                fixture_predictors=args.fixture_predictors,
                external_predictors=[ExternalPredictor.from_file(path) for path in args.external],
                saved_attempts={
                    key: [Path(p) for p in paths] for key, paths in read_json(args.saved).items()
                }
                if args.saved
                else None,
            )
            print(report["status"], report["fixture_checks"])
            if report["fixture_checks"] != "PASS":
                raise SystemExit(1)
        elif args.command == "worker":
            prediction_worker(args.input, args.out, args.predictor)
        elif args.descriptor:
            if args.prepared is None:
                parser.error("TabPFN descriptor requires --prepared")
            descriptor = ExternalPredictor.from_file(args.descriptor)
            if descriptor.name != "tabpfn":
                parser.error("TabPFN command requires a tabpfn descriptor")
            result = attempt(
                args.prepared,
                args.out,
                descriptor,
                args.draw,
                diagnostic=args.fixture_diagnostic,
                cgroup_root=args.cgroup_root,
            )
            print(result["status"])
            if result["status"] != "SUCCESS":
                raise SystemExit(1)
        elif args.fixture_diagnostic:
            if args.prepared is None:
                parser.error("fixture TabPFN requires --prepared")
            result = attempt(args.prepared, args.out, "tabpfn_fixture", args.draw, diagnostic=True)
            print("FIXTURE ONLY (context-frequency substitute)", result["status"])
            if result["status"] != "SUCCESS":
                raise SystemExit(1)
        else:
            write_json(args.out, tabpfn_status())
            print("TabPFN UNAVAILABLE; real inference NOT_RUN")
    except Unsupported as error:
        print(str(error))
        raise SystemExit(3) from error
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
