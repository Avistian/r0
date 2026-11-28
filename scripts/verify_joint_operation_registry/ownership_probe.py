"""Independent preorder/span model, corruption rejection and bounded lifecycle checks."""

import copy
import json
import pickle
from dataclasses import asdict

import numpy as np
from common import rejected, require, save, start

from r0.modules import priors


def main():
    output = start()
    cases = []
    for seed in range(25):
        model_rng = np.random.default_rng(seed)
        context = priors.JointSamplingContext(np.random.default_rng(1901))
        model_events, model_order = [], []

        def build(
            parent,
            path,
            depth,
            model_order=model_order,
            model_events=model_events,
            model_rng=model_rng,
        ):
            model_order.append((path, len(model_events)))
            with parent.transaction(path[-1]) as scope:
                for before in range(int(model_rng.integers(0, 3))):
                    scope.forced_identity(f"before-{before}", reason="literal")
                    model_events.append(path)
                if depth:
                    for child in range(int(model_rng.integers(0, 4))):
                        build(scope, (*path, f"child-{child}"), depth - 1)
                for after in range(int(model_rng.integers(0, 3))):
                    scope.forced_identity(f"after-{after}", reason="literal")
                    model_events.append(path)

        for root in range(3):
            build(context, (f"root-{root}",), 3)
        registry = context.freeze_registry()
        expected = []
        for path, opened in model_order:
            owned = [
                index for index, owner in enumerate(model_events) if owner[: len(path)] == path
            ]
            expected.append(
                (path, owned[0] if owned else opened, owned[-1] + 1 if owned else opened)
            )
        actual = [
            (span.operation, span.event_start, span.event_stop) for span in registry.operations
        ]
        require(actual == expected, "independent preorder/span model mismatch")
        require(
            [event.operation for event in registry.decisions.events] == model_events,
            "event model mismatch",
        )
        require(
            priors.joint_operation_registry_from_dict(asdict(registry)) == registry,
            "model roundtrip mismatch",
        )
        cases.append({"seed": seed, "operations": len(model_order), "events": len(model_events)})
    context = priors.JointSamplingContext(np.random.default_rng(19))
    with context.transaction("root") as root:
        with root.transaction("a") as a:
            a.forced_identity("a", reason="literal")
        with root.transaction("empty"):
            pass
        with root.transaction("b") as b:
            b.forced_identity("b", reason="literal")
    original = json.loads(json.dumps(asdict(context.freeze_registry())))
    save(output, "valid-corruption-control", original)
    mutations = {}
    for label in (
        "duplicate",
        "orphan",
        "missing-parent",
        "out-of-preorder",
        "overlap",
        "child-range",
        "empty-nonempty",
        "bool-index",
        "span-extra",
        "foreign-owner",
    ):
        value = copy.deepcopy(original)
        spans = value["operations"]
        if label == "duplicate":
            spans.append(copy.deepcopy(spans[1]))
        elif label == "orphan":
            del spans[1]
        elif label == "missing-parent":
            del spans[0]
        elif label == "out-of-preorder":
            spans[1], spans[3] = spans[3], spans[1]
        elif label == "overlap":
            spans[3]["event_start"] = 0
        elif label == "child-range":
            spans[1]["event_stop"] = 3
        elif label == "empty-nonempty":
            spans[2]["event_stop"] += 1
        elif label == "bool-index":
            spans[1]["event_start"] = False
        elif label == "span-extra":
            spans[1]["event_stop"] = 2
        else:
            value["decisions"]["events"][0]["operation"] = ["foreign"]
        rejected(lambda value=value: priors.joint_operation_registry_from_dict(value))
        mutations[label] = "REJECTED_VALUEERROR"
    for limits in (
        priors.JointRegistryLimits(operations=1),
        priors.JointRegistryLimits(path_bytes=3),
    ):
        rejected(
            lambda limits=limits: priors.joint_operation_registry_from_dict(
                original, registry_limits=limits
            )
        )
    for limits in (priors.CausalLimits(parameters=1), priors.CausalLimits(working_bytes=100)):
        rejected(
            lambda limits=limits: priors.joint_operation_registry_from_dict(original, limits=limits)
        )
    utf8 = priors.JointSamplingContext(
        np.random.default_rng(19), registry_limits=priors.JointRegistryLimits(path_bytes=4)
    )
    with utf8.transaction("éé"):
        pass
    rejected(lambda: utf8.transaction("a").__enter__())
    for invalid in ("", " padded", "null\0", "a" * 129, "é" * 65):
        rejected(lambda invalid=invalid: context.transaction(invalid).__enter__())
    chain = priors.JointSamplingContext(np.random.default_rng(19))

    def deep(parent, depth):
        if depth == 16:
            rejected(lambda: parent.transaction("too-deep").__enter__())
            return
        with parent.transaction("a") as child:
            deep(child, depth + 1)

    deep(chain, 0)
    require(len(chain.freeze_registry().operations) == 16, "depth boundary failed")
    for streams in ([], {"bad ": np.random.default_rng(1)}, {"raw": 1}):
        rejected(
            lambda streams=streams: priors.JointSamplingContext(
                np.random.default_rng(2), streams=streams
            )
        )
    named, raw = np.random.default_rng(1), np.random.default_rng(2)
    for limits in (priors.JointStreamLimits(count=1), priors.JointStreamLimits(state_bytes=1)):
        rejected(
            lambda limits=limits: priors.JointSamplingContext(
                named, streams={"raw": raw, "extra": np.random.default_rng(3)}, stream_limits=limits
            )
        )
    ctx = priors.JointSamplingContext(named, streams={"raw": raw})
    with ctx.transaction("root") as parent:
        handle = parent.stream("raw")
        oracle = copy.deepcopy(handle)
        before = parent.freeze()
        try:
            with parent.transaction("child") as child:
                child.stream("raw").random(10)
                handle.random(2)
        except ValueError as error:
            require("caller RNG changed" in str(error), "wrong nested conflict error")
        else:
            raise RuntimeError("parent-handle mutation was not detected")
        oracle.random(2)
        require(
            pickle.dumps(handle.bit_generator.state) == pickle.dumps(oracle.bit_generator.state),
            "nested conflict overwrote parent draw",
        )
        require(parent.freeze() == before, "nested conflict committed named decisions")
        with parent.transaction("child"):
            pass
    require(len(ctx.freeze_registry().operations) == 2, "nested conflict reserved failed path")
    save(
        output,
        "result",
        {
            "verdict": "PASS",
            "model_cases": cases,
            "corruptions": mutations,
            "bounds_and_nested_conflict": "PASS",
        },
    )
    print("PASS: 25 independent ownership models, 10 corruptions, bounds and nested mutation")


if __name__ == "__main__":
    main()
