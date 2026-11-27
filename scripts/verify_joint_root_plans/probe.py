"""Independent public-API checks of root plan composition, laws and rejection."""

import ast
import copy
import itertools
import json
import math
import traceback
from dataclasses import asdict, replace

import numpy as np
from verification import INPUTS, prepare

from r0.modules import priors as p

OUT = prepare(p)
RESULTS = []
COVERAGE = set()
BASES = ("uniform", "ball", "gaussian", "covariance")
FUNCTIONS = ("mlp", "tree", "disc", "lin", "quad", "gp", "em", "prod")
MATRICES = ("gaussian", "importance", "factorized", "kernel", "transformed")
CHEAP = ("gp", "tree", "disc", "lin", "quad")


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + "\n")


def check(label, condition):
    if not condition:
        raise AssertionError(label)


def reject(label, call):
    try:
        call()
    except (ValueError, TypeError, KeyError) as exc:
        RESULTS.append(
            {"case": label, "result": "REJECTED", "type": type(exc).__name__, "reason": str(exc)}
        )
    else:
        raise AssertionError("invalid case accepted: " + label)


def context(seed=5, **kwargs):
    return p.JointSamplingContext(np.random.default_rng(seed), **kwargs)


def prior(base, function, matrix="gaussian", weight="polynomial", **kwargs):
    return p.JointRootPrior(
        families=(base,),
        function=p.JointFunctionPrior(
            functions=function,
            matrix=p.JointMatrixPrior(families=(matrix,), vector_family=weight),
            **kwargs,
        ),
    )


def sample(ctx, root_prior, name="root", width=2, rows=4):
    return p.sample_joint_root(ctx, name, width=width, calibration_rows=rows, prior=root_prior)


def events(record, path):
    return [e for e in record.events if e.operation[: len(path)] == path]


def draw(record, event):
    return getattr(record, event.kind).draws[event.draw_index]


def source_catalogs():
    tree = ast.parse((INPUTS / "_points.py").read_text())
    catalogs = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "choice"
            and isinstance(node.args[0], ast.Constant)
        ):
            catalogs[node.args[0].value] = [x.id for x in node.args[1].elts]
    check(
        "source root catalog",
        catalogs["random_base_points"]
        == [
            "RandomUniformPoints",
            "RandomCirclePoints",
            "RandomGaussianPoints",
            "RandomCovariancePoints",
        ],
    )
    check(
        "source covariance catalog",
        catalogs["random_covariance_base_points"]
        == ["RandomUniformPoints", "RandomGaussianPoints"],
    )
    tree = ast.parse((INPUTS / "I2func.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "presets_dict" for t in node.targets
        ):
            presets = ast.literal_eval(node.value)
            check("source default", tuple(presets["default"].split(",")) == FUNCTIONS)
            check("source tabpfnv2", presets["tabpfnv2"] == "mlp,tree,disc")
    for catalog, expected in (
        ("default", FUNCTIONS),
        ("tabpfnv2", FUNCTIONS[:3]),
        ("lin,default,lin", ("lin", *FUNCTIONS, "lin")),
    ):
        actual = p.JointFunctionPrior(functions=catalog).request().choices
        check(
            "ordered catalog slots",
            actual == tuple(f"function-slot-{i}-{f}-v1" for i, f in enumerate(expected)),
        )
    check(
        "cheap fixed catalog",
        p.JointFunctionPrior(functions="mlp").request(cheap=True).choices
        == tuple(f"function-slot-{i}-{f}-v1" for i, f in enumerate(CHEAP)),
    )
    check(
        "root full catalog",
        p.JointRootPrior().request().choices == tuple(f"{base}-root-base-v1" for base in BASES),
    )
    RESULTS.append({"case": "cached_source_catalogs", "result": "PASS", "catalogs": catalogs})


def numeric(record):
    for d in record.numerical.draws:
        q = d.request
        for unit, actual in zip(d.units, d.values, strict=True):
            u = np.clip(1.2 * unit - 0.1, 0, 1) if q.boundary_mass else unit
            value = (
                math.exp(math.log(q.low) * (1 - u) + math.log(q.high) * u)
                if q.scale == "log"
                else q.low * (1 - u) + q.high * u
            )
            if q.integer == "round":
                value = float(np.rint(value))
            if q.integer in ("source-clip-floor-v1", "floor"):
                value = float(np.clip(math.floor(value), q.low, q.high - 1))
            check(
                "literal numeric interpolation " + q.name,
                math.isclose(value, actual, rel_tol=2e-13, abs_tol=1e-12),
            )


def weight_oracle(weight, record):
    es = events(record, weight.operation)
    count = weight.batch if weight.scalarization == "scalarized-weight-batches-v1" else 1
    check("weight event count", len(es) == (2 * count if weight.family == "polynomial" else 0))
    if weight.family == "polynomial":
        for i, e in enumerate(es):
            q = draw(record, e).request
            decay = i < count
            check(
                "weight source name",
                q.name == "simple_random_weights_" + ("decay_rate" if decay else "std_scale"),
            )
            check(
                "weight source range",
                math.isclose(q.low, 0.1 / math.log1p(weight.width) if decay else 0.0001)
                and q.high == (6 if decay else 10),
            )
            check("weight source law", q.scale == "log" and q.boundary_mass == decay)
    return es


def function_oracle(f, record):
    own = [e for e in events(record, f.operation) if e.operation == f.operation]
    check("function own initial selector", own[0].role == "family")
    q = draw(record, own[0]).request
    check(
        "function selector name",
        q.name == ("cheap_random_function_fct_type" if f.cheap else "random_function_fct_type"),
    )
    family = f.slot.split("-")[-2]
    COVERAGE.add("function:" + family)
    v = dict(f.parameters)
    expected = {
        "mlp": ["nn_width", "nn_n_layers", "nn_start_act", "nn_end_act"],
        "tree": ["rt_n_trees", "rt_depth"],
        "lin": [],
        "quad": [],
        "prod": [],
        "disc": (
            ["rand_discr_n_centers"]
            if f.calibration_rows > 1 and f.prior.discretization_centers is None
            else []
        )
        + ["cat_disc_p"],
        "gp": ["gp_a", "generalized_gp_use_product_kernel"]
        + (["rand_gen_gp_scale"] if v.get("product_kernel") is False else []),
        "em": [
            "random_em_components",
            "gen_em_assignment_std_factor",
            "gen_em_assignment_p_inv",
            "gen_em_assignment_q",
        ],
    }[family]
    check("function own source order", [draw(record, e).request.name for e in own[1:]] == expected)
    expected_events = [own[0]]
    for e in own[1:]:
        if family == "gp" and e.role == "scale":
            expected_events += weight_oracle(f.weights, record)
        expected_events.append(e)
    if family == "mlp":
        layers, width = v["layers"], v["width"]
        check("all MLP linears", len(f.matrices) == layers)
        check(
            "all MLP activations",
            len(f.activations) == layers - 1 + v["start_activation"] + v["end_activation"],
        )
        widths = [f.input_width] + [width] * (layers - 1) + [f.output_width]
        children = []
        index = 0
        if v["start_activation"]:
            children.append(f.activations[index])
            index += 1
        for i, matrix in enumerate(f.matrices):
            check(
                "MLP source geometry",
                (matrix.rows, matrix.columns, matrix.batch) == (widths[i + 1], widths[i], 1),
            )
            children.append(matrix)
            if i < layers - 1 or v["end_activation"]:
                children.append(f.activations[index])
                index += 1
        COVERAGE.add(f"mlp:{layers}:{v['start_activation']}:{v['end_activation']}")
    elif family == "prod":
        check("two cheap children", len(f.children) == 2 and all(c.cheap for c in f.children))
        children = list(f.children)
        for side, child in zip(("left", "right"), children, strict=True):
            check("product child path", child.operation == (*f.operation, side))
            check(
                "product shape",
                (child.input_width, child.output_width) == (f.input_width, f.output_width),
            )
            COVERAGE.add("cheap:" + side + ":" + child.family)
            function_oracle(child, record)
    else:
        children = list(f.matrices)
        if family in ("lin", "disc", "em", "quad"):
            check("one matrix", len(children) == 1)
            size = min(f.input_width, 20) + 1
            shape = (
                (size, size, f.output_width)
                if family == "quad"
                else (f.output_width, v["components"] if family == "em" else f.input_width, 1)
            )
            check(
                "source matrix geometry",
                (children[0].rows, children[0].columns, children[0].batch) == shape,
            )
        if family == "gp":
            COVERAGE.add("gp:" + str(v["product_kernel"]))
            check("GP no normalized matrix", not f.matrices)
    for child in children:
        expected_events += events(record, child.operation)
    check("full function subtree event order", events(record, f.operation) == expected_events)
    for matrix in f.matrices:
        COVERAGE.add("matrix:" + matrix.family)
        if matrix.child:
            COVERAGE.add("simple:" + matrix.child.family)
            check(
                "simple matrix geometry",
                (matrix.child.rows, matrix.child.columns, matrix.child.batch)
                == (matrix.rows, matrix.columns, matrix.batch),
            )
        if matrix.weights:
            weight_oracle(matrix.weights, record)


def root_oracle(root, record):
    es = events(record, root.operation)
    check("root starts with base", es[0].operation == root.operation and es[0].role == "base")
    check("base binding", draw(record, es[0]).value == root.family + "-root-base-v1")
    prefix = [es[0]]
    if root.family == "covariance":
        check(
            "raw factor weight geometry",
            (root.weights.width, root.weights.batch) == (root.width, 1),
        )
        check("raw weight path", root.weights.operation == (*root.operation, "weights"))
        prefix += weight_oracle(root.weights, record)
        cov = es[len(prefix)]
        check(
            "covariance base follows weights",
            cov.operation == root.operation and cov.role == "covariance_base",
        )
        check(
            "covariance literal catalog",
            draw(record, cov).request.choices == ("uniform-root-base-v1", "gaussian-root-base-v1"),
        )
        prefix.append(cov)
        COVERAGE.add("covariance:" + root.covariance_base)
    else:
        check("noncovariance no factor", root.weights is None and root.covariance_base is None)
    f = root.function
    check(
        "full width preserving function",
        not f.cheap
        and (f.input_width, f.output_width, f.calibration_rows)
        == (root.width, root.width, root.calibration_rows),
    )
    check("function ownership", f.operation == (*root.operation, "function"))
    check("whole root named order", es == prefix + events(record, f.operation))
    function_oracle(f, record)
    numeric(record)


def composition():
    for index, (base, fun, matrix, weight) in enumerate(
        itertools.product(BASES, FUNCTIONS, MATRICES, ("uniform", "polynomial"))
    ):
        ctx = context(index, mode="local")
        root = sample(ctx, prior(base, fun, matrix, weight))
        record = ctx.freeze()
        save(f"fixture-{index:03}.json", {"plan": asdict(root), "decisions": asdict(record)})
        root_oracle(root, record)
        payload = json.loads(json.dumps(asdict(root)))
        check("roundtrip", p.joint_root_plan_from_dict(payload, decisions=record) == root)
    RESULTS.append(
        {
            "case": "all_base_function_matrix_weight_combinations",
            "count": index + 1,
            "result": "PASS",
        }
    )


def raw_cost(payload):
    pending = [payload]
    numbers = containers = strings = 0
    while pending:
        value = pending.pop()
        if type(value) in (bool, int, float):
            numbers += 1
        elif type(value) is str:
            strings += len(value.encode("utf-8"))
        elif type(value) is dict:
            containers += 1
            pending.extend(value.keys())
            pending.extend(value.values())
        elif type(value) in (list, tuple):
            containers += 1
            pending.extend(value)
    return numbers, 64 * numbers + 32 * containers + strings


def adversarial():
    ctx = context()
    root = sample(ctx, prior("covariance", "lin", weight="uniform"))
    record = ctx.freeze()
    payload = json.loads(json.dumps(asdict(root)))
    n, storage = raw_cost(payload)
    check(
        "exact raw numeric and bytes admission",
        p.joint_root_plan_from_dict(
            payload, decisions=record, limits=p.CausalLimits(parameters=n, working_bytes=storage)
        )
        == root,
    )
    for label, limits in (
        ("numeric boundary", p.CausalLimits(parameters=n - 1)),
        ("storage boundary", p.CausalLimits(working_bytes=storage - 1)),
    ):
        reject(
            label,
            lambda limits=limits: p.joint_root_plan_from_dict(
                payload, decisions=record, limits=limits
            ),
        )
    RESULTS.append(
        {
            "case": "independent raw admission count",
            "numbers": n,
            "estimated_bytes": storage,
            "result": "PASS",
        }
    )
    mutations = [
        (["width"], True),
        (["width"], 3),
        (["calibration_rows"], 0),
        (["version"], "v999"),
        (["function"], None),
        (["function"], []),
        (["prior"], []),
        (["prior", "version"], "v999"),
        (["prior", "function", "version"], "v999"),
        (["prior", "function", "matrix", "version"], "v999"),
        (["prior", "function", "matrix", "unknown"], 1),
        (["prior", "function", "unknown"], 1),
        (["function", "prior", "matrix", "unknown"], 1),
        (["function", "operation"], ["root", "weights"]),
        (["weights", "operation"], ["root", "function"]),
        (["weights", "batch"], True),
        (["weights", "version"], "v999"),
        (["function", "cheap"], True),
        (["function", "output_width"], 3),
        (["function", "matrices"], [[1], [1, 2]]),
        (["operation"], np.array(["root"], dtype=object)),
        (["width"], math.inf),
        (["width"], math.nan),
        (["prior", "function", "frequencies"], object()),
        (["prior", "families"], ["covariance", "covariance"]),
    ]
    deep = []
    for _ in range(30):
        deep = [deep]
    mutations.append((["operation"], deep))
    for path, value in mutations:
        changed = copy.deepcopy(payload)
        node = changed
        for part in path[:-1]:
            node = node[part]
        node[path[-1]] = value
        reject(
            "mutate:" + "/".join(path) + ":" + repr(value)[:80],
            lambda changed=changed: p.joint_root_plan_from_dict(changed, decisions=record),
        )
    for bad in ([], object(), {1: "root"}):
        reject(
            "malformed root " + repr(bad),
            lambda bad=bad: p.joint_root_plan_from_dict(bad, decisions=record),
        )
    bad_events = list(record.events)
    bad_events[0] = replace(bad_events[0], role="wrong")
    reject(
        "changed root event role", lambda: root.validate(replace(record, events=tuple(bad_events)))
    )
    reject("duplicate operation", lambda: sample(ctx, root.prior))


def sharing_and_rollback():
    for mode in ("local", "meta", "global"):
        rng = np.random.default_rng(55)
        ctx = p.JointSamplingContext(rng, mode=mode)
        good = prior("covariance", "tree")
        first = sample(ctx, good, "existing")
        before, state = ctx.freeze(), copy.deepcopy(rng.bit_generator.state)
        bad = prior("covariance", "disc", max_discretization_cardinality=1)
        reject(
            mode + " nested function failure", lambda ctx=ctx, bad=bad: sample(ctx, bad, "retry")
        )
        check("atomic ledger rollback", ctx.freeze() == before)
        check("atomic caller RNG rollback", rng.bit_generator.state == state)
        second = sample(ctx, good, "retry")
        third = sample(ctx, good, "third")
        record = ctx.freeze()
        root_oracle(second, record)
        root_oracle(third, record)
        grouped = {}
        for d in record.numerical.draws:
            grouped.setdefault(d.request.name, []).append(d)
        for name, ds in grouped.items():
            check("one shared preference per name", len({d.preference for d in ds}) == 1)
            if mode == "global":
                check("global first-value reuse " + name, all(d.values == ds[0].values for d in ds))
            else:
                check("fresh local/meta draws " + name, len({d.units for d in ds}) > 1)
        if mode == "meta":
            for pref in record.categorical.preferences:
                check("separate categorical hyperprior scope", pref.hyperparameters is not None)
                check("nested weight scope is not outer", pref.hyperparameters != record.numerical)
        before, state = ctx.freeze(), copy.deepcopy(rng.bit_generator.state)
        try:
            with ctx.transaction("outer") as nested:
                sample(nested, good, "root")
                raise RuntimeError("deliberate post-root enclosing transaction failure")
        except RuntimeError:
            pass
        check("outer root rollback ledger", ctx.freeze() == before)
        check("outer root rollback RNG", rng.bit_generator.state == state)
        with ctx.transaction("outer") as nested:
            sample(nested, good, "root")
        first.validate(ctx.freeze())
        RESULTS.append({"case": mode + " sharing and rollback", "result": "PASS"})


try:
    source_catalogs()
    composition()
    adversarial()
    sharing_and_rollback()
    required = {"function:" + x for x in FUNCTIONS} | {"matrix:" + x for x in MATRICES}
    required |= {"covariance:uniform", "covariance:gaussian", "gp:True", "gp:False"}
    check("mandatory family coverage", required <= COVERAGE)
    save(
        "report.json",
        {"verdict": "PASS_COMPONENT", "results": RESULTS, "coverage": sorted(COVERAGE)},
    )
    print("PASS_COMPONENT", len(RESULTS), "checks", len(COVERAGE), "coverage cells")
except Exception:
    save(
        "report.json",
        {
            "verdict": "FAIL_REQUIRES_CLASSIFICATION",
            "results": RESULTS,
            "coverage": sorted(COVERAGE),
            "traceback": traceback.format_exc(),
        },
    )
    raise
