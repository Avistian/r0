"""Feature-axis convolution: source operation, explicit r0 geometry and initializer laws."""

from dataclasses import asdict
from typing import Any

from r0.modules import priors as p


def _example(
    request: p.FeatureConv1DRequest,
    *,
    kind: str = "functional",
    categorical: bool = False,
    authored: p.FrozenFeatureConv1D | None = None,
) -> dict[str, Any]:
    width = request.input_length * request.input_channels
    length, channels = request.input_length, request.input_channels
    for prior in request.layers:
        geometry = prior.geometries[0]
        length = geometry.shape(length, channels)[0]
        channels = geometry.output_channels
    output = length * channels
    task = p.LinearRecipe(
        rows=32,
        context_rows=20,
        features=2,
        signal_features=2,
        active_features=2,
        task_type="multiclass" if categorical else "regression",
        classes=output if categorical else 0,
    )
    root = p.FunctionalCausalNode(
        (),
        p.FrozenLinearFunction(
            tuple(tuple(float(i == j) for j in range(width)) for i in range(width))
        ),
        root=p.JointRootBase("gaussian", width),
    )
    child = p.FunctionalCausalNode(
        (0,), p.FrozenLinearFunction(tuple((0.0,) * width for _ in range(output)))
    )
    observed = ((0, 0), (0, width - 1))
    if kind == "functional":
        graph = p.FunctionalCausalRecipe(
            task,
            (root, child),
            observed,
            tuple((1, i) for i in range(output)) if categorical else ((1, 0),),
        )
    elif kind == "native":
        domain = (
            p.VariableDomain("categorical", tuple(f"class-{i}" for i in range(output)), "argmax")
            if categorical
            else p.VariableDomain()
        )
        graph = p.NativeCausalRecipe(
            task, (p.NativeCausalNode(root), p.NativeCausalNode(child, domain)), observed, (1, 0)
        )
    elif kind == "vector":
        domains = (
            (p.VariableDomain("categorical", tuple(f"class-{i}" for i in range(output)), "argmax"),)
            if categorical
            else (p.VariableDomain(),) * output
        )
        graph = p.NativeVectorRecipe(
            task,
            (
                p.NativeVectorNode(root, (p.VariableDomain(),) * width),
                p.NativeVectorNode(child, domains),
            ),
            observed,
            (1, 0),
        )
    else:
        converter = p.CategoricalConverter(
            "side", 0, 1, ("negative", "positive"), "neighbor_disc", ((-1.0,), (1.0,))
        )
        graph = p.ConvertedCausalRecipe(
            task,
            (p.ConvertedCausalNode(root, (converter,)), p.ConvertedCausalNode(child, ())),
            (p.ConvertedReadout(0, capture="side"), p.ConvertedReadout(0, channel=width - 1)),
            p.ConvertedReadout(1, channel=0),
        )
    entry = (
        {"node": 1, "function": asdict(authored)}
        if authored
        else {"node": 1, "request": asdict(request)}
    )
    return {"topology": "feature-convolution", "skeleton": asdict(graph), "nodes": [entry]}


def convolution_examples() -> dict[str, Any]:
    examples: dict[str, Any] = {}
    variants = (
        ("dense odd kernel", p.Conv1DGeometry(2, 3, padding="same-upper")),
        ("even kernel + stride", p.Conv1DGeometry(2, 2, stride=2, padding="same-upper")),
        ("dilated grouped", p.Conv1DGeometry(2, 3, dilation=2, padding="same-upper", groups=2)),
        ("depthwise multiplier", p.Conv1DGeometry(4, 3, padding="same-upper", groups=2)),
        (
            "explicit asymmetric padding",
            p.Conv1DGeometry(2, 2, padding="explicit", explicit_padding=(2, 0)),
        ),
        ("valid stride", p.Conv1DGeometry(2, 2, stride=2)),
        ("identity residual", p.Conv1DGeometry(2, 3, padding="same-upper", residual=True)),
    )
    for label, geometry in variants:
        request = p.FeatureConv1DRequest(
            4,
            2,
            (
                p.Conv1DLayerPrior(
                    (geometry,),
                    activations=(p.CausalActivation("tanh"),),
                    kernel_laws=("normal", "uniform", "student_t", "cauchy", "rademacher"),
                ),
            ),
        )
        examples[f"Feature convolution: sampled {label} (r0)"] = _example(request)
    for layout, scope in (
        ("channel-major", "flat"),
        ("position-major", "positions"),
        ("position-major", "channels"),
    ):
        layers = (
            p.Conv1DLayerPrior(
                (p.Conv1DGeometry(2, 1),),
                activations=(p.CausalActivation("softmax"),),
                placement="before",
                activation_scope=scope,
            ),
            p.Conv1DLayerPrior(
                (p.Conv1DGeometry(1, 2),),
                activations=(p.CausalActivation(),),
                kernel_laws=("uniform",),
            ),
        )
        examples[f"Feature convolution: two-layer {layout} / {scope} softmax (r0)"] = _example(
            p.FeatureConv1DRequest(4, 2, layers, layout=layout)
        )
    request = p.FeatureConv1DRequest(
        4, 1, (p.Conv1DLayerPrior((p.Conv1DGeometry(1, 2),), activations=(p.CausalActivation(),)),)
    )
    authored = p.FrozenFeatureConv1D(
        4, 1, (p.FeatureConv1DLayer(p.Conv1DGeometry(1, 2), (((1.0, -2.0),),), (0.25,)),)
    )
    examples["Feature convolution: authored affine shared kernel"] = _example(
        request, authored=authored
    )
    for kind in ("native", "vector", "converted"):
        examples[f"Feature convolution: sampled {kind} graph"] = _example(request, kind=kind)
    categorical = p.FeatureConv1DRequest(
        3, 1, (p.Conv1DLayerPrior((p.Conv1DGeometry(3, 3),), activations=(p.CausalActivation(),)),)
    )
    for kind in ("functional", "native", "vector"):
        examples[f"Feature convolution: sampled {kind} multiclass"] = _example(
            categorical, kind=kind, categorical=True
        )
    return examples
