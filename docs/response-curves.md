# Response curves from saved functions

The local and embedded Colab explorer share a response-curve control. Choose a
feature, an anchor row and 2–512 evaluation points. Other inputs stay at that row's
values; the selected feature varies over its full observed minimum/maximum. This
is a slice through the saved function, not an average over the other inputs.

The curve displays latent signal before observation noise or classification
separation/offsets. It evaluates the actual saved parameters through the public
`evaluate_signal` API, without sampling anything or reading observed labels.
Lines join the evaluated points; high-frequency behavior between them can remain
unresolved. The panel includes the fixed inputs and every evaluated value, plus
SVG/PNG exports. Feature or anchor changes clear the previous curve.

```python
from r0.modules.explorer import inspect_curve

curve = inspect_curve(task, feature=0, anchor_row=7, points=129)
```

The Python API also accepts explicit `bounds=(low, high)`, which may extrapolate
beyond the observed feature range. It returns the range source and feature role.
The task is unchanged. Inspection failures remain visible; the function is never
resampled, clipped or replaced to make a plot succeed.

Supported: linear, grouped linear, polynomial, quadratic, finite random Fourier
features, periodic, smooth radial, smooth sinusoidal and analytic compositions.
Direct GP draws explicitly reject this view: original sampled values, conditional
mean and additional conditional draws require distinct inspection semantics.
That GP view, generation graphs and paired comparisons remain pending #27 work.

Hosted Colab execution remains **NOT_RUN**. Shared assets and callback transport
make the controls available to the adapter, but local browser checks do not prove
interaction inside a real hosted notebook.
