# Generation graphs from saved provenance

`explain_generation` reconstructs a graph for 1–16 selected original rows. It
returns ordinary JSON-compatible node records: identity, label, parent nodes,
formula, parameters, matrix shape, displayed column indices and values. No graph
or intermediate matrices are stored in training tasks. Inspection runs on CPU
and does not mutate inputs, targets, parameters or split membership.

```python
from r0.modules.priors import explain_generation

graph = explain_generation(task, [0, 7, 63])
child = explain_generation(composed_task, [0, 7, 63], component=1)
later_terms = explain_generation(fourier_task, [0, 7, 63], column_start=128)
```

Nodes expose saved coefficients, powers, projections, phases, smooth activations,
raw signals and calibration. Compositions expose each child and every sum/product
in the recorded expression; expand a child by its index to inspect its own steps.
Child graphs show mapped inputs and component strength, without the parent target
adapter. The root graph includes regression noise or class logits, probabilities,
uniform label draws, resulting targets and the saved context/query assignment.
Noise and uniform draws are replayed from the original versioned random stream.
Root signal and targets must agree with saved values within `rtol=atol=1e-10`;
otherwise inspection fails explicitly. Classification labels must match exactly.

Direct GP graphs expose each unweighted kernel, the combined covariance and the
original joint draw reconstructed from its complete saved support and normals.
The shown covariance is the selected task-row submatrix. Factorization still uses
the full covariance, including calibration support; it is bounded by the existing
reference GP limits and is deliberately more expensive than analytic inspection.
Normal previews show the first up to 16 support points, not selected task rows.
Kernel and coefficient matrix axes are identified in their parameters. These are
**original sampled values**, with no interpolation or additional GP sampling.

Each value matrix shows at most 32 columns, starting at `column_start`, clamped to
the last column for narrower matrices. Full shape and actual column indices make
this truncation explicit; any later term/channel can be requested. Parameter
lists show their first 32 items at each level and include the total count when
truncated. Complete parameters remain in `task.mechanism_json` and replay exports.
Matrix rows for coefficients and GP normals have their own stated meaning; they
must not be labelled as observation rows by a renderer.

The implementation uses ordinary functions and the existing signal/covariance
evaluators. The existing multiplication guard is shared under a nonprivate helper
name inside the priors module; generation arithmetic is unchanged.

The shared local/Colab explorer renders these records in its generation-graph
panel. Select a node with a mouse or keyboard to inspect its formula, parameters,
an input matrix and the output matrix. Expand composition children in place, or
use the mechanism selector to return to the whole task. Edit original row indices
and the first value column before rebuilding to inspect different rows or terms.
Changing those controls clears the old graph. Dataset changes reset the panel.

Smaller graphs use a compact top-to-bottom flow; wide graphs keep a horizontal
layout. Color and labels distinguish inputs, random draws, transformations, signals
and targets. Selecting a node highlights its incoming and outgoing connections.
Fit and zoom controls keep larger diagrams navigable.

Diagrams scroll within the panel and export as SVG/PNG; PNG keeps the diagram's
aspect ratio and caps its longest edge at 4096 pixels. Graph JSON preserves the
returned records. Matrix heatmaps show the first 16 returned matrix rows; tables
show all returned rows and the bounded columns. Source matrices such as weights
and GP normals retain their stated axes, rather than being called observations.
Heatmaps also export as SVG/PNG. Diagram arrows connect actual parent records;
connections across several levels route around intervening boxes.

Conditional GP inspection and controlled pairs remain pending #27 work. Hosted
Colab verification remains **NOT_RUN**; sharing the UI does not establish that
it has been exercised in an actual hosted notebook.
