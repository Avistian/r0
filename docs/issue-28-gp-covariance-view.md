# Retained GP population matrix inspection

The explorer generation graph includes a separate matrix panel for finite GP
feature columns. Select a shared or per-column kernel and inspect retained
covariance K, lower triangular factor L, latent locations, standard normal
innovations Z, and feature values X. Each feature satisfies X[:,j] = L @ Z[:,j]
for its kernel. The view never resamples or conditions a population.

Covariance and factor axes use original population row identities. Locations
use latent-coordinate columns; innovations and values use feature columns.
These axes are independent of selected graph rows, including repeated rows.
Offsets expose every entry in windows of at most 16 by 16. Tooltips show exact
values and tables label the axes. Heatmap scales are local to the window.
Complete matrices and population identity can be downloaded as JSON; existing
chart controls provide SVG/PNG exports.

Constructing another task clears the panel. It appears after successful guarded
graph inspection. The UI uses retained arrays without recomputing covariance.

GP hybrid roots, broader catalogs, measured resource envelopes and hosted Colab
verification remain required. This slice supports original finite GP populations;
edited-environment integration requires its own verification.
