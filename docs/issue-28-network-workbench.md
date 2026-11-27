# Standalone network explorer and notebook workbench

The shared local explorer and notebook callback expose standalone frozen networks
separately from causal graphs and neural parent reducers. Select a **Standalone
network** preset under **Construct a graph or standalone network**. Seven sampled
families cover dense, individual sparse, block sparse, skip, residual, irregular
and gated architectures. Two additional presets expose P1 individual/block source
laws; an authored combination demonstrates sparse experts, input-dependent softmax
gates and projected skips together. This is deliberately a broad option surface,
not one preferred architecture or a claim that every combination reproduces a paper.

Construction JSON exposes widths, family and activation pools, homogeneous versus
per-block choices, activation placement, output activation, weight distribution and
scale, fan-in scaling, bias scale, dropout, source mask profile, source dropout
profile, block limit, expert count and skip probability. The named P1 profiles retain
the core source scaling/bias laws; the individual profile exempts its first block
from dropout. Individual masks may
use the separate `p1-beta` dropout law. See the [core source-law description](issue-28-structural-networks.md).
Fixed structural dropout masks are sampled once and remain fixed across rows.
Zero/disconnected draws remain valid and are displayed without resampling.

Construction runs inside the existing guarded generation worker. The construction
seed controls the named/raw sampling streams; dataset seed and task ID control the
emitted rows. Caller caps apply to replay/admission. Construction details expose the
request, decisions, raw draw count and stream roles; the archive retains complete
raw records and snapshots. Next task passes the raw record unchanged and updates
only exact decimal identity strings, including integers above JavaScript's safe
integer range. Reopening and replay preserve the original record.

**Inspect architecture and prepare edit** opens a dedicated architecture diagram.
It shows ordered input paths, projected skip paths, block widths, experts and gates;
selection highlights ancestors. The selected block exposes mask/zero counts,
activation placement and editable equations. The complete authored NetworkRecipe
JSON permits broader combinations and root/task edits under the public validator.
Applying a block first validates the candidate inside the timeout-protected worker.
Rejected block text remains available when switching blocks; the accepted diagram
and complete recipe stay unchanged. Unapplied complete-recipe edits are not
overwritten by block Apply. Validation is read-only and does not save a dataset.
Edits produce linked before/after archives and comparison exports, retaining the
original sampled provenance in the original archive. They do not claim to be causal
interventions. The separate generation trace retains original row indices and
shows affine/activation, gate, skip and state values. Architecture and generation
figures have SVG/PNG export; the generation trace also has JSON export.

Response curves evaluate the frozen network on observed input coordinates; the
root transform is already present in those coordinates and is not applied again.
Curves expose raw latent output channels, not class probabilities, labels,
separation/offset adjustments or observation noise. Classification uses the
existing score-channel selector. Query grids, decoding and replay in the session
run inside the guarded worker under caller time/byte/geometry caps.

Local session and actual Chromium checks cover construction, exact identity changes,
original-row inspection, replay, export, linked edits, reopening and invalid-input
recovery. Timeout checks cover construction and edit baseline preservation. A mocked
local Colab callback exercises the same construction/edit/export/reopen transport
and page, without invoking hosted services. **Hosted Colab: NOT_RUN.** These are
bounded author checks; combined integration and independent review are separate.
GPU agreement, scientific comparisons and the remaining #28 feature breadth remain
required. This work does not close those requirements.

Task-only comparison controls expose regression noise or classification separation,
preserving the frozen network and input coordinates. Feature-extension and active-
feature/sparsity task comparisons remain unsupported required breadth; the UI does
not advertise them as working. The network editor provides explicit architecture
changes as a distinct linked edit, not an equivalent task-only comparison.
