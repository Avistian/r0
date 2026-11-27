# Issue 28: authored frozen-function graphs (in progress)

This is an implementation slice of the accepted [issue specification](specs/0028-causal-network-generators.md), not completion of issue 28. The previous published joint registry/root/function checkpoints remain separate evidence.

`FunctionalCausalRecipe` adds `functional-causal-v1` to `generate_task`, recipe JSON and original-row inspection. Nodes contain a real frozen function, ordered parents, an optional parentless root base, an exogenous law, noise response and optional node pipeline. A root's base width need not equal its output width. Nonroot input width must equal the concatenated parent state widths.

All eight previously materialized function families can be nodes: linear, quadratic, oblivious tree, RFF, discretization, EM, product and MLP. The four joint root bases are uniform, Gaussian, ball and covariance; these do not exhaust issue 28's root scope. This entry point accepts authored frozen records. It does not attest how a caller obtained calibration constants, and does not yet attach sampled graph plans, operation ownership or raw-draw manifests.

## Evaluation contract

For each node, in topological order:

1. Evaluate its frozen function on the root base or concatenated parent states.
2. Compute structural scale from that function output: `noise_scale * (offset + strength * abs(output)**power)`.
3. Apply pipeline stages, channel importance, frozen L2 scaling and converters.
4. Add scaled structural innovations, then apply node importance.
5. Propagate final state to descendants. Captured converter channels are read before innovation and node importance; other readouts use final state.

Generation and inspection use one evaluator and the `functional-causal-v1` keyed stream namespace. Observation noise/label draws and splitting remain shared task adapters. Inspection reproduces full original random draws before selecting original row indices, including repeated and out-of-order indices. No function is refitted during emission or inspection.

Node functions and structural noise have finite-output checks. Invalid values raise without replacement draws or silent clipping. Constant and disconnected valid mechanisms remain present with diagnostic flags. Node-level constant flags describe emitted samples, not certified population constancy. Internal MLP and pipeline activation diagnostics use the existing boundary thresholds and remain separate from acceptance. The shared stable signal-noise scaling implementation preserves the affine evaluator's arithmetic.

## Analysis and admission

Channel certificates conservatively track potential root/innovation ancestry, bounded support and positive absolute-moment orders. They use exact nonzero linear paths, quadratic monomials, tree split features, RFF frequency paths and function composition. Some dependencies deliberately over-approximate actual dependence; samples never erase a path. Products with shared inputs use Holder bounds, not an independence assumption. Pipeline converter captures have separate certificates from descendant state.

Every frozen normalization stage, including MLP internal activation normalization, requires certified input second moments; calibrated activation output normalization also requires certified output second moments. Finite calibration values do not prove population moments. Exponential functions on bounded inputs retain bounded certificates. General unbounded exponential integrability is not solved here: unknown remains unknown. There is no automatic change to a finite-fit normalization policy. Target normalization and fresh ancestral graph fitting remain unfinished.

Aggregate limits count actual graph nodes, all retained numeric metadata, declared function dimensions, root/function/pipeline workspaces and conservative Python ancestry-set storage. Estimates are not measured RSS or OS-enforced caps. JSON decoding checks raw metadata before typed construction; payload limits can tighten default caller caps. Malformed container shapes and unknown fields are rejected. Full graph admission replaces the standalone function evaluator's virtual dimension-node accounting.

## Checks and remaining work

Public tests in `tests/priors/test_functional_causal.py` include literal outputs for all eight families together, a dimension-changing tree/quadratic chain, noise/pipeline ordering, heavy-tail normalization rejection, shared-input cubic moment bounds, strict activation JSON shapes, single-node decoding and aggregate structural/resource contracts.

`tests/explorer/test_functional_session.py` exercises the shared session's create/save/open/replay and selected-row inspection for four root bases across regression, binary and multiclass tasks. This is session-level evidence, not a browser or hosted Colab demonstration.

The original source checkpoint passed `make check`: 1200 passed, 1 skipped in 525.48 seconds, with all static gates. Its independent SPEC REJECT below remains valid; the green suite did not cover that defect. The repaired source still needs its own full check and independent review. Retained development logs are `/tmp/r0-28-graph-*.log`; they include both genuine failing cases and fixture/type/lint corrections. No GPU, throughput, RSS, browser or Colab claims are made.

Next: finish calibration and sampled graph ownership integration; extend trace detail and workbench views; retain independent verification and resource evidence; then continue all remaining paper options, typed variables, interventions and delivery requirements. The full accepted specification remains active.

## Review history

Source checkpoint `7993d9c787d73192adf8b7332dc2965967371aca` received an independent **SPEC REJECT**: a constant internal function remained present but had no diagnostic flag. Retained reproducer, exact commands, raw outputs, recipe/card/trace and source pins are at `/tmp/r0-28-functional-graph-spec-7993d9c`; these remain historical rejection evidence. Other independent probes were not run after that first defect.

The standards review reported one documentation violation (missing executable checksum procedure) and one optional decoder-duplication heuristic. The [checksum procedure](research/0028-joint-graph-source-check.md) now records the command and exact output; duplication is unchanged pending a proportionate future refactor.

The repair adds constant function/state sample diagnostics, forwards internal MLP and pipeline activation diagnostics, emits pipeline inspection stages and separates captured readout dependencies from later structural innovation. It also places observations after their producing graph readouts so the browser renderer can consume the returned topological order. A failing public trace test demonstrated this ordering defect in both the new version and existing `causal-v1`/`causal-v2`; the repair covers all three. This is trace-structure evidence, not an actual browser demonstration. Fresh review and the repair's full repository check remain pending.
