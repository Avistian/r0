# Issue 28: replayable fresh ancestor populations

`sample_functional_prefix(prefix, rng=..., owner=(...), capture=...)` samples an
admitted frozen prefix from one injected calibration generator. Roots and structural
innovations consume it in node, column and draw order. Every receiving fit can call
it afresh; successful calls advance the stream, while any failure discards the whole
population's draws and records. Caller mappings and RNG states retain the raw ledger's
transaction contract. The existing keyed `evaluate_functional_prefix` and authored
generation/inspection APIs keep their behavior.

Capture records original root base values and ball radii; Gaussian, uniform and
Student-t innovations; mixture normals and component choices; and original directions
and radii for ball innovations before covariance factors. Truncation records every
proposal batch, including rejected proposals and unused accepted tails. It does not
mislabel post-truncation values as unconstrained draws. Laws, parameters, shapes and
states use the existing seven-law raw recorder. No retry policy changes.

Owners extend the caller's receiving-fit path with node/root/innovation/column
components. These paths identify draw sites; complete typed graph ownership and
plan-to-runtime bindings remain separate unfinished work. A nested graph transaction
must bind this generator as `calibration`. The prefix is readmitted before allocation;
raw-record limits additionally bound retained draws. Complete graph-wide aggregate
admission remains unfinished. Sampling on CUDA is explicit, without fallback;
GPU execution and measured resource use remain unverified.

Author checks include missing-API RED, five-engine capture/replay across all four
independent innovation laws with/without truncation and ball geometry, covariance
factors, original root ball radii, ordered parent states and literal mixture equations.
Successive calls produce distinct populations; late draw-budget and nonfinite pipeline
failures roll back the generator and ledger. Original keyed evaluation checks remain.
The initial expanded command named two nonexistent test files and ran no tests; its
log is retained separately from the corrected successful command. Full-suite and
independent acceptance are pending at this source commit.

This progresses [the full #28 specification](specs/0028-causal-network-generators.md)
and [sampled graph ownership design](research/0028-sampled-graph-ownership.md).
The graph builder still must request a fresh population for each receiving function
and pipeline fit, retain its receipts and bind it to frozen ancestors. This component
alone does not prove that graph schedule, browser/Colab delivery, broader paper options
or scientific performance.
