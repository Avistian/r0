# Curie transfer review

Pinned [331e6108a97c89e3144d8d482f9c49ea113144da](https://github.com/Avistian/curie-llm/tree/331e6108a97c89e3144d8d482f9c49ea113144da). Unrelated edits and untracked alignment work in its developer checkout are excluded. Findings use committed source.

| Source | Transferable method | r0 difference and owner | Review evidence |
| --- | --- | --- | --- |
| src/curie/model/model.py, Embedding, RMSNorm, SwiGLU, attention | Inspectable transformer components | Numeric/categorical features and context labels replace tokens; #31/#35 choose representation | Trace an encoded context/query through masking to decoded outputs |
| model.py, RotaryPositionEmbedding, TransformerLM | Inspect positional assumptions | Text ordering does not settle tabular row ordering; #35 | Inspect prediction changes under context permutations |
| optimizer.py and nn_utils.py | AdamW, scheduling, clipping, stable losses | Episodic query loss and explicit task accounting; #32 | Inspect gradients, updates, learning rates and tiny-batch behavior |
| training.py, save_checkpoint/load_checkpoint | Persist model, optimizer, iteration, config and random states; path destinations use temporary-file replacement | Global Python/NumPy/Torch states do not capture r0's explicit generators and next task identity; #32/#33 | Compare interrupted/uninterrupted next-task IDs and predictions |
| docs/PLAN_TRANSFORMER.md, experiments/validation | Short integration run, small models, learning-rate sweeps and profiling | Replace text samples with task cards/oracle predictions; #32/#35/#37 | Retain traces and compare predicted pilot costs with observations |

Current permitted reuse is independent implementation. Curie's numerical behavior and historical training claims were not rerun.

## Asset provenance

Curie's committed LICENSE declares MIT with notice conditions. r0 metadata declares Apache-2.0. These are repository declarations; r0's no-copy instruction remains binding.

Checkpoint and dataset rights require asset-specific provenance. Curie checkpoints and source datasets were not selected or verified for reuse. #25 owns benchmark and TabPFN asset selection, #26 provisioning, and #41 relational dataset selection. Repository licensing or hosting on OpenML cannot establish all asset terms.

For the reader walkthrough, inspect training.py fields and restoration calls, then trace #32's requirement for next-task continuity. The explicit-generator convention in r0 requires state beyond curie's global random-state restoration.

The concrete recovery proposal for #32 is to persist each explicit generator's state, generator/config version, the next task identifier or stream cursor, and the last consumed batch. #33 decides whether worker assignment, queued tasks and shard offsets must also be retained or regenerated deterministically. Compare the first several task identities and optimizer updates immediately after resume. These are requirements to resolve at those seams, with the storage representation left open.
