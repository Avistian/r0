# r0 requirements

Captured 2026-11-14. Refine after the research phase.

## Goal

Get as close as possible to TabPFN's predictive performance. TabPFN is the reference baseline for every evaluation.

## Tasks the model must handle

- Regression
- Binary classification
- Multi-class classification

## Pretraining data

- r0 pretrains on synthetic data.
- The generator design (priors, task mix, table shapes) comes from the research phase. Do not pick it before then.

## Prior art to reuse

- curie-llm (Piotr's repo, ~/Projects/curie-llm) has a pretrained transformer. Check it for reusable parts (model code, training loop, embeddings) before writing new ones.

## Open questions

- Which TabPFN version and which benchmark suite to compare against.
- Which curie-llm copy is canonical (there are several under ~/Projects) and what its licence allows.
- Output heads: one shared head, or one per task type.
- Maximum number of classes for multi-class.
