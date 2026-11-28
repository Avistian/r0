# Source/spec axis at 7eea0bd5ab6f553aa07f014ed5a353b4d0db69e9

Reviewed diff: `git diff 9871fdaf7037c7ed9b09926f5073f63f5cf6ff31...7eea0bd5ab6f553aa07f014ed5a353b4d0db69e9`.
Local scoped authority: `docs/issue-28-matrix-families.md`; broader authority `docs/specs/0028-causal-network-generators.md` stays open.

Primary sources read live with web open/find on 2026-11-27:
- https://raw.githubusercontent.com/soda-inria/tabicl/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_matrix.py
- https://raw.githubusercontent.com/soda-inria/tabicl/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_points.py
- https://arxiv.org/html/2602.11139v1 (E.10, E.12)

Source distinctions agree with scope: the paper uses covariance points for kernels; pinned matrix code imports mixture points, and the covariance helper supports uniform or Gaussian innovations. Matrix source has five top families, four nonrecursive child families, flattened activation, independent child/top jitter and row normalization; Gaussian factors are not orthogonal singular-vector matrices. Uniform/polynomial choices, independent matrix selections, dimension-three points, range laws, activation choice law and omission of the correlated GlobalSampler are disclosed local exploration decisions, not an exact source joint-prior reproduction.

Read-only structural inspection confirms plans call geometry/parameter admission before new matrix arrays are sampled, and constructors admit recipe parameters before materializing stored constructions. Raw factor/point/activation stages independently require finiteness. The oracle additionally probes this ordering and independently counts numeric record leaves, including permutations, assignments, duplicated geometry and stage scalars, excluding arithmetic limits.

This review does not accept the entire issue, downstream scientific claims, browser/Colab/GPU behavior or resource RSS measurements. Backend saved-session execution is separately retained by the runner.
