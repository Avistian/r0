# Source inventory

- r0-research-implementation (2).html, SHA-256 `99dd8bd3445bfc667dc98a99cce771fd42a84bb651ff92938a33347245953bd3`.
- synthetic-priors-deep-dive.html, SHA-256 `e8606a6e5d0bf7e2cc968910e9b256f445ac7cef429cc9feedd8a203c727a192`.

Original text is retained in [T1–T10](original-tickets-1.md) and [T11–T20](original-tickets-2.md). These excerpts establish proposed promises, not implemented behavior.

The synthetic-priors guide develops SCM, neural and tree families, mixtures, hyperpriors and filtering. Its proposed merged design excludes a standalone BNN prior. The accepted roadmap keeps that family eligible in #28, assigns filtering/mixture auditing to #30 and contribution studies to #36. This supersedes the fixed merged design.

Repository requirements retain regression, binary and multiclass classification and leave generator design, heads and baseline version open. Current roadmap issues are #19–#42 in Avistian/r0.

Paper results and cost estimates in the guides were not reproduced. The crosswalk records implementation proposals; scientific facts needed for candidate selection must be verified by the owning ticket.
