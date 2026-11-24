# Verification record

Base: 107c5051084a0a918db16b18874981d430bc1828 (fetched origin/main, 2026-11-24).
Isolated authoring checkout: /tmp/r0-25. Original checkout edits were preserved.
Environment reuse was explicit: UV_PROJECT_ENVIRONMENT=/tmp/r0-24/.venv, with the repository's
no-sync verification wrapper selecting /tmp/r0-25/src/r0. Preflight and tracked-file guards passed.

## Packet and arithmetic checks

A one-off Python standard-library calculation parsed comparison.json and compared calculated
primary scores against the independent literal/radical values in README.md. It checked every
context/query ID intersection, probability-vector dimensions/ranges/sums, each draw's three
primary scores, mean gap and the simple-baseline comparison. This is fixture verification,
not a runtime integration test, model execution or an independent scientific evaluation.

```text
Candidate identity: f68a211d5491a204c9f0556cb07dd55806a33a976e678c58cd4168f39348305b
PASS: spec binding, formatting stability, changed-split identity sensitivity, six roles, pending assets
PASS binary: candidate mean=0.1966666667, gap mean=+0.0716666667
PASS multiclass: candidate mean=0.2383333333, gap mean=+0.0983333333
PASS regression: candidate mean=0.8047378541, gap mean=-0.2798541044
Authored arithmetic only; models NOT_RUN; human freeze PENDING
```

The identity check reparsed differently formatted JSON and recovered the same digest; changing
query_seed changed it. All six role combinations are present, all admissions remain PENDING,
and no unverified asset hash is fabricated. The README identity command is reproducible locally.
No failure-injection outcomes in the walkthrough are claimed as executed harness tests.
