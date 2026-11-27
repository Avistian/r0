# Independent SPEC review: REJECT

Source: 11102283ec79503af75a3b16b133911e7fa686be.
Base: 51684d98b719494c2a20d1a07e7a0470c6c3b4c3.
Commits: 906f99b complete operation ownership; 1110228 jointly staged streams.
Read-only source preflight and final status: only intentional untracked .venv.

P2: New registry decoder accepts malformed decision-ledger arrays.
At src/r0/modules/priors/lib/joint_registry_decoding.py:60-65 it delegates the
ledger without checking JSON shapes. Public reproduction creates an event-free
committed operation, serializes its valid registry, and changes only decisions.events.
An empty object {} and empty string "" both decode to the exact valid record;
[0] instead leaks AttributeError. These are three manifestations of one missing
boundary-validation gate. Validate nested decision/preference records before delegation;
merely catching AttributeError would not fix malformed acceptance.

Research contract docs/research/0028-joint-root-function-plans.md:159 requires
"Derive exact expected events/catalogs/values/shapes on decode"; lines184-186 require
bounding raw containers before conversion and rejecting malformed object arrays.
This is in scope for the new caller-bounded registry decoder even though the
permissive delegation target existed before these commits.

Valid control and all three malformed probes executed twice on the exact pinned
source/import path. Program uses explicit exceptions for gates (also effective with
python -O), only public prior APIs, fresh output directories, and source SHA pins.
Ruff format and check passed before execution. Both executions retained REJECT;
run-001 shell wrapper displayed the log after failure, so its shell exit was zero;
run-002.exit-code records the actual gate exit1. No failed lint attempts.

Stopped at first demonstrated defect, as requested. Further adversarial registry,
stream/engine matrix and isolated-process legacy-byte checks NOT_RUN in this review.
Static inspection is not executable acceptance. Author tests and parent makecheck
are separate evidence. No full issue28 or aggregate graph admission, raw manifest,
typed plan tree, ancestry calibration, concurrency or seed-independence claim.
