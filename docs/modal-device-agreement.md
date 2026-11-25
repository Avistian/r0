# Bounded device comparison on Modal

GPU generation through the guarded `agent-train.yml` workflow first compares
93 named cases: three neutral target-type diagnostics from the requested recipe,
followed by the [fixed 90-case matrix](generator-device-agreement.md). These are
verification cases, not independent training examples. CPU requests retain their
three diagnostic recipes in the plan but do not run CUDA comparison.

Admission records `agreement_cases` and `agreement_seconds` before the credential
step. GPU comparison has a 180-second subprocess deadline; requested generation
keeps its own configured deadline. The Modal function timeout includes generation,
comparison and a 60-second margin. Startup remains limited to 120 seconds. The
admission check conservatively budgets both generation and independent-container
resume at this expanded maximum, including the comparison allowance for both.

For a 60-second GPU request the workflow must allow at least 840 seconds:
`2 * (60 generation + 180 comparison + 60 margin + 120 startup)`. At the published
T4, one-core and 4-GiB rates, the function estimate is
`840 * (0.000164 + 0.0000131 + 4 * 0.00000222) = $0.1562232`.
Admission requires twice that estimate plus a $0.25 build allowance, so a
reservation must be at least $0.5624464. Rates were checked against
[Modal pricing](https://modal.com/pricing) on 2026-11-25. This is a reservation
calculation; actual provider billing remains `NOT_MEASURED` until reconciled.
Image builds and uncertain platform failures are not made cost-free by a function
timeout. This does not authorize increasing the user's aggregate budget.

The image carries the same comparison command used locally and sets CPU numerical
libraries to one thread. Comparison runs in a child process with an explicit CUDA
candidate. It retains the plan, source and driver hashes, environment, per-case
outcomes and logs under the run's receipt directory. Successful or failed command
records are copied into the Actions result artifact and committed to the Modal
Volume before return. There are no intermediate Volume commits during comparison;
a provider termination before the final commit can leave only the earlier durable
submission claim. Such an invocation has an uncertain outcome and cannot retry
automatically.

Comparison failure, unavailable CUDA or timeout skips requested generation and
resume. A subprocess timeout retains completed records and any started case;
remaining cases are not accepted. Only a successful comparison proceeds to the
shared guarded generator, then a separate container reopens the saved run and
must generate zero replacement tasks. Modal retries remain zero, maximum container
count one, and containers are single-use. Provider memory remains 4 GiB and GPU
selection remains T4.

`configs/generation/sampled-gpu.toml` selects eight small polynomial tasks from
`broad-v1`, with 60 seconds and 10 MiB of generated output. The comparison's fixed
large shapes are separate from these training-output caps and have their own
180-second limit. No full-size comparison arrays are retained. Dispatch only
through the guarded workflow, after local checks and within explicitly authorized
remaining reservations. A successful CPU preflight is not proof of CUDA agreement.
