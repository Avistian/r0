# Retained independent standards reviews

Standards verdict: PASS with one nonblocking heuristic, scoped to checkpoint f75c9cdf298ab4e79a4bab805373984dfa50e7a9 against 19dd757bcbc978bd56d1c6ed76e1e71a7fd824c5. This does not accept all of #28 or certify numerical correctness.

Preflight: /tmp/r0-28; HEAD matched the frozen revision; only untracked .venv. Reviewed all 16 changed files and the 22-commit log. Read AGENTS.md, CODING_STANDARDS.md, README.md, module README, requirements, protocol-review, accepted #28 specification and checkpoint documentation.

Hard documented violations: none found. Explicit RNG streams and public test imports conform to CODING_STANDARDS.md:1–3 and src/r0/modules/README.md. Fixed independent mechanism calibration is permitted by accepted specification decision 12. The checkpoint accurately keeps wider implementation/environment evidence pending. Its focused-test claim identifies source b86ff16, environment, exact retained command and local output; ignored docs/evidence/ storage follows README policy. Author test results were inspected, not independently rerun.

Heuristic only — possible Duplicated Code / Repeated Switches: recipe_sampling.py:313–328 and causal_sampling.py:52–58 maintain the same four causal-profile names. The added hunks repeat `"causal-exogenous-v1": ("forward", "layered", "redirection")`, `if profile in (... "causal-exogenous-v1", ...)`, and `if profile not in (... "causal-exogenous-v1", ...)`. Another profile requires synchronized validation and dispatch edits, risking inconsistent accepted names. A shared internal profile registry could supply both membership checks. Low priority; no current mismatch demonstrated.

Procedure: git diff 19dd757bcbc978bd56d1c6ed76e1e71a7fd824c5 f75c9cdf298ab4e79a4bab805373984dfa50e7a9; git log 19dd757bcbc978bd56d1c6ed76e1e71a7fd824c5..f75c9cdf298ab4e79a4bab805373984dfa50e7a9 --oneline; targeted reads and line-number inspection. No tests, cloud, dependency operations, checkout edits or commits performed. During review author advanced HEAD to a7d24ad852dae449e7d487b5c8cfc163859bf661; verified delta contains only activation-priors.md formatting. Source verdict remains tied to f75c9cd; formatting is tooling-enforced and excluded.


---

Standards delta verdict: PASS at 684a3ecf053986ccf961cae37dde4c3fdb7e8189, scoped exclusively to f75c9cdf298ab4e79a4bab805373984dfa50e7a9..684a3ecf053986ccf961cae37dde4c3fdb7e8189. No hard documented violations or new heuristic findings. The earlier duplicated-profile-registry heuristic remains nonblocking.

Preflight: /tmp/r0-28; HEAD exactly 684a3ecf053986ccf961cae37dde4c3fdb7e8189; only untracked .venv. Reviewed all three commits and three changed files.

causal_stochastic.py:31–50 keeps the numerical repair local to structural-scale evaluation. Zero base scales remove the power path before evaluation. The log calculation applies only to nonfinite positive-base entries, preserving ordinary arithmetic for unaffected entries; final nonfinite scales still raise explicitly. It adds no clipping, retry, resampling, global randomness, I/O, or dependency. This shape is consistent with accepted decisions 3 and 13 and CODING_STANDARDS.md's explicit-randomness rule. It introduces no documented-rule breach or actionable new baseline smell.

test_causal_exogenous.py:251–281 adds three public-interface regressions with explicit seed/task identity and declared expected scales. Those expectations are separate from the runtime log implementation. Tests inspect both saved original-row scale traces and resulting targets, consistent with src/r0/modules/README.md's entry-point requirement. I inspected these tests; I did not execute them or independently establish full numerical correctness.

issue-28-activation-priors.md changes only fenced-code formatting; tooling-enforced formatting is excluded from findings.

Retained inspection procedure: git status --short; git rev-parse HEAD; git log f75c9cd..684a3ec --oneline; git diff f75c9cd 684a3ec; numbered reads of changed source/tests. No checkout edits, commits, tests, dependencies, data access or cloud calls. The separate numerical/spec reviewer owns independent rescue and final-overflow probes; author full checks remain unverified by this review. This verdict does not supersede earlier findings or imply completion of #28.


---

Standards delta verdict: PASS at a47b14e2d22298e54f41926ad9dc83ac3410a842, scoped only to 684a3ecf053986ccf961cae37dde4c3fdb7e8189..a47b14e2d22298e54f41926ad9dc83ac3410a842. No hard documented violations or new actionable heuristic findings. The earlier duplicated-profile-registry heuristic remains nonblocking.

Preflight: /tmp/r0-28; HEAD matched a47b14e. Author-owned working changes comprised checkpoint documentation and untracked evidence files, plus the existing .venv; none was modified or included in this committed-source verdict. Reviewed all three commits and both changed source/test files.

causal_stochastic.py:31–58 detects subnormal/zero power and strength-product intermediates before adding the offset, then recomputes affected positive-base entries in logs. Zero base paths remain excluded. Finite ordinary paths retain their arithmetic order; final nonfinite scales still raise. No clipping, retry, global RNG, new dependency or I/O was introduced. This numerical-code shape conforms to the applicable explicit-randomness/module rules and accepted extreme-draw preservation scope. Repeated mask expressions check distinct evaluation stages and do not warrant an extraction finding.

test_causal_exogenous.py:251–292 retains independent declared scales and public-interface generation/inspection. The scale assertion uses math.isclose(..., rel_tol=2e-13) with its default abs_tol=0: zero cannot pass against expected 1e-200 or 1e-100. Output expectations still use expected_scale, not the computed scale. The output absolute allowance is proportional to expected_scale*abs(raw) for tiny innovations, so it does not introduce a large constant tolerance that could hide their loss. Tanh's 1-Lipschitz property justifies propagating scale error before cancellation; the min(1,...) cap is stricter than that bound for large innovations, not a relaxation. The separate scale assertion remains the decisive underflow oracle.

Procedure retained: git status --short; git rev-parse HEAD; git log 684a3ec..a47b14e --oneline; git diff 684a3ec a47b14e; numbered source/test inspection. No tests or numerical probes were run. Author's reported 15 passes were not independently verified. This delta verdict neither supersedes historical numerical failures nor certifies full numerical correctness or completion of #28.
