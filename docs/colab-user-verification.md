# Colab user verification

On 2026-11-25 the user ran the prepared notebook in their Colab runtime and
reported the behavior below. This is **user-run hosted verification**, separate
from the author's local notebook/browser checks and automated Modal execution.
The author had no tool connected to the Colab kernel or browser and did not
independently inspect that runtime.

The supplied notebook link pinned repository revision
`cd57d10390338c7db261023aca6c86d4e38393a0`, whose installation cell pins generator
revision `24929292b8e788392244ae5544e3fbf73e7d5765`. Instructions kept
`DEVICE="cpu"`, `USE_DRIVE=True` and `RUN_LARGER=False`. These identify the supplied
procedure, not independently observed runtime metadata.

| Hosted check | User report |
| --- | --- |
| Generation | “generation works in colab” |
| Generation graph and dataset ZIP plus SVG/PNG download | “Graph and downloads work” |
| Runtime restart, rerun with the same settings and Drive folder | “its ok”, in response to a check expecting completed=8 and generated=0 |
| Multiclass probabilities and class counts | “All three work”, covering this and the next two checks |
| Saved noise-change comparison | Confirmed in the same response |
| Exact GP conditional view with additional draws | Confirmed in the same response |

The restart answer confirms the requested check at the user-report level. Numeric
output was not supplied; do not present the expected eight-completed/zero-new
counts as an independently inspected Colab receipt. Exact environment/version
output, saved-group checksums and the executed notebook have not yet been supplied
to the author. Retain that output to complete the hosted evidence record. No
additional paid run or larger Colab generation was requested by these checks.

Question/answer summaries remain ignored under
`docs/evidence/issue-27-colab-user-acceptance/`. The notebook itself prints source,
actual hardware/software, bounded-run measurements and verified archive receipts;
its outputs should be saved after execution. Its shared UI retains task identity,
recipe and diagnostic truth in exported replay bundles.

These reports support the stated hosted interactions. They do not prove arbitrary
profile capacity, full-scale throughput, frontier-model performance or independent
CPU/GPU agreement. The separate guarded T4 result supplies the measured GPU checks.
