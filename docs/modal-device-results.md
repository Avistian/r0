# Modal generator device results

The [guarded T4 run](https://github.com/Avistian/r0/actions/runs/45697210266)
completed on 2026-11-25 at source
`c871ae799b1d0a0c592043368549d31e4d32fb32`. It used
`configs/generation/sampled-gpu.toml` and the
[bounded device comparison](modal-device-agreement.md).

| Stage | Observed result |
| --- | --- |
| Actions admission and CPU smoke | Both passed |
| CPU/CUDA agreement | 93 planned, 93 passed, zero failed or not run |
| Comparison scope | Three requested polynomial diagnostics plus the fixed 90-case matrix: ten families × three sizes × three target types |
| Candidate hardware | Tesla T4; CUDA 13.2; PyTorch 2.15.0+cu132; NumPy 2.5.5; Python 3.12.6 |
| Numerical agreement | Largest recorded absolute difference `5.129230373768223e-14`; required `rtol=atol=1e-10` |
| Comparison resources | Sum of per-case elapsed times 140.956 seconds; cumulative process peak RSS 4,271,460,352 bytes; PyTorch peak GPU allocation 125,831,680 bytes |
| Sampled generation | Eight attempted, eight completed/generated, zero failed/pending; `broad-v1` polynomial, 128 rows and eight features |
| Durable resume | Separate single-use container: eight completed, zero attempted/generated replacements, zero failed/pending |
| Reservation | $0.80; previous two smoke reservations $2.50; aggregate reserved $3.30 against the user's $5 authorization |
| Actual billing | `NOT_MEASURED`; reservation arithmetic is not provider billing |

The comparison's RSS was close to the 4-GiB function memory limit; this result does
not establish substantial memory headroom. Per-case elapsed time includes both
CPU and CUDA generation plus comparison, and excludes initial process import and
provider startup. GPU allocation is PyTorch's measurement, not total device memory.
No throughput claim for the sampled profile follows from eight tasks.

The fixed matrix includes 20,000-row, 256-feature analytic tasks and 100-class
targets; exact GP stays at 1,024 rows. It uses fixed representative settings and
Gaussian inputs. It does not qualify all sampled mechanisms, non-Gaussian settings,
complexity maxima or a production mixture. The earlier sampled CPU timeouts remain
failures; this GPU run neither retries nor replaces them. Full foundation-model
training quality and actual hosted Colab use remain unverified by this run.

## Retained receipts and reproduction

The Actions artifact `generation-45697210266` contains `reservation.json`,
`local-smoke.json` and `result.json`. Raw copies remain ignored under
`docs/evidence/issue-27-modal-agreement/hosted/`.

- Result SHA256: `ea451bf09343ee204fab4994666aea7cc957bf51361167a653d6f6bf492f77a5`
- Reservation SHA256: `019b992b0b260df4b6645d2adaf40331ff62b443ae77e399604988ec7f452cd1`
- Comparison driver SHA256: `aa8d840d28ea1f6c1b7857cbd0ae003539ddc1e8ad963f395e9be972a1af57f6`
- Comparison plan SHA256: `f34a391e25ed6836dea87f99f4cc21f6b10309f9874bfa0d67be2f9304280250`

Download is read-only and starts no compute:

```bash
gh run download 45697210266 --repo Avistian/r0 \
  --name generation-45697210266 --dir /tmp/r0-modal-receipts
sha256sum /tmp/r0-modal-receipts/result.json /tmp/r0-modal-receipts/reservation.json
```

The result embeds the comparison's plan, manifest and 93 started/completed case
records. This reproduces the author-run receipt calculation without generation:

```python
import hashlib
import json
from pathlib import Path

result = json.loads(Path('/tmp/r0-modal-receipts/result.json').read_text())
records = result['generation']['device_agreement']['records']
plan, manifest = records['plan.json'], records['manifest.json']
assert hashlib.sha256(json.dumps(plan).encode()).hexdigest() == manifest['plan_sha256']
cases = [records[f'case-{i}.json'] for i in range(93)]
assert [c['name'] for c in cases] == [c['name'] for c in plan['agreement_cases']]
assert all(c['status'] == 'PASS' for c in cases)
print(records['summary.json'])
print(sum(c['elapsed_seconds'] for c in cases))
print(max(v for c in cases for v in c['max_absolute_error'].values()))
print(result['generation']['summary'], result['resume']['summary'])
```

Generation and resume were two separate remote invocations using
`single_use_containers=True`. These receipts are hosted execution evidence plus
retained-record consistency checks; they are not an independent mathematical
oracle or an independent second execution. The compute switch was restored to
false and the original daily dispatch/reservation settings were restored after
admission. No automatic retry was submitted.
