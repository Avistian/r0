# Issue 22 reference cards

These observations are authored examples, not random samples. No learner is trained and no generator-quality claim follows from these checks. Reference parameters are declared prior knowledge, never sampled latent truth. Targets are supplied separately for query-only scoring. Coordinates and targets are unitless in the GP cards.

Run the complete CPU walkthrough from the checkout:

```sh
uv run python scripts/oracle_walkthrough.py --out /tmp/r0-oracle-evidence
```

Run an individual card through the CLI:

```sh
uv run r0 oracle --config configs/oracles/coin-four.toml \
  --expected tests/fixtures/oracles/coin-four.json \
  --out /tmp/coin-four.json
```

Without `--expected`, prediction/scoring still run, but verification is explicitly NOT_RUN. With a card, each numeric comparison retains expected/observed values, absolute error, relative error (undefined at expected zero), allowed error and pass/fail. Unrepresentable numeric differences are null with an explicit range reason; exact rational comparison of the supplied floating-point values preserves the verdict even when a displayed difference overflows. A mismatching card produces a retained FAIL report and a nonzero CLI exit. Output paths do not enter identities. Existing different run records cannot be overwritten. The optional `--expected` argument supplies the independent answers required by the agreed walkthrough, without mixing them into predictor inputs.

## Classification arithmetic

All query rows share one unknown class-probability vector, so feature values are uninformative and probabilities are constant within an episode. For concentration vector a and context counts n, posterior concentration is a+n and the next-label probability is (a_i+n_i)/sum(a+n).

| Card | Independent calculation |
| --- | --- |
| coin-prior | Beta(1,1) gives heads probability 1/2. |
| coin-four | Three H, one T: Beta(4,2), heads 4/6. Query H has Brier 2/9; query T has Brier 8/9; mean 5/9. Mean log loss is -log(2/9)/2. |
| coin-next | Append one observed H: Beta(5,2), heads 5/7. |
| coin-asymmetric | Start Beta(2,1), observe H,H,H,T: Beta(5,2), heads 5/7. |
| coin-undefined-auc | Same posterior, query H,H: Brier 2/9. AUC undefined because there is no negative query. |
| multiclass | Prior (1,2,3), context A,A: posterior (3,2,3), probabilities (3/8,1/4,3/8). C is absent from context yet retains probability 3/8. With queries A,B,C, Brier mean = sum(p²)+1/3 = 11/32+1/3 = 65/96. |
| multiclass-undefined-auc | Same probabilities, queries A,C. B-vs-rest AUC is undefined; full macro AUC is undefined, while valid A and C components remain visible. |

AUC is the fraction of positive-negative pairs correctly ranked, with half credit for ties. All query probabilities here tie. For H,T, there is one pair, receiving 1/2 credit: AUC=0.5. In the three-class card, each one-vs-rest component has two tied pairs and AUC=0.5, so macro AUC=0.5. Query-only AUC is computed per episode, never pooled across tasks. It checks reporting, not ranking ability for these exchangeable priors.

## Gaussian-process arithmetic

Let A = K(context,context) + noise_variance I + explicit_jitter I and b = K(context,query). Posterior mean is m + bᵀ A⁻¹(y-m); latent covariance is K(query,query)-bᵀ A⁻¹b. These equations explain the independent answers; the implementation uses Cholesky solves, not inversion. New-observation covariance adds only modeled noise_variance I. Jitter changes conditioning only.

| Card | Independent calculation |
| --- | --- |
| gp-linear | k(x,x')=xx'; context x=1,y=2; noise=1; query x=2. A=2, b=2, prior query variance=4. Mean=2·2/2=2; latent variance=4-4/2=2; observation variance=3. Query y=3 gives MSE=1 and Gaussian NLL=(log(6π)+1/3)/2. |
| gp-mean | Same setup with m=1 and context y=3. Mean=1+2·(3-1)/2=3; latent variance remains 2. |
| gp-prior | RBF signal variance=1, length scale=1, no observations. Mean=0, latent diagonal=1, covariance(0,1)=exp(-1/2)=0.6065306597126334. Noise=1 makes observation diagonal=2, leaving off-diagonal unchanged. |
| gp-near-far | One context x=0,y=2, noise=1, so A=2. At query q, let r=exp(-q²/2). Mean=r, latent variance=1-r²/2, observation variance=2-r²/2. At q=0: (1,1/2,3/2). At q=8: r=exp(-32)=1.2664165549094176e-14; variances round to 1 and 2 in float64. |
| gp-duplicates | RBF context x=(0,0), y=(1,3), noise=1; query x=0. A=[[2,1],[1,2]], inverse=(1/3)[[2,-1],[-1,2]]. Weights=(-1/3,5/3), mean=4/3, latent variance=1-2/3=1/3, observation variance=4/3. |
| gp-noiseless | Linear context x=1,y=2, noise=0. The slope is exactly 2. Queries x=(1,2) have means (2,4), zero posterior covariance and zero observation variance. Density scoring is undefined for this degenerate distribution, not epsilon-clipped. |
| gp-jitter | RBF context x=0,y=2, noise=0, explicit jitter=1; query x=0. A=2 gives mean=1 and latent variance=1/2. Observation variance stays 1/2. This is explicitly a stabilized reference, not the original noiseless posterior. |

The JSON answer cards were authored from these derivations independently of oracle output. Acceptance uses absolute error <= 1e-10 + 1e-10*abs(expected). Numerical covariance correction is separately bounded by 1e-12*max(1, largest prior query variance); raw covariance and every correction are retained. An eigenvalue outside that bound fails; eigenvalues are not repaired.

## Additional walkthrough evidence

The script verifies query-target changes leave predictor inputs/predictions unchanged, context/query reordering and query extension preserve corresponding marginals within tolerance, repeated runs retain identical semantic identities, and explicit errors occur for invalid configuration, unsupported inputs, singular noiseless systems and oversized cases. It records elapsed CPU-walkthrough wall time as observed evidence only. The reference command's 64/64/16 limits do not constrain future generators.
