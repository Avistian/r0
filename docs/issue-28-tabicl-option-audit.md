# TabICL v1 option audit

Current audit of the 22 `REQUIRED_28` I1 entries in the finite inventory. This is
an implementation audit, not a paper/checkpoint reproduction claim. `Component`
means the listed operation exists; `Gap` identifies remaining source-profile work.
Other inventory groups are not closed by this table.

| ID | Current disposition |
| --- | --- |
| I1.function.01 | Component: paper-exponent finite sine sum, raw parameters, network/pipeline generation, ZIP replay and independent numerical oracle. Source-specific outer/inner standardization and selector composition are implemented in the activation selector. |
| I1.function.02 | Component: versioned sample-standard-deviation plus 1e-6 convention with independent frozen calibration, JSON/ZIP replay and numerical oracle. Integrated into the source-weighted activation selector. |
| I1.function.03 | Component: retained exp(2a) scale with standard Gaussian a; independent draw oracle across five engines. Integrated into the source-weighted activation selector. |
| I1.function.04 | Component: retained standard Gaussian b in exp(2a)*(x+b), composed with standardization and Fourier in archived pipelines. Integrated into the source-weighted activation selector. |
| I1.function.05 | Component: source selector shares activation type while drawing fresh Fourier and scaling parameters per layer; verified with retained parameters and composed numerical oracle. |
| I1.function.06 | Component: source selector independently chooses types per layer with half the mass when diversity is enabled; five-engine type-choice oracle. |
| I1.function.07 | Components: actual XGBoost vector fitting, retained depth/count sampling, independent fitted graph archives; actual-parent population fitting is a separate lifecycle. These do not establish the full source architecture/default prior. |
| I1.function.08 | Component: explicit source pool of 18 simple types plus ten Fourier slots; every slot checked and integrated into shared/independent selection. |
| I1.code.01 | Component: pinned-code exponent law as a separate finite sine activation. Integrated with source standardization and weighted selector. |
| I1.code.02 | Component: random-frequency sine factory, source default log-uniform frequency/uniform phase, independent frozen TabICL standardization, configurable frequency bounds, five-engine oracle and generated ZIP replay. |
| I1.code.03 | Component: `heaviside_positive`, including equality at zero; existing primitive tests. |
| I1.code.04 | Component: frozen shared Gaussian shift/lognormal scale factory. Integrated into the source-weighted activation selector. |
| I1.code.05 | Component: frozen per-channel Gaussian shift/lognormal scale factory. Integrated into the source-weighted activation selector. |
| I1.code.06 | Component: actual sklearn DT regressor, random splitter, retained synthetic fits and frozen inference. See fitted-tree contracts. |
| I1.code.07 | Component: actual ExtraTrees regressor and retained synthetic fitting. |
| I1.code.08 | Component: actual random-forest regressor and retained synthetic fitting. |
| I1.code.09 | Component: actual CPU histogram multi-output XGBoost, retained native export and vector inference. Historical unspecified backend defaults remain distinct. |
| I1.code.10 | Named adaptation: explicit actual-batch construction retains and freezes each realization. A new input batch requires a new construction; replay never silently refits. |
| I1.code.11 | Component: named `tabicl-individual` initialization, first matrix unmasked, later p capped at 0.99, corrected sqrt(1-p) scaling or explicit disabled scaling; independent raw-mask/scale checks. |
| I1.code.12 | Component: named `tabicl-block` initialization, source block-count/geometry law and sqrt(realized keep) scaling or explicit disabled scaling; independent geometry/scale checks. |
| I1.code.13 | Gap: a named source-default configuration with noncausal observations. Existing input-view controls and source scalar hyperpriors do not establish the complete tree architecture override. |
| I1.code.14 | Explicit adaptation: actual-population fitting rejects nonfinite input rows, rather than copying source NaN replacement. This preserves the accepted failure-reporting requirement. |

Concrete current contracts:

- [Source-weighted activation selector and pipeline composition](issue-28-tabicl-selector.md).

- [Standardization and shared/per-channel random scaling](issue-28-tabicl-standardization.md).

- [Corrected mask profiles and retained legacy identity](issue-28-tabicl-mask-profiles.md).
- [Fourier activation implementation and limitations](issue-28-tabicl-fourier.md).
- [XGBoost sampled parameters and graph archives](issue-28-xgboost-parameter-laws.md).
- [Actual-population XGBoost lifecycles](issue-28-xgboost-population.md).
- [Sklearn fit parameter laws](issue-28-fitted-parameter-laws.md) and
  [population fitting](issue-28-fitted-population-lifecycles.md).
- [Source distinctions and pinned tree settings](research/0028-fitted-tree-source-laws.md).

The remaining source-profile gaps above stay in #28. UI presentation of these
options is already assigned to #126. Passing component tests does not turn a
`Gap` into a completed option or establish every meaningful composition.
