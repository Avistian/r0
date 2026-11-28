# TabICL v1 option audit

Current audit of the 22 `REQUIRED_28` I1 entries in the finite inventory. This is
an implementation audit, not a paper/checkpoint reproduction claim. `Component`
means the listed operation exists; `Gap` identifies remaining source-profile work.
Other inventory groups are not closed by this table.

| ID | Current disposition |
| --- | --- |
| I1.function.01 | Component: paper-exponent finite sine sum, raw parameters, network/pipeline generation, ZIP replay and independent numerical oracle. Source standardization composition remains a gap. |
| I1.function.02 | Gap: exact source-profile standardization and independent calibration adaptation. Generic standardization exists but uses different variance/epsilon conventions. |
| I1.function.03 | Gap: connect the source lognormal rescaling law to the complete activation profile. Generic numerical scales are not evidence of this law. |
| I1.function.04 | Gap: connect the Gaussian shift law to that profile, retaining the sampled shift. |
| I1.function.05 | Component: homogeneous activation preferences. Gap: source selector must share the activation type while sampling each layer's random function parameters separately. |
| I1.function.06 | Component: independent activation preferences. Gap: complete source selector integration. |
| I1.function.07 | Components: actual XGBoost vector fitting, retained depth/count sampling, independent fitted graph archives; actual-parent population fitting is a separate lifecycle. These do not establish the full source architecture/default prior. |
| I1.function.08 | Gap: connect tenfold Fourier mass to the complete source selector. Generic configurable weights alone do not prove the source recipe. |
| I1.code.01 | Component: pinned-code exponent law as a separate finite sine activation. Standardization/profile composition remains a gap. |
| I1.code.02 | Component: sine with configurable scale/phase and log-uniform parameter laws. Gap: source random-frequency factory with its standardization lifecycle. |
| I1.code.03 | Component: `heaviside_positive`, including equality at zero; existing primitive tests. |
| I1.code.04 | Gap: complete shared shift/scale source profile. |
| I1.code.05 | Gap: complete per-channel shift/scale source profile. |
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
