# CONTEXT.md — r0 vocabulary
- **RelDB**: A collection of related tables with primary keys, foreign keys and optional time columns.
- **Entity table / event table**: a table without / with a time column that grows over time.
- **Seed time**: the moment a prediction is made; rows at or after it are invisible (ADR 0005).
- **Task rows**: (entity id, seed time, label) — RelBench's "training table".
- **Prior**: A distribution over data-generating mechanisms and the datasets they produce.
- **Synthetic task**: A sampled prediction problem with generated observations and targets, using one table or related tables.
- **Label rule**: The mechanism defining a prediction task’s targets.
- **Featurizer**: A transformation from available observations to features used for prediction.
- **Feature row**: The feature values associated with one prediction example.
- **Context set / query set**: Labeled examples available to condition predictions / examples whose targets are to be predicted.
- **RPFN**: A relational prior-fitted network.
- **k (shots)**: The number of labeled context examples supplied for prediction.
- **Dev suite**: Evaluation tasks used to guide model and generator development.
- **Benchmark suite**: Evaluation tasks governed by a fixed protocol for milestone performance claims.
- **Protocol hash**: A content fingerprint identifying an evaluation protocol.
- **Run record / leaderboard / ratchet / budget ledger**: An execution’s recorded evidence / comparable results / a performance acceptance rule / resource and cost records.
- **Smoke config**: A configuration for a small, bounded end-to-end execution.
- **Prior mixture**: A weighted combination of prior components.
- **Tabular dev track**: The single-table evaluation tasks used during development.
- **Bar head**: A regression output distribution represented by probabilities over target intervals.

- **Source task**: The underlying sampled prediction problem from which one or more episodes are drawn.
- **Episode**: One prediction problem consisting of labeled context examples and query examples drawn from a source task.
- **Predictor view**: The observations and declared task information available for making predictions in an episode.

- **Missing value**: A feature value that is unobserved for an example.
- **Unknown category**: An observed category absent from the category mapping fitted on context examples.
- **Class vocabulary**: The declared set of possible classification targets for a task, independent of the targets observed in a particular episode.

- **Episode schema**: The declared feature meanings and target vocabulary shared by the context and queries of an episode.

- **Replay**: Reconstructing an identified episode and its recorded processing under the same declared recipe and environment.

- **Predictive oracle**: A reference that computes a predictive distribution from context observations and explicitly declared prior assumptions. It does not observe query targets or the particular latent mechanism sampled for the task.
- **Reference specification**: The declared probabilistic assumptions under which an oracle prediction is defined.
- **Latent-function uncertainty**: Uncertainty about the underlying function value at a query location, excluding noise from a new observation.
- **Observation uncertainty**: Uncertainty about a new observed target, including both latent-function uncertainty and observation noise.

- **Mechanism family**: A class of rules that generate dependencies or targets, with variation governed by declared parameters.
- **Generator coverage**: The set and distribution of prediction-problem regimes that a generator can produce or has demonstrably produced; declared support and measured coverage are distinct claims.
- **Observation process**: The process that turns underlying variables into observed features and targets, including noise, missingness and measurement transformations.
- **Selection filter**: A declared policy that chooses among otherwise valid generated tasks for a particular use.
- **Task card**: An inspectable account of one prediction problem, its generating assumptions, observations and diagnostic findings.

- **Proposal**: An identified request for a generated task under declared mechanism and regime choices, whether or not generation succeeds or the resulting task is selected.
- **Coverage cell**: A declared combination of task properties used to count and compare proposed and realized prediction problems.
- **Mechanism instance**: One sampled generating rule and its parameters, which may produce multiple observed datasets or episodes.
- **Task equivalence**: An explicitly defined relationship under which transformations preserve the prediction problem; it is narrower than unrestricted semantic sameness.
- **Structural difficulty**: Declared properties of a generating mechanism that may affect prediction, distinct from performance measured using a particular learner.

- **Added observation noise**: A declared perturbation applied to a generated target, distinct from randomness inside the underlying mechanism.
- **Label replacement**: A randomization event that draws a class label from the declared vocabulary; the sampled label may equal the original label.

- **Generator diagnostics**: Findings and measurements about generated prediction problems and their coverage, distinct from task-selection policies and downstream usefulness.

- **Milestone evaluation**: A final measurement of a frozen model selection under an accepted benchmark protocol, separated from development selection.
- **Dataset family**: Datasets sharing an underlying collection, including versions and derivatives whose overlap matters for evaluation.
- **Context draw**: One selection of labeled context examples for an identified prediction problem.
- **Protocol freeze**: Human acceptance of a fully identified evaluation procedure and its evidence; subsequent score-affecting changes define a different protocol.

- **Comparison**: An offline measurement of predictors on the same frozen query rows and context draws, retaining inputs, raw predictions, scores and attempt identities.

- **Generation recipe**: The declared mechanisms and parameter distributions used to sample synthetic tasks.
- **Generation run**: An identified collection of task-generation attempts under one recipe and explicit resource limits.
- **Generation trace**: The steps and sampled values explaining how a task's observations and targets were produced.

**Dataset explorer**: An interactive view of a generated task, its observations, diagnostic truth and generation trace.

**Chart sample**: A declared subset of a task's rows used for visualization; full-task summaries and exports still cover every row.

**Replay bundle**: A collection of a task's identity, recipe, sampled mechanism and observations, with predictor data separated from diagnostic truth.

## Generator diagnostics

**Diagnostic flag**:
A recorded property of a sampled synthetic task that warrants inspection without,
by itself, making the task invalid.

**Inspection filter**:
An explicit selection rule used to examine a subset of sampled synthetic tasks and
compare its distribution with the original draws.


## Causal and neural generators

**Structural causal model**:
A collection of variables whose values are determined by structural mechanisms,
their causal parents and exogenous variables.

**Hidden variable**:
A variable in the generating model that is absent from the predictor's observations.
_Avoid_: Missing value

**Structural innovation**:
An exogenous input supplying variation to a variable's structural mechanism.
_Avoid_: Added observation noise

**Hard intervention**:
A replacement of a variable's structural mechanism with an externally specified
value or value-setting rule, removing its usual dependence on causal parents.

**Soft intervention**:
A change to a variable's structural mechanism or exogenous-input distribution
without necessarily removing its dependence on causal parents.


**Node channel**:
One component of the value of a vector-valued causal variable; different channels
may be observed or hidden in a prediction task.

**Mechanism calibration**:
The determination of fixed properties of a sampled data-generating mechanism using
analytic quantities or an independent calibration sample.

**Matched intervention comparison**:
A comparison of outcomes for the same generated individuals before and after an
intervention, preserving their exogenous draws wherever the intervention permits.

**Population intervention comparison**:
A comparison of distributions before and after an intervention that can use new
exogenous draws rather than matching the same generated individuals.


**Native categorical target**:
A categorical variable of the generating model that is itself the prediction
outcome, rather than a label produced by a separate target adapter.
