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
