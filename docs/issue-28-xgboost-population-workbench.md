# Actual-parent XGBoost Workbench

Choose a **Population XGBoost** construction example. The catalog spans all four
authored runtime families, linear/Conv/network roots, and three fitting lifecycles:

- `support-only`: fit the episode's actual context parent rows.
- `batch-refit-reference`: fit all actual rows of the named episode.
- `independent-ancestors`: fit a separately identified synthetic ancestor population;
  subsequent queries reuse its frozen native models.

The dedicated panel exposes the authored graph, node fitting plans, lifecycle,
fit seed/task identity and all five supported RNG engines. Plans retain the native
objective, fake-target distribution, tree count, depth and backend parameters.
Changing a draft does not modify a saved model. Explicit Construct fits a new model.
For support/batch fits, the generated episode must have the fitting identity.
Independent queries must have a different identity from the fit population.

Generate and Replay use the complete saved record. Next is available for
independent-ancestor models and changes the query identity without retraining.
The generation graph shows repeated original query rows and separately scoped
retained fitting tables, with fitting row IDs. Dataset export includes the model,
fit inputs, fake targets, streams and source graph. Reopened native and converted
tasks retain their declared output semantics.

All fitting and replay run inside the bounded worker. Construction custody is
checked against exact controls and stream snapshots without a second native fit.
Failures retain the prior active dataset. The public construction function is
`construct_xgboost_population(controls, construction_seed=..., seed=...,
task_id=..., limits=...)`.

Paired edits, target calibration, causal reference curves, sampled policies and
nested fitted environments remain required follow-ons. This slice does not
claim those compositions or hosted Colab/GPU acceptance. Existing source-option
hyperpriors and native backend categorical variants retain their original scope.
