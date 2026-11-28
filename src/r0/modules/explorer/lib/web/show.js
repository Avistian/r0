"use strict";
function show(result) {
  clearGPRootCapabilities();
  clearXGBPopulationCapabilities();
  current = result; const view = result.view;
  $("empty").hidden = true; $("results").hidden = false;
  $("next").textContent=view.recipe.recipe.version==="gp-column-recipe-v1" ? "Next GP population + target" : isPrototypeBatch(view.recipe.recipe) ? "Next population" : "Next task";
  $("family-title").textContent = isPrototypeBatch(view.recipe.recipe) ? "standalone network · batch prototype population" : view.recipe.recipe.version === "standalone-network-innovations-v1" ? "standalone network · internal innovations" : isObservation(view.recipe.recipe) ? "network observations · selected channels" : view.family;
  $("task-title").textContent = view.task_type === "regression" ? "Regression dataset" : `${view.classes.length}-class dataset`;
  $("metrics").replaceChildren();
  for (const [label, value] of [["TOTAL ROWS", view.total_rows], ["FEATURES", view.features.length], ["CONTEXT ROWS", view.split_counts.context], ["QUERY ROWS", view.split_counts.query]]) {
    const box = document.createElement("div"); box.className = "metric";
    const strong = document.createElement("strong"), caption = document.createElement("span"); strong.textContent = format(value); caption.textContent = label; box.append(strong, caption); $("metrics").append(box);
  }
  $("sampling").textContent = `Charts show ${view.shown_rows} of ${view.total_rows} rows: ${view.sampling}. The table previews the first ${Math.min(64, view.shown_rows)} sampled rows. Distribution summaries use all ${view.total_rows} rows.`;
  $("identity").textContent = pretty({source_task_id: view.source_task_id, episode_id: view.episode_id, archive: result.archive, generation: result.run?.summary, elapsed_seconds: result.run?.elapsed_seconds, execution: view.recipe.execution, ...(view.feature_schema ? {feature_schema:view.feature_schema,target_origin:view.target_origin} : {})});
  $("feature").replaceChildren(...view.features.map(feature => new Option(`${feature.name} · ${feature.role}`, feature.index)));
  $("channel").replaceChildren(...(view.classes.length ? view.classes : ["Signal"]).map((name, i) => new Option(name, i)));
  $("class-panel").hidden = !view.classification;
  if (view.classification) {
    const stats = view.classification;
    table("class-counts", ["Class", "Observed", "Context", "Query", "Expected"], view.classes.map((name, i) => [name, stats.observed_counts[i], stats.context_counts[i], stats.query_counts[i], stats.expected_counts[i]]));
  }
  $("selected-row").hidden = true; $("parameter-values").textContent = "";
  $("row-index").max = view.total_rows - 1; $("row-index").value = 0;
  $("archive-path").value = result.archive; $("saved-id").value = view.recipe.task_id;
  $("replay-record").value = result.recipe_json; $("seed").value = view.recipe.seed; $("task-id").value = view.recipe.task_id;
  $("curve-row").value = 0; $("curve-row").max = view.total_rows - 1;
  $("gp-approximate").checked = false;
  $("construction-details").hidden=!view.construction_details;
  $("construction-detail-notice").textContent=view.construction_details?.notice || "";
  $("construction-detail-record").textContent=pretty(view.construction_details || {});
  fillControls(view.recipe.recipe); charts(); clearCurve(); resetGraph(); resetPair(); resetWorkbench(); nativeViewControls(view); resetIntervention(); specialPopulationViews(view); xgbPopulationView(view);
}
