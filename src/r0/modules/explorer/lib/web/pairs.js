"use strict";
const pairNames = {noise: "Noise / signal", irrelevant: "Total feature count", sparsity: "Active feature count", separation: "Class separation"};
const pairFields = {noise: "noise_ratio", irrelevant: "features", sparsity: "active_features", separation: "separation"};
function resetPair() {
  const regression = current.view.task_type === "regression";
  const choices = ["causal","observed_network"].includes(current.view.family) ? [regression ? "noise" : "separation"] : [regression ? "noise" : "separation", "irrelevant", "sparsity"];
  $("pair-intervention").replaceChildren(...choices.filter(key => current.view.recipe.recipe.version !== "network-input-view-v2" || key !== "sparsity").map(key => new Option(pairNames[key], key)));
  $("pair-metric").replaceChildren(...(regression ? [["Observed target", "target"], ["Latent signal", "scores"], ["Residual noise", "noise"]] : [["Class probability", "probabilities"], ["Latent score", "scores"], ["Sampled label index", "target"]]).map(([label, value]) => new Option(label, value)));
  $("pair-estimate").checked = false;
  $("pair-feature").replaceChildren(); $("pair-channel").replaceChildren();
  pairControls(); showPair();
}
function pairControls() {
  if (!current) return;
  const intervention = $("pair-intervention").value, task = taskControls(current.view.recipe.recipe), old = task[pairFields[intervention]];
  $("pair-value-label").firstChild.textContent = `${pairNames[intervention]} · currently ${old}`;
  $("pair-value").step = ["sparsity", "irrelevant"].includes(intervention) ? "1" : "0.1";
  $("pair-value").value = intervention === "irrelevant" ? Math.min(256, old + 8) : intervention === "sparsity" ? Math.max(1, Math.floor(old / 2)) : intervention === "separation" ? 0 : old > 0 ? old * 2 : 0.5;
  const estimated = intervention === "sparsity" && current.view.recipe.recipe.version !== "linear-v1";
  $("pair-estimate-label").hidden = !estimated;
  $("pair-estimate").checked = false;
  $("pair-policy-note").textContent = estimated ? "This family uses independent calibration to target unit signal variance. True expected variances may differ. Check the box to request this comparison explicitly." : intervention === "sparsity" ? "Gaussian linear coefficients preserve expected unit signal variance; realized sample variance can differ." : "The saved latent function stays unchanged. Only the selected control changes.";
  if (current.view.family === "standalone_network") $("pair-policy-note").textContent = intervention === "sparsity" ? "An explicit mask zeroes selected inputs after the original joint root. The frozen architecture is retained. Signal variance is uncontrolled by default. Check the box to estimate the original baseline mean and spread on an independent shared population; this does not target unit variance." : intervention === "irrelevant" ? "Extra observed columns use independent keyed Gaussian draws. Original inputs and frozen network outputs stay unchanged." : "The saved network, input mask and root distribution stay unchanged. Only the selected task control changes.";
}
function showPair() {
  const pair = current?.pair;
  $("pair-results").hidden = !pair;
  $("pair-row").hidden = true;
  if (!pair) return;
  const common = Math.min(current.view.features.length, pair.after.view.features.length);
  const previous = $("pair-feature").value;
  $("pair-feature").replaceChildren(...current.view.features.slice(0, common).map(feature => new Option(`${feature.name} · ${feature.role} → ${pair.after.view.features[feature.index].role}`, feature.index)));
  const firstActive = current.view.features.slice(0, common).find(feature => feature.role === "active")?.index || 0;
  $("pair-feature").value = previous !== "" && Number(previous) < common ? previous : String(firstActive);
  $("pair-channel").replaceChildren(...(current.view.classes.length ? current.view.classes : ["Signal"]).map((name, i) => new Option(name, i)));
  $("pair-path").value = pair.path;
  drawPair();
}
function pairValue(row, metric, channel) {
  return metric === "scores" ? row.scores[Math.min(channel, row.scores.length - 1)] : metric === "probabilities" ? row.probabilities[channel] : row[metric];
}
function drawPair() {
  if (!current?.pair) return;
  const before = current.view, pair = current.pair, after = pair.after.view, card = pair.card;
  const feature = number("pair-feature"), channel = number("pair-channel"), metric = $("pair-metric").value;
  const name = pairNames[card.intervention];
  $("pair-title").textContent = `${name}: ${card.control.before} → ${card.control.after}`;
  $("pair-before-title").textContent = `Before · ${name} ${card.control.before}`;
  $("pair-after-title").textContent = `After · ${name} ${card.control.after}`;
  const variance = {independent_network_baseline_moments_estimate: "Independent shared-population estimate matches the original baseline mean and spread; true population equality is not established.", network_input_mask_uncontrolled: "Signal variance is uncontrolled after input masking.", unchanged_mechanism: "Identical latent function and normalizers.", analytic_unit_variance: "Expected signal variance is exactly 1 under standard Gaussian inputs. Sample spread may differ.", independent_calibration_unit_variance_estimate: "Calibration estimate: each variant targets unit variance using an independent calibration sample. Equality of true expected variances is not established."};
  $("pair-variance").textContent = `${variance[card.variance_control]} One coupled root draw, two variants.`;
  const labels = {seed: "root seed", task_id: "task identifier", execution: "generation environment", common_inputs: "shared input columns", scores: "latent scores", mechanism: "sampled mechanism", context_indices: "context rows", query_indices: "query rows", classes: "declared classes", targets: "observed targets", probabilities: "class probabilities", nested_active_features: "nested active feature sets"};
  $("pair-fixed").textContent = `Verified: ${Object.keys(card.required_equalities).map(key => labels[key] || key).join(" · ")}`;
  $("pair-sampling").textContent = `Both plots use the same ${before.shown_rows} of ${before.total_rows} original rows and identical axes. Other features vary across rows; these are observed points, not response curves. Before: ${before.features.length} columns; after: ${after.features.length}.`;
  const rows = [before.rows, after.rows], xs = rows.flatMap(items => items.map(row => row.inputs[feature])), ys = rows.flatMap(items => items.map(row => pairValue(row, metric, channel)));
  const xLow = Math.min(...xs), xHigh = Math.max(...xs), yLow = metric === "probabilities" ? 0 : Math.min(...ys), yHigh = metric === "probabilities" ? 1 : Math.max(...ys);
  const metricName = metric === "scores" && before.task_type === "binary" ? "Positive-class latent score" : $("pair-metric").selectedOptions[0].textContent;
  const yLabel = metric === "probabilities" || metric === "scores" && before.task_type === "multiclass" ? `${metricName} · ${before.classes[channel]}` : metricName;
  ["before", "after"].forEach((side, index) => {
    const view = index ? after : before;
    const role = index ? "After" : "Before", value = index ? card.control.after : card.control.before;
    const {svg, x, y} = plotFrame(`pair-${side}-plot`, `${role}: ${name} ${value}`, `x${feature}`, yLabel, xLow, xHigh, yLow, yHigh);
    svg.dataset.xLow = xLow; svg.dataset.xHigh = xHigh; svg.dataset.yLow = yLow; svg.dataset.yHigh = yHigh;
    svg.setAttribute("viewBox", "0 0 480 312"); svg.querySelector("rect").setAttribute("height",312);
    svg.append(element("text", {x: 62, y: 280, fill: "#334556", "font-size":11, "font-family":"sans-serif"}, `${role} · ${name} ${value} · ${view.shown_rows}/${view.total_rows} rows`));
    svg.append(element("text", {x: 62, y: 298, fill: "#637482", "font-size":10, "font-family":"sans-serif"}, "Green: context · orange: query · shared axes across the pair"));
    for (const row of view.rows) {
      const point = element("circle", {cx:x(row.inputs[feature]), cy:y(pairValue(row, metric, channel)), r:3, fill:row.role === "context" ? "#17826b" : "#d78949", opacity:.65, tabindex:0, role:"button", "aria-label":`Inspect paired row ${row.index}`, "data-row":row.index});
      point.append(element("title", {}, `Row ${row.index} · ${row.role}\nx${feature}: ${row.inputs[feature]}\n${yLabel}: ${pairValue(row,metric,channel)}`));
      const inspect = () => { $("pair-row").hidden = false; $("pair-row").textContent = pretty({before:before.rows.find(item => item.index === row.index), after:after.rows.find(item => item.index === row.index)}); };
      point.onclick = inspect; point.onkeydown = event => {if (["Enter", " "].includes(event.key)) {event.preventDefault(); inspect();}};
      svg.append(point);
    }
  });
  const scoreChannel = Math.min(channel, card.before.signal_std.length - 1);
  const measures = [["Total features", "features"], ["Active features", "active_features"], ["Signal std (selected score)", "signal_std"], ...(before.task_type === "regression" ? [["Residual std", "residual_std"]] : [["Mean class entropy (nats)", "mean_entropy"]])];
  table("pair-measurements", ["Measurement", "Before", "After", "After − before"], measures.map(([label,key]) => {const a = key === "signal_std" ? card.before[key][scoreChannel] : card.before[key], b = key === "signal_std" ? card.after[key][scoreChannel] : card.after[key]; return [label,a,b,b-a];}).concat([["Fraction of observed targets changed", "—", card.changed_target_fraction, "—"]]));
  $("pair-classes").hidden = !before.classes.length;
  if (before.classes.length) table("pair-counts", ["Class", "Before observed", "After observed", "Before expected", "After expected"], before.classes.map((label,i) => [label,card.before.class_counts[i],card.after.class_counts[i],card.before.expected_class_counts[i],card.after.expected_class_counts[i]]));
  $("pair-details").textContent = pretty({saved_comparison:pair.path,before_archive:current.archive,after_archive:pair.after.archive,card:card,variant_run:pair.run});
}
$("pair-intervention").onchange = pairControls;
for (const id of ["pair-feature", "pair-channel", "pair-metric"]) $(id).onchange = drawPair;
$("pair-create").onclick = () => act("Generating and verifying comparison…", async () => {
  current = await request("pair", {intervention:$("pair-intervention").value,value:number("pair-value"),allow_estimated_variance:$("pair-estimate").checked,max_seconds:number("seconds"),max_bytes:number("bytes") * 1024**2});
  showPair();
});
$("pair-open").onclick = () => act("Verifying saved comparison…", async () => {show(await request("open_pair", {path:$("pair-path").value})); $("pair-panel").scrollIntoView({block:"start"});});
$("pair-card").onclick = () => download("comparison-card.json", new Blob([current.pair.card_json], {type:"application/json"}));
$("pair-export").onclick = () => act("Exporting both datasets…", async () => {
  const result = await request("pair_export", {expected_pair_id:current.pair.card.pair_id,max_bytes:number("bytes") * 1024**2});
  download(result.name, new Blob([Uint8Array.from(atob(result.data), c => c.charCodeAt(0))], {type:result.mime}));
});
