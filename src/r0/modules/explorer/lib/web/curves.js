"use strict";
let curve = null;
function isNoisyNetwork() {return current?.view.recipe.recipe.version === "standalone-network-innovations-v1";}
function isFunctionalReference() {return ["functional-causal-v1","functional-causal-v2","xgboost-fitted-functional-causal-v1"].includes(current?.view.recipe.recipe.version);}
function isGP() { return current?.view.recipe.recipe.version === "exact-gp-v1"; }
function clearCurve() {
  curve = null;
  $("response-plot").replaceChildren(); $("curve-values").textContent = ""; $("curve-json").hidden = true;
  const observed=isObservation(current?.view?.recipe?.recipe);
  $("curve").hidden=false; $("curve").disabled=false; $("curve").dataset.unavailable="false";
  const noisy=isNoisyNetwork(), functional=isFunctionalReference(); $("network-reference-label").hidden=!(noisy || functional || observed);
  const gp = isGP(); $("gp-options").hidden = !gp;
  $("curve-title").textContent = gp ? "Conditional GP view" : "Response curve";
  $("curve-description").textContent = gp
    ? "Vary one feature while fixing the others to the selected row. Condition on the complete saved GP draw, including calibration points. The mean and additional draws are new values; dots show original dataset values on this slice. Observation noise and class adjustments are excluded."
    : "Vary the feature selected above. Every other input stays fixed at the chosen row. This shows the saved function, without observation noise or class-separation adjustments.";
  if(noisy) {$("curve-title").textContent="Internal-noise latent reference"; $("curve-description").textContent="Vary one observed feature while fixing other inputs. Choose zero innovations (mean terms retained), or hold every internal innovation at the selected original row's draw. Root transformations and target adapters are not reapplied. This is not a conditional mean or class probability.";}
  if(functional) {$("curve-title").textContent="Causal fixed-latent reference"; $("curve-description").textContent="Vary one observed channel while fixing the other observed channels at the selected row. These are simultaneous state-and-readout clamps. Hidden root draws stay fixed; choose zero structural draws or their original-row values. The result is a raw target readout, not a conditional mean, calibrated task score or enclosing task probability.";}
  $("curve").textContent = gp ? "Evaluate conditional view" : "Evaluate response curve";
  $("curve-points").max = gp ? 256 : 512;
  $("curve-note").textContent = gp
    ? "Uses the full observed feature range. Additional draws use the seed above; changing the query points changes their joint draw."
    : "The curve uses the full observed feature range. Select a row and evaluate to see the response.";
  if(observed) {$("curve-title").textContent="Observation causal reference"; $("curve-description").textContent="Vary one selected observed channel, clamping every observed channel at both state and readout. Hidden roots stay at the selected original row; choose zero or original structural innovations. Target aliases propagate their parent clamps. These are raw target readouts, not conditional means, class probabilities or target-adapted scores."; $("curve-note").textContent="The full source and selection receipt remains attached to this original-row causal reference."; $("curve").textContent="Evaluate causal reference";}
  if(["sampled-intervention-policy-v1","sampled-typed-intervention-policy-v1"].includes(current?.view?.recipe?.recipe?.version)){$("curve").disabled=true;$("curve").dataset.unavailable="true";$("curve-description").textContent="Policy-environment reference curves remain pending explicit intervention-aware semantics. Inspect repeated original rows and the saved before/after comparison; no conditional mean is inferred.";}
  if(isConvolutionRecipe(current?.view?.recipe?.recipe) && !functional){$("curve").disabled=true;$("curve").dataset.unavailable="true";$("curve-description").textContent="Typed convolution curves remain pending explicit category/capture reference semantics. Inspect original-row kernel stages or use causal interventions; no continuous response is substituted for native emissions.";}

  gpColumnCurveControls();
}
function drawCurve() {
  if (!curve) return;
  if (curve.kind === "gp_column_target_reference") drawGPColumnCurve();
  else if (curve.kind === "gp_conditional_curve") drawGP();
  else {
    const channel = Math.min(number("channel"), curve.scores[0].length - 1);
    const values = curve.scores.map(row => row[channel]);
    const {svg, x, y} = plotFrame("response-plot", ["network_innovation_reference","functional_causal_reference","observation_causal_reference"].includes(curve.kind) ? `${curve.reference} innovation reference` : "Saved function response", `x${curve.feature}`, `${curve.kind === "observation_causal_reference" ? "Raw target readout" : "Raw latent signal"} ${channel}`, ...curve.bounds, Math.min(...values), Math.max(...values));
    svg.append(element("polyline", {points: curve.values.map((value, i) => `${x(value)},${y(values[i])}`).join(" "), fill: "none", stroke: "#17826b", "stroke-width": 2}));
    $("curve-note").textContent = `${curve.points} evaluated points across the observed range of x${curve.feature} (${curve.feature_role}). All other inputs are fixed to row ${curve.anchor_row}. Lines connect these points; behavior between them is not resolved. ${curve.score_semantics || ""}`;
  }
  $("curve-values").textContent = pretty(curve); $("curve-json").hidden = false;
}
function drawGP() {
  const c = Math.min(number("channel"), curve.mean[0].length - 1);
  const mean = curve.mean.map(row => row[c]), sd = curve.standard_deviation.map(row => row[c]);
  const lower = mean.map((value, i) => value - 2 * sd[i]), upper = mean.map((value, i) => value + 2 * sd[i]);
  const samples = curve.draws.map(draw => draw.map(row => row[c]));
  const original = curve.original.scores.map(row => row[c]);
  const limits = [...lower, ...upper, ...samples.flat(), ...original];
  if (!limits.every(Number.isFinite)) throw new Error("Conditional plot range exceeds finite precision; reduce the requested range.");
  const {svg, x, y} = plotFrame("response-plot", "Conditional GP view", `x${curve.feature}`, `Conditional latent signal ${c}`, ...curve.bounds, Math.min(...limits), Math.max(...limits));
  svg.setAttribute("viewBox", "0 0 480 350");
  svg.querySelector("rect").setAttribute("height", 350);
  const coordinates = values => curve.values.map((value, i) => `${x(value)},${y(values[i])}`);
  svg.append(element("polygon", {points: [...coordinates(lower), ...coordinates(upper).reverse()].join(" "), fill: "#d9eee8", "data-kind": "uncertainty"}));
  for (const [i, values] of samples.entries()) svg.append(element("polyline", {points: coordinates(values).join(" "), fill: "none", stroke: ["#8663c3", "#c88932", "#4b8ac3"][i % 3], "stroke-width": 1.4, "stroke-dasharray": "5 3", "data-kind": "additional-draw"}));
  svg.append(element("polyline", {points: coordinates(mean).join(" "), fill: "none", stroke: "#17826b", "stroke-width": 2.5, "data-kind": "conditional-mean"}));
  for (let i = 0; i < original.length; i++) {
    const dot = element("circle", {cx: x(curve.original.values[i]), cy: y(original[i]), r: 3.5, fill: "#273a4c", stroke: "white", "stroke-width": 1, "data-kind": "original"});
    dot.append(element("title", {}, `Original row ${curve.original.row_indices[i]}: ${original[i]}`)); svg.append(dot);
  }
  const diagnostics = curve.conditioning;
  const numerical = diagnostics.approximate
    ? `Approximate: ${diagnostics.retained_rank}/${diagnostics.support_count} directions; error ${format(diagnostics.discarded_support_max_error)}`
    : `Strict conditioning on ${diagnostics.support_count} saved support points`;
  const captions = ["Mean: solid green · band: ±2 conditional standard deviations", "Dashed: additional draws · dots: original dataset values", `Original: ${curve.original.shown_count}/${curve.original.matched_count} matching rows · fixed row ${curve.anchor_row}`, numerical];
  captions.forEach((text, i) => svg.append(element("text", {x: 16, y: 282 + i * 19, fill: "#526677", "font-size": 11, "font-family": "sans-serif"}, text)));
  $("curve-note").textContent = `${numerical}. ${curve.points} points; ${samples.length} additional draws${samples.length ? ` with seed ${curve.sampling.seed}` : ""}. The band is pointwise uncertainty of the latent function, not observation noise. Lines connect evaluated points. Dots match the fixed active coordinates; irrelevant inputs may differ. ${diagnostics.approximate ? "The reported error is the largest normalized change to the saved support projection. Original values still take precedence at exact saved coordinates." : ""}`;
}
function curveArguments() {
  return {...(isGPTargetReference()?gpColumnCurveArguments():{}), feature: number("feature"), anchor_row: number("curve-row"), points: number("curve-points"), ...((isNoisyNetwork() || isFunctionalReference() || isObservation(current?.view?.recipe?.recipe)) ? {reference:$("network-reference").value} : {}), ...(isGP() ? {draws: number("gp-draws"), seed: number("gp-draws") ? $("gp-seed").value : null, allow_rank_truncation: $("gp-approximate").checked} : {})};
}
for (const id of ["network-reference", "curve-row", "curve-points", "gp-draws", "gp-seed", "gp-approximate"]) $(id).onchange = clearCurve;
$("curve").onclick = () => act("Evaluating saved function…", async () => {
  clearCurve();
  const parameters = curveArguments();
  const result = await request(isGP() ? "gp_curve" : "curve", {...parameters,...limits()});
  if (pretty(parameters) === pretty(curveArguments())) { curve = result.curve; drawCurve(); }
});
$("curve-json").onclick = () => download("response-curve.json", new Blob([pretty(curve)], {type: "application/json"}));
