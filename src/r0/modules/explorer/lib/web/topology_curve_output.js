"use strict";
function clearTopologyCurveOutput() {
  const recipe=current?.view?.recipe?.recipe;
  const active=["sampled-hyperprior-graph-v1","sampled-joint-hyperprior-graph-v1"].includes(recipe?.version);
  $("topology-output-label").hidden=!active;
  $("network-reference-label").querySelector("span").textContent=active?"This selects structural innovations. Choose raw readouts or enclosing target-adapter outputs below.":"Raw latent channels; neither a conditional mean nor a class probability.";
  $("topology-output").value="scores";
  const regression=recipe?.request?.task?.task_type==="regression";
  for(const option of $("topology-output").options) {
    option.disabled=option.value==="target_mean"?!regression:option.value==="target_probabilities"?regression:false;
  }
}
function topologyCurveOutput(value) {
  if(!value.target_reference_version)return {matrix:value.scores,label:null};
  const mode=$("topology-output").value;
  if(mode==="target_probabilities" && value.target_probabilities) return {matrix:value.target_probabilities,label:"Class probability"};
  if(mode==="target_mean" && value.target_mean)return {matrix:value.target_mean.map(x=>[x]),label:"Target-noise mean"};
  if(mode==="fixed_draw_targets")return {matrix:value.fixed_draw_targets.map(x=>[x]),label:value.task_type==="regression"?"Fixed-noise target":"Fixed-uniform class code"};
  return {matrix:value.scores,label:"Raw target readout"};
}
$("topology-output").onchange=()=>drawCurve();

function describeTopologyCurveOutput() {
  if($("topology-output-label").hidden)return;
  $("curve-description").textContent="Keep the realized topology and mechanisms frozen. Clamp observed state and readout channels; hidden roots and structural innovations use the selected original row or explicit zero innovations. Select raw readouts, regression target-noise mean, enclosing class probabilities, or targets using the original row's fixed noise/uniform draw. These are causal references, not observational conditional distributions. No query population or calibration is refitted.";
}
