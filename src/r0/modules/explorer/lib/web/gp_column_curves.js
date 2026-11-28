"use strict";
function isGPColumns(){return current?.view?.recipe?.recipe?.version==="gp-column-recipe-v1";}
function isGPTargetReference(){return isGPColumns() || isGPTargetEnvironment();}
function gpColumnCurveControls(){
  gpColumnAvailability();
  $("gp-target-controls").hidden=!isGPTargetReference();
  if(!isGPTargetReference())return;
  const paired=current?.intervention?.card?.kind==="gp-target-edit";
  for(const option of $("gp-reference-side").options)option.disabled=option.value!=="active" && !paired;
  if(!paired)$("gp-reference-side").value="active";
  const previous=$("gp-target-metric").value, regression=current.view.task_type==="regression";
  const options=regression?[["mean","Expected observed target (noise mean zero)"],["scores","Independently normalized target"],["raw","Raw frozen target"]]:[["probabilities","Class probability (task separation + offsets)"],["scores","Independently normalized target score"],["raw","Raw frozen target score"]];
  $("gp-target-metric").replaceChildren(...options.map(([value,label])=>new Option(label,value)));
  if(options.some(([value])=>value===previous))$("gp-target-metric").value=previous;
  $("curve-title").textContent="Frozen GP-column target reference";
  $("curve").textContent="Evaluate frozen target reference";
  $("curve-description").textContent="Vary one observed GP column while holding the others at a retained original row. Evaluate the same frozen target with its original independent calibration. This target-equation counterfactual is not a conditional GP feature posterior, a new joint population or a general DAG intervention. Synthetic combinations may be outside joint feature support, even within observed marginal ranges. Explicit bounds may extrapolate.";
  $("curve-note").textContent=regression?"The expected target excludes zero-mean Gaussian observation noise; its saved standard deviation will be reported. No new targets or calibration are sampled.":"Class probabilities use the saved task separation and offsets. No labels or calibration are sampled. Binary raw/normalized scores are the positive-class score.";
}
function gpColumnCurveArguments(){
  const low=$("gp-target-lower").value.trim(), high=$("gp-target-upper").value.trim();
  const selection={gp_reference_side:$("gp-reference-side").value,expected_intervention_id:current?.intervention?.after_id};
  if(!low && !high)return selection;
  if(!low || !high)throw new Error("Supply both explicit bounds or leave both empty for the observed range.");
  return {...selection,bounds:[Number(low),Number(high)]};
}
function drawGPColumnCurve(){
  const metric=$("gp-target-metric").value;
  const matrix=metric==="raw"?curve.raw_scores:metric==="probabilities"?curve.probabilities:metric==="mean"?curve.regression_mean.map(v=>[v]):curve.scores;
  const channel=Math.min(number("channel"),matrix[0].length-1), values=matrix.map(row=>row[channel]);
  const label=metric==="probabilities"?`Class ${channel} probability`:metric==="mean"?"Expected observed target":metric==="raw"?`Raw target score ${channel}`:`Normalized target score ${channel}`;
  const {svg,x,y}=plotFrame("response-plot","Frozen target counterfactual",`x${curve.feature}`,label,...curve.bounds,Math.min(...values),Math.max(...values));
  svg.append(element("polyline",{points:curve.values.map((v,i)=>`${x(v)},${y(values[i])}`).join(" "),fill:"none",stroke:"#17826b","stroke-width":2}));
  $("curve-note").textContent=`${curve.points} points; fixed original row ${curve.anchor_row}; ${curve.range_source==="explicit"?"explicit bounds (extrapolation possible)":"observed marginal range"}. ${curve.observation_noise_std!==null?`Saved independent Gaussian observation-noise standard deviation: ${curve.observation_noise_std}. `:"Probabilities include separation and offsets; no labels drawn. "}${curve.score_semantics}`;
}
$("gp-target-metric").onchange=()=>{if(curve?.kind==="gp_column_target_reference")drawCurve();};
for(const id of ["gp-target-lower","gp-target-upper"])$(id).onchange=clearCurve;

function gpColumnAvailability(){
  gpTargetEditAvailability();
  for(const id of ["pair-create","mechanism-load","intervention-load","policy-create"]){
    const button=$(id);
    if(isGPColumns() || isGPTargetEnvironment()){button.disabled=true;button.dataset.unavailable="true";button.dataset.gpUnavailable="true";}
    else if(button.dataset.gpUnavailable==="true"){button.disabled=false;button.dataset.unavailable="false";delete button.dataset.gpUnavailable;}
  }
  if(isGPColumns()){
    $("pair-policy-note").textContent="Use the dedicated GP target editor for matched/fresh equation comparisons. Generic feature-count/noise pairs are unavailable here; the retained population cannot be treated as an independently evaluable iid function.";
    $("workbench-intro").textContent="Use the generation graph and frozen target reference to inspect this retained finite population. Use the dedicated GP target editor below for paired edits; generic DAG editing is unavailable.";
    $("intervention-notice").textContent="General DAG interventions and sampled policies require a causal graph. Finite-GP hybrid graph integration remains required; the target reference evaluates only the retained target equation.";
  }
}

$("gp-reference-side").onchange=clearCurve;
