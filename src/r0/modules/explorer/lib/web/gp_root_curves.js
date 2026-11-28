"use strict";
function isGPRootReference(){return ["gp-root-dag-v1","gp-root-calibration-v1","gp-root-calibration-query-v1"].includes(current?.view?.recipe?.recipe?.version);}
function gpRootCurveControls(){
  $("gp-root-reference-controls").hidden=!isGPRootReference();
  if(!isGPRootReference())return;
  $("curve-title").textContent="Original-row GP causal reference";
  $("curve-description").textContent="Clamp every observed input while preserving the selected original row's hidden GP coordinates. Choose retained or zero structural innovations. The original independent fit remains fixed; no new GP population or target fit is created. This is not a conditional expectation.";
  $("curve").textContent="Evaluate GP causal reference";
  $("curve-points").max=129;
  $("gp-root-reference-metric").querySelector('option[value="scores"]').textContent=current.view.recipe.recipe.version==="gp-root-dag-v1"?"Target score (no independent fit)":"Fixed-calibrated target score";
  const categorical=current.view.task_type!=="regression";
  for(const option of $("gp-root-reference-metric").options)option.disabled=option.value==="probabilities"?!categorical:option.value==="target_mean"?categorical:false;
  if($("gp-root-reference-metric").selectedOptions[0]?.disabled)$("gp-root-reference-metric").value="scores";
  $("curve-note").textContent="At most 129 points and 20,000 total original-row evaluations. Category codes are discrete; full-grid JSON supports repeated rows.";
}
function gpRootCurveArguments(){
  const text=$("gp-root-reference-grid").value.trim();
  return {reference:$("gp-root-reference-noise").value,capture_location:$("gp-root-reference-capture").value,...(text?{inputs:JSON.parse(text)}:{})};
}
function drawGPRootCurve(){
  const metric=$("gp-root-reference-metric").value, raw=curve[metric];
  if(!raw){$("curve-note").textContent="This output does not apply to the retained target type.";return;}
  const matrix=Array.isArray(raw[0])?raw:raw.map(v=>[v]);
  const channel=Math.min(number("channel"),matrix[0].length-1),values=matrix.map(row=>row[channel]);
  const feature=curve.feature_schema[curve.feature], category=feature.vocabulary.length>0;
  const {svg,x,y}=plotFrame("response-plot","Original-row GP causal reference",category?`Category code x${curve.feature}`:`x${curve.feature}`,`${metric} ${channel}`,Math.min(...curve.values),Math.max(...curve.values),Math.min(...values),Math.max(...values));
  if(!category && metric!=="targets")svg.append(element("polyline",{points:curve.values.map((v,i)=>`${x(v)},${y(values[i])}`).join(" "),fill:"none",stroke:"#17826b","stroke-width":2}));
  curve.values.forEach((value,i)=>svg.append(element("circle",{cx:x(value),cy:y(values[i]),r:3,fill:"#17826b"})));
  $("curve-note").textContent=`${curve.points} explicit points, original row ${curve.row}; ${curve.structural_innovations} structural innovations, ${curve.capture_location} category cuts. ${category?`Vocabulary: ${feature.vocabulary.map((v,i)=>`${i}=${v}`).join(", ")}. `:""}Fit ${curve.fit_population_id || "none"}. Points are evaluated cuts; no behavior between category codes is implied.`;
}
for(const id of ["gp-root-reference-noise","gp-root-reference-capture","gp-root-reference-grid"])$(id).onchange=clearCurve;
$("gp-root-reference-metric").onchange=()=>{if(curve?.kind==="gp_root_causal_reference")drawCurve();};
