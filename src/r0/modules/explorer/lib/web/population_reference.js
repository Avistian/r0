"use strict";
function isPopulationReference(recipe=current?.view?.recipe?.recipe){return ["recursive-population-graph-v1","recursive-population-graph-intervention-v1","calibrated-population-graph-v1","calibrated-population-query-v1","calibrated-population-intervention-v1"].includes(recipe?.version);}
function populationReferenceControls(){
  const active=isPopulationReference();$("population-reference-controls").hidden=!active;
  if(!active)return;
  $("curve").disabled=false;$("curve").dataset.unavailable="false";$("network-reference-label").hidden=true;
  $("curve-title").textContent="Original-row population causal reference";
  $("curve-description").textContent="Recompute descendants with this original row's own forest at every replaced node. Other observed values and keyed stochastic draws stay fixed. Scores, class probabilities and realized targets retain their actual target semantics.";
  $("curve-note").textContent="Continuous inputs use the observed range; categorical inputs enumerate declared categories. No forest resampling or query fitting. Reference JSON retains the complete source receipt and exact identity.";
  $("curve-points").max=129;
  const source=current.view.recipe.recipe, execution=source.version.startsWith("calibrated-population-")?source.original:source, base=(execution.base||execution).request.base;
  const rawOption=$("population-reference-output").querySelector('option[value="raw_scores"]');rawOption.disabled=!source.version.startsWith("calibrated-population-");if(rawOption.disabled&&$("population-reference-output").value==="raw_scores")$("population-reference-output").value="scores";
  const captures=base.version==="converted-causal-v1"&&base.observed.some(ref=>ref.capture!==null);
  $("population-reference-capture-label").hidden=!captures;
  if(!captures)$("population-reference-capture").value="capture";
  for(const option of $("population-reference-side").options)option.disabled=option.value!=="active"&&!current.intervention;
  if(!current.intervention)$("population-reference-side").value="active";
}
function populationReferenceArguments(){return isPopulationReference()?{population_reference_side:$("population-reference-side").value,capture_location:$("population-reference-capture").value,expected_intervention_id:current.intervention?.after_id}:{};}
function drawPopulationReference(){
  const output=$("population-reference-output").value;
  const matrix=output==="targets"?curve.targets.map(v=>[v]):curve[output];
  if(!matrix){$("response-plot").replaceChildren();$("curve-note").textContent="This retained regression target has no class probabilities. Select scores or realized targets.";return;}
  const channel=Math.min(number("channel"),matrix[0].length-1), values=matrix.map(row=>row[channel]);
  const title=`${curve.source_side} · original row ${curve.anchor_row} · ${curve.population_mode}`;
  const label=output==="probabilities"?`Class probability ${curve.target_classes[channel]}`:output==="targets"?"Realized target":output==="raw_scores"?"Raw structural target score "+channel:curve.calibration_fingerprint?"Fixed calibrated target score "+channel:"Target score/readout "+channel;
  const {svg,x,y}=plotFrame("response-plot",title,`x${curve.feature}`,label,...curve.bounds,Math.min(...values),Math.max(...values));
  if(!curve.vocabulary.length)svg.append(element("polyline",{points:curve.values.map((v,i)=>`${x(v)},${y(values[i])}`).join(" "),fill:"none",stroke:"#17826b","stroke-width":2}));
  curve.values.forEach((v,i)=>{const dot=element("circle",{cx:x(v),cy:y(values[i]),r:3,fill:"#17826b"});dot.append(element("title",{},`${curve.vocabulary[i]||v}: ${values[i]}`));svg.append(dot);});
  $("curve-note").textContent=`${curve.points} evaluated values; ${curve.population_mode} retained population, original row ${curve.anchor_row}. ${curve.score_semantics} ${curve.vocabulary.length?"Category codes: "+curve.vocabulary.map((v,i)=>`${i}=${v}`).join(", "):""}`;
}
for(const id of ["population-reference-side","population-reference-capture"])$(id).onchange=clearCurve;
$("population-reference-output").onchange=()=>{if(curve?.kind==="population_original_row_reference")drawCurve();};
