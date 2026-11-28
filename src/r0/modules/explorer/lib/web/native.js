"use strict";
function categoryHistogram(container, labels, counts, title) {
  const maximum = Math.max(1, ...counts), width = Math.max(480, labels.length * 90);
  const svg = element("svg", {viewBox:`0 0 ${width} 270`,role:"img","aria-label":title});
  svg.append(element("rect",{width,height:270,fill:"white"}),element("text",{x:20,y:22,fill:"#637482","font-size":12},"Rows by declared category"));
  labels.forEach((label,i)=>{
    const slot=(width-40)/labels.length, h=170*counts[i]/maximum;
    const bar=element("rect",{x:20+i*slot,y:210-h,width:Math.max(1,slot-8),height:h,fill:"#25836b"});
    bar.append(element("title",{},`${label}: ${counts[i]} rows`)); svg.append(bar);
    svg.append(element("text",{x:20+i*slot+slot/2,y:232,"text-anchor":"middle","font-size":11,fill:"#23443b"},label));
  });
  $(container).replaceChildren(svg); chartExports(container,svg,container,title);
}
function nativeCharts(view,feature,channel) {
  const f=view.features[feature], summary=f.summary;
  if(summary.categories) categoryHistogram("feature-hist",summary.categories,summary.counts,f.name);
  else histogram("feature-hist",summary,f.name);
  if(view.classes.length) categoryHistogram("target-hist",view.classes,view.classification.observed_counts,"Native target");
  else histogram("target-hist",view.target,"Native target value");
  const calibration=view.target_calibration, calibrated=Boolean(calibration)&&!view.classes.length;
  const scoreLabel=calibrated ? "Fixed-calibrated target score" : view.converted ? "Converter score before category capture" : "Native equation score";
  const xLabel=summary.categories ? `${f.name} · declared category index` : f.name;
  scatter("signal-plot",view.rows.map(r=>({...r,x:r.inputs[feature],y:r.scores[Math.min(channel,r.scores.length-1)]})),xLabel,scoreLabel);
  $("diagnostic-title").textContent=view.classes.length ? `Native emission: ${view.classes[channel]}` : calibrated ? "Fixed-calibrated native target" : "Direct native target";
  $("diagnostic-note").textContent=view.classes.length ? (view.target_emission==="softmax" ? "Native softmax emission probabilities; no target adapter." : `Deterministic ${view.target_emission} emission; probabilities are one-hot indicators, not stochastic sampling.`) : calibrated ? `The selected variable target is centered and scaled using all ${calibration.statistics.rows} rows of the retained ${calibration.population_kind}. The fixed fit is applied unchanged to query rows; query rows do not determine these statistics. No observation-noise target adapter is applied.` : "The selected variable channel is the target; no observation-noise adapter or empirical normalization.";
  scatter("diagnostic-plot",view.rows.map(r=>({...r,x:r.inputs[feature],y:r.probabilities ? r.probabilities[channel] : r.target})),xLabel,view.classes.length ? "Native emission probability" : "Native target value");
  table("table",["Row","Split",...view.features.map(f=>`${f.name} (${f.kind})`),"Native target",scoreLabel],view.rows.slice(0,64).map(r=>[r.index,r.role,...r.display_inputs,r.target_label,r.scores[Math.min(channel,r.scores.length-1)]]));
}
function nativeViewControls(view) {
  const native=view.target_origin==="native-variable-v1";
  $("native-notice").hidden=!native;
  if(native) $("native-notice").textContent="Native direct target · declared categorical values and vocabularies are retained. Numeric arrays encode categories by vocabulary index. Edit equations and domains below; adapter pairs are unavailable. Original-row population references are available for retained population DAGs and GP-root graphs; other native response curves remain pending." + (["sampled-native-graph-v1","sampled-native-vector-graph-v1"].includes(view.native_construction) ? " Sampled domain construction and four RNG streams are retained. Frequency weights shift logits or reference cutpoints; they do not guarantee marginal class frequencies." : "");
  if(view.converted) $("native-notice").textContent="Typed converter captures · categories are extracted before structural noise and node importance. Descendants use the declared numeric representation, not an implicit one-hot category. Original-input modes preserve latent state. Frozen fitting and raw construction remain in the archive.";
  $("pair-panel").hidden=native;
  $("curve").closest("article").hidden=native&&!isPopulationReference(view.recipe.recipe)&&!isGPRootReference();
  $("signal-title").textContent=native ? (view.target_calibration ? "Feature and fixed-calibrated target score" : "Feature and native equation score") : "Feature and latent signal";
  $("difficulty-label").hidden=native;
  for(const id of ["features","active","task-type","classes"]) $(id).disabled=native;
}
