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
  const xLabel=summary.categories ? `${f.name} · declared category index` : f.name;
  scatter("signal-plot",view.rows.map(r=>({...r,x:r.inputs[feature],y:r.scores[Math.min(channel,r.scores.length-1)]})),xLabel,"Native equation score");
  $("diagnostic-title").textContent=view.classes.length ? `Native emission: ${view.classes[channel]}` : "Direct native target";
  $("diagnostic-note").textContent=view.classes.length ? (view.target_emission==="softmax" ? "Native softmax emission probabilities; no target adapter." : `Deterministic ${view.target_emission} emission; probabilities are one-hot indicators, not stochastic sampling.`) : "The selected variable channel is the target; no observation-noise adapter or empirical normalization.";
  scatter("diagnostic-plot",view.rows.map(r=>({...r,x:r.inputs[feature],y:r.probabilities ? r.probabilities[channel] : r.target})),xLabel,view.classes.length ? "Native emission probability" : "Native target value");
  table("table",["Row","Split",...view.features.map(f=>`${f.name} (${f.kind})`),"Native target","Equation score"],view.rows.slice(0,64).map(r=>[r.index,r.role,...r.display_inputs,r.target_label,r.scores[Math.min(channel,r.scores.length-1)]]));
}
function nativeViewControls(view) {
  const native=view.target_origin==="native-variable-v1";
  $("native-notice").hidden=!native;
  if(native) $("native-notice").textContent="Native direct target · declared categorical values and vocabularies are retained. Numeric arrays encode categories by vocabulary index. Edit equations and domains below; adapter pairs and response curves are unavailable.";
  $("pair-panel").hidden=native;
  $("curve").closest("article").hidden=native;
  $("signal-title").textContent=native ? "Feature and native equation score" : "Feature and latent signal";
  $("difficulty-label").hidden=native;
  for(const id of ["features","active","task-type","classes"]) $(id).disabled=native;
}
