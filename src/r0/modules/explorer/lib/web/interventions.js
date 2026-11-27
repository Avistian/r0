"use strict";
let interventionInfo=null, interventionTrace=null;
function resetIntervention() {
  interventionInfo=null; interventionTrace=null;
  const standalone=current?.view?.recipe?.recipe?.version==="standalone-network-v1";
  $("intervention-load").hidden=standalone;
  $("intervention-notice").textContent=standalone
    ? "Causal interventions require an explicit causal graph. This standalone network supports architecture edits and task-property comparisons. Network mechanisms inside causal graphs support interventions."
    : "CPU execution. Hosted Colab and GPU verification remain pending.";
  $("intervention-controls").hidden=true; $("intervention-traces").hidden=true;
  showIntervention();
}
function interventionLocation() {
  if(interventionInfo?.vector) {
    $("intervention-location-note").textContent="Clamps one logical emitted value and its entire descendant-state block. Partial clamps retain the joint equation and correlated noise; clamping every variable bypasses that equation.";
    return;
  }
  const loc=$("intervention-location").value;
  $("intervention-location-note").textContent=loc==="state" ? "Clamps final state passed to descendants. Earlier converter captures remain unchanged; do(state) may not change an observed/target readout." : loc==="readout" ? "Clamps the measurement only. Descendant state stays unchanged: this is a readout intervention, not do(causal state)." : "Explicitly replaces both descendant state and selected readout. A fully overwritten mechanism can be bypassed; partial coupled mechanisms are still evaluated.";
}
function selectedInterventionDomain() {
  const node=interventionInfo.nodes[Number($("intervention-node").value)];
  return interventionInfo.vector ? node.domains[Number($("intervention-variable").value)] : node.domain;
}
function selectInterventionNode() {
  const node=interventionInfo.nodes[Number($("intervention-node").value)];
  $("intervention-variable-label").hidden=!interventionInfo.vector;
  $("intervention-variable").replaceChildren(...(node.domains || []).map((d,j)=>new Option(`Variable ${j} · ${d.kind} · ${d.emission}`,j)));
  selectInterventionVariable();
}
function selectInterventionVariable() {
  const node=interventionInfo.nodes[Number($("intervention-node").value)], domain=selectedInterventionDomain(), discrete=domain && domain.kind!=="continuous";
  $("intervention-category-label").hidden=!discrete; $("intervention-numerical").hidden=!!discrete;
  $("intervention-category").replaceChildren(...(domain?.vocabulary || []).map(label=>new Option(label,label)));
  $("intervention-location").disabled=interventionInfo.native;
  if(interventionInfo.native) $("intervention-location").value="state";
  $("intervention-function").value=pretty(node.function); $("intervention-noise").value="{}";
  $("intervention-node-note").textContent=`Node ${node.index} · ${node.function.version} · parents [${node.parents.join(", ")}]. ${discrete ? "Native category clamp replaces emitted value and one-hot descendant state; forced probabilities are indicators." : `Captured converter channels: [${node.captured_channels.join(", ")}].`}`;
  $("intervention-channels").disabled=interventionInfo.vector;
  $("intervention-domain-json").value=pretty(domain);
  $("intervention-stage-fields-json").value=pretty({root:null,replace_pipeline:false,pipeline:null});
  interventionLocation();
}
function showIntervention() {
  const pair=current?.intervention;
  $("intervention-results").hidden=!pair;
  if(!pair) return;
  $("intervention-path").value=pair.path;
  $("intervention-result-note").textContent=`${pair.card.mode==="matched"?"Matched units: unchanged exogenous draws are checked.":"Fresh population: equal row positions are not the same units."} Baseline remains active. Both panels use the same original row positions and shared axes.`;
  $("intervention-card").textContent=pretty(pair.card);
  const views=[current.view,pair.after], xs=views.flatMap(v=>v.rows.map(r=>r.inputs[0])), ys=views.flatMap(v=>v.rows.map(r=>r.target));
  ["before","after"].forEach((side,i)=>{
    const {svg,x,y}=plotFrame(`intervention-${side}-plot`,`${side} intervention`,"x0","Target value / declared category index",Math.min(...xs),Math.max(...xs),Math.min(...ys),Math.max(...ys));
    for(const row of views[i].rows) {
      const dot=element("circle",{cx:x(row.inputs[0]),cy:y(row.target),r:3,fill:row.role==="context"?"#17826b":"#d78949"});
      dot.append(element("title",{},`Original row ${row.index}: ${row.target_label ?? row.target}`));svg.append(dot);
    }
  });
}
function drawInterventionTrace() {
  const choice=$("intervention-stage").value, [side,index]=choice.split(":"), graph=interventionTrace[side], stage=graph.nodes[Number(index)];
  table("intervention-trace-table",["Original row",...stage.column_indices.map(c=>`channel ${c}`)],graph.rows.map((row,i)=>[row,...stage.values[i]]));
  $("intervention-trace-json").textContent=pretty({side,rows:graph.rows,...stage});
}
$("intervention-load").onclick=()=>act("Preparing intervention controls…",async()=>{
  interventionInfo=await request("intervention_controls");
  $("intervention-controls").hidden=false;$("intervention-notice").textContent=interventionInfo.notice;
  $("intervention-node").replaceChildren(...interventionInfo.nodes.map(n=>new Option(`Node ${n.index} · ${n.domains ? "joint vector" : (n.domain?.kind || "numerical")}`,n.index)));
  $("intervention-json").value="[]";selectInterventionNode();
});
$("intervention-node").onchange=selectInterventionNode;
$("intervention-variable").onchange=selectInterventionVariable;
$("intervention-location").onchange=interventionLocation;
$("intervention-kind").onchange=()=>{
  const kind=$("intervention-kind").value;
  $("intervention-hard").hidden=kind!=="hard";$("intervention-soft").hidden=kind!=="soft";
  $("intervention-stage-fields").hidden=kind!=="stage";$("intervention-domain-fields").hidden=kind!=="domain";
};
$("intervention-mode").onchange=()=>{$("intervention-mode-note").textContent=$("intervention-mode").value==="matched"?"Keep original row identities and unchanged root/innovation/emission draws. Explicitly changed noise laws and bypassed nodes are identified.":"Use a new keyed exogenous population with the same frozen graph and calibration. Distribution summaries describe these samples, not observational causal estimates.";};
$("intervention-mode").onchange();
$("intervention-add").onclick=()=>act("Adding explicit intervention…",async()=>{
  const node=Number($("intervention-node").value);let change;
  if($("intervention-kind").value==="hard") {
    const domain=selectedInterventionDomain(), discrete=domain && domain.kind!=="continuous";
    if(interventionInfo.vector) {
      const values=JSON.parse($("intervention-values").value);
      if(!discrete && (!Array.isArray(values) || values.length!==1)) throw new Error("A vector variable clamp requires one numeric value.");
      change={version:"native-vector-hard-intervention-v1",node,variable:Number($("intervention-variable").value),value:discrete?null:values[0],category:discrete?$("intervention-category").value:null};
    } else
    change={version:"hard-causal-intervention-v1",node,channels:discrete?[]:JSON.parse($("intervention-channels").value),values:discrete?[]:JSON.parse($("intervention-values").value),category:discrete?$("intervention-category").value:null,location:discrete?"state":$("intervention-location").value};
  } else if($("intervention-kind").value==="stage") {
    change={version:"stage-causal-intervention-v1",node,root:null,replace_pipeline:false,pipeline:null,domain:null,...JSON.parse($("intervention-stage-fields-json").value)};
  } else if($("intervention-kind").value==="domain") {
    const domain=JSON.parse($("intervention-domain-json").value);
    change=interventionInfo.vector?{version:"native-vector-domain-intervention-v1",node,variable:Number($("intervention-variable").value),domain}:{version:"stage-causal-intervention-v1",node,root:null,replace_pipeline:false,pipeline:null,domain};
  } else change={version:"soft-causal-intervention-v1",node,function:JSON.parse($("intervention-function").value),noise_scale:null,noise_response:null,exogenous:null,...JSON.parse($("intervention-noise").value)};
  const changes=JSON.parse($("intervention-json").value); if(!Array.isArray(changes)) throw new Error("Interventions must be a JSON array.");changes.push(change);$("intervention-json").value=pretty(changes);
});
$("intervention-create").onclick=()=>act("Generating, verifying and saving causal comparison…",async()=>{
  const result=await request("intervene",{interventions_json:$("intervention-json").value,mode:$("intervention-mode").value,population:$("intervention-population").value,...limits()});current=result;interventionTrace=null;$("intervention-traces").hidden=true;showIntervention();
});
$("intervention-open").onclick=()=>act("Replaying saved causal comparison…",async()=>show(await request("open_intervention",{path:$("intervention-path").value,...limits()})));
$("intervention-inspect").onclick=()=>act("Replaying actual before/after traces…",async()=>{
  const indices=$("intervention-rows").value.split(",").map(v=>Number(v.trim()));
  interventionTrace=await request("intervention_graph",{indices,expected_intervention_id:current.intervention.after_id,...limits()});
  $("intervention-stage").replaceChildren(...["before","after"].flatMap(side=>interventionTrace[side].nodes.map((n,i)=>new Option(`${side} · ${n.label}`,`${side}:${i}`))));
  $("intervention-traces").hidden=false;drawInterventionTrace();
});
$("intervention-stage").onchange=drawInterventionTrace;
$("intervention-export").onclick=()=>act("Verifying and exporting causal pair…",async()=>{
  const result=await request("intervention_export",{expected_intervention_id:current.intervention.after_id,...limits()});
  download(result.name,new Blob([Uint8Array.from(atob(result.data),c=>c.charCodeAt(0))],{type:result.mime}));
});
