"use strict";
function isObservation(recipe) { return recipe?.version === "network-observation-recipe-v1"; }
function observationConstructionNote(example) {
  const enabled=example.topology==="network-observations";
  $("observation-controls").hidden=!enabled;
  if(!enabled)return;
  const p=example.observations;
  $("observation-profile").value=p.profile; $("observation-selection").value=p.selection;
  $("observation-target").value=p.target; $("observation-order").value=p.feature_order;
  $("observation-seed").value=example.selection_seed;
}
$("observation-apply").onclick=()=>act("Updating observation policy draft…",async()=>{
  const draft=JSON.parse($("construction-controls").value);
  if(draft.topology!=="network-observations")throw new Error("Choose an observation construction first.");
  const seed=$("observation-seed").value.trim();
  if(!/^\d+$/.test(seed)||BigInt(seed)>=2n**63n)throw new Error("Selection seed must be an exact integer below 2^63.");
  draft.selection_seed=seed;
  Object.assign(draft.observations,{profile:$("observation-profile").value,selection:$("observation-selection").value,target:$("observation-target").value,feature_order:$("observation-order").value});
  $("construction-controls").value=pretty(draft);
});

async function applyObservationDraft() {
  const candidate=structuredClone(mechanismDraft), index=Number($("mechanism-node").value),node=candidate.nodes[index];
  if(node.pipeline)node.pipeline=JSON.parse($("mechanism-pipeline").value);
  node.parents=JSON.parse($("mechanism-parents").value);node.noise_scale=JSON.parse($("mechanism-noise").value);node.function=JSON.parse($("mechanism-function").value);
  const info=await request("observation_draft",{recipe:candidate,...limits()});
  mechanismDraft=info.editable_recipe;mechanismInfo.nodes=info.nodes;
  $("mechanism-recipe").value=pretty(mechanismDraft);selectMechanismNode();
  $("workbench-note").textContent="Validated authored graph draft. Original source receipt remains unchanged in the before archive; save to create a linked edit.";
}
let observationNodeDrafts=new Map(),observationSelectedNode=null;
function resetObservationDrafts(){observationNodeDrafts.clear();observationSelectedNode=null;}
function observationNodeDraft(index){
  if(!mechanismInfo?.observation_selection && !mechanismInfo?.convolution)return null;
  const fields=["mechanism-function","mechanism-parents","mechanism-noise","mechanism-pipeline",...(mechanismInfo.convolution?["mechanism-domain"]:[])];
  if(observationSelectedNode!==null)observationNodeDrafts.set(observationSelectedNode,Object.fromEntries(fields.map(id=>[id,$(id).value])));
  observationSelectedNode=index;
  return observationNodeDrafts.get(index)||null;
}
