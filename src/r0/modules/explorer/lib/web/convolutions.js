"use strict";
function isConvolutionRecipe(value) {
  if(!value || typeof value!=="object")return false;
  if(["sampled-feature-conv1d-r0-v1","frozen-feature-conv1d-r0-v1"].includes(value.version))return true;
  return Object.values(value).some(isConvolutionRecipe);
}
function convolutionConstructionNote(example){
  const enabled=example.topology==="feature-convolution";
  $("convolution-controls").hidden=!enabled;
  if(enabled)$("convolution-nodes").value=pretty(example.nodes);
}
$("convolution-apply").onclick=()=>act("Updating convolution construction draft…",async()=>{
  const draft=JSON.parse($("construction-controls").value), nodes=JSON.parse($("convolution-nodes").value);
  if(draft.topology!=="feature-convolution" || !Array.isArray(nodes) || !nodes.length)throw new Error("Choose a convolution construction and a nonempty node list.");
  draft.nodes=nodes;$("construction-controls").value=pretty(draft);
});
async function applyConvolutionDraft(){
  const candidate=structuredClone(mechanismDraft),selected=candidate.nodes[Number($("mechanism-node").value)],node=mechanismEquation(selected);
  if(node.pipeline)node.pipeline=JSON.parse($("mechanism-pipeline").value);
  if(mechanismInfo.converted)selected.converters=JSON.parse($("mechanism-domain").value);
  if(mechanismInfo.native)selected[selected.domains?"domains":"domain"]=JSON.parse($("mechanism-domain").value);
  node.parents=JSON.parse($("mechanism-parents").value);node.noise_scale=JSON.parse($("mechanism-noise").value);node.function=JSON.parse($("mechanism-function").value);
  const info=await request("convolution_draft",{recipe:candidate,...limits()});
  mechanismDraft=info.editable_recipe;mechanismInfo.nodes=info.nodes;
  $("mechanism-recipe").value=pretty(mechanismDraft);selectMechanismNode();
  $("workbench-note").textContent="Validated authored convolution draft. Save creates a linked edit; original kernel construction remains unchanged in the before archive.";
}
