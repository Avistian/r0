"use strict";
function gpRootEnvelope(recipe){if(recipe?.version==="gp-root-intervention-v1")return {...gpRootEnvelope(recipe.base),source:recipe.population};return ["gp-root-calibration-v1","gp-root-calibration-query-v1"].includes(recipe?.version)?recipe.original:recipe?.version==="gp-root-dag-v1"?recipe:null;}
function gpRootDraft(){
  const draft=JSON.parse($("construction-controls").value);
  if(draft.topology!=="gp-root-dag")throw new Error("Select or load a GP root DAG construction first.");
  return draft;
}
function fillGPRootDraft(draft){
  $("gp-root-request").value=pretty(draft.request);
  $("gp-root-bindings").value=pretty(draft.bindings);
  $("gp-root-runtime").value=pretty(draft.base);
}
$("construction-family").addEventListener("change",()=>{
  const draft=JSON.parse($("construction-controls").value);
  if(draft.topology==="gp-root-dag")fillGPRootDraft(draft);
});
async function loadGPRootDraft(){
  const info=await request("gp_root_controls",{});
  if(!info.gp_root_dag)throw new Error("Generate or open a GP-root dataset first.");
  $("construction-controls").value=pretty(info.controls);fillGPRootDraft(info.controls);
  $("gp-root-notice").textContent=info.notice;
}
$("gp-root-load").onclick=()=>act("Loading GP-root construction controls…",loadGPRootDraft);
$("gp-root-replay").onclick=()=>act("Replaying retained GP-root population…",async()=>show(await request("gp_root_replay",limits())));
$("gp-root-apply").onclick=()=>act("Updating GP-root draft…",async()=>{
  const draft=gpRootDraft();
  draft.request=JSON.parse($("gp-root-request").value);
  draft.bindings=JSON.parse($("gp-root-bindings").value);
  draft.base=JSON.parse($("gp-root-runtime").value);
  $("construction-controls").value=pretty(draft);
  $("gp-root-notice").textContent="Draft updated. Construct checks the complete GP population, graph, bindings and resource limits before drawing a new population.";
});
$("next").addEventListener("click",event=>{
  const stored=current?.view?.recipe?.recipe;if(!gpRootEnvelope(stored))return;
  event.stopImmediatePropagation();event.preventDefault();
  act("Constructing next GP-root population…",async()=>{

    if($("recipe").value!==pretty(stored))throw new Error("Unapplied archive edits retained. Use the GP-root construction draft to change the graph.");
    if(stored.version!=="gp-root-dag-v1"){
      show(await request("gp_root_query",{seed:$("seed").value,task_id:String(BigInt($("task-id").value)+1n),locations:JSON.parse($("gp-query-locations").value),...limits()}));return;
    }
    const info=await request("gp_root_controls",{});
    show(await request("construct",{
      controls:info.controls,construction_seed:$("construction-seed").value,
      seed:$("seed").value,task_id:String(BigInt($("task-id").value)+1n),...limits()
    }));
  });
},true);
$("mechanism-load").addEventListener("click",event=>{
  if(!gpRootEnvelope(current?.view?.recipe?.recipe))return;
  event.stopImmediatePropagation();event.preventDefault();
  act("Loading GP-root construction draft…",loadGPRootDraft);
},true);
function gpRootView(view){
  if(!gpRootEnvelope(view.recipe.recipe))return;
  $("next").textContent="Next GP population";
  $("mechanism-load").textContent="Load GP-root controls";
  $("gp-root-notice").textContent="Replay preserves the retained GP population. Next constructs a new population with the same authored graph and bindings.";
}
function specialPopulationViews(view){
  gpColumnAvailability();populationDAGView(view);gpRootView(view);gpCalibrationView(view);gpRootCapabilities(view);
}
