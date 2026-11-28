"use strict";
function isCalibratedPopulation(recipe){return ["calibrated-population-graph-v1","calibrated-population-query-v1"].includes(recipe?.version);}
function isPopulationDAG(recipe){return recipe?.version==="recursive-population-graph-v1" || isCalibratedPopulation(recipe);}
function populationConstructionDraft(){
  const draft=JSON.parse($("construction-controls").value);
  if(draft.topology!=="recursive-population-dag")throw new Error("Choose or load a Population DAG construction first.");
  return draft;
}
function fillPopulationDraft(draft){
  $("population-dag-node").replaceChildren(...draft.request.replacements.map((binding,i)=>new Option(`Node ${binding.node} · original-row forest`,i)));
  $("population-dag-runtime").value=pretty(draft.request.base);
  $("population-dag-engine").value=draft.engine;
  selectPopulationNode();
}
function selectPopulationNode(){
  const draft=populationConstructionDraft(), binding=draft.request.replacements[Number($("population-dag-node").value)];
  $("population-dag-forest").value=pretty(binding.forest);
}
$("population-dag-node").onchange=()=>act("Selecting original-row mechanism…",async()=>selectPopulationNode());
$("construction-family").addEventListener("change",()=>{
  const draft=JSON.parse($("construction-controls").value);
  if(draft.topology==="recursive-population-dag")fillPopulationDraft(draft);
});
async function loadPopulationDraft(){
  const info=await request("population_dag_controls",{});
  if(!info.population_dag)throw new Error("Generate or open a population DAG first.");
  $("construction-controls").value=pretty(info.controls);fillPopulationDraft(info.controls);
  $("population-dag-notice").textContent=info.notice;
}
$("population-dag-load").onclick=()=>act("Loading retained population controls…",loadPopulationDraft);
$("population-dag-apply").onclick=()=>act("Updating population construction draft…",async()=>{
  const draft=populationConstructionDraft();
  draft.request.base=JSON.parse($("population-dag-runtime").value);
  draft.request.replacements[Number($("population-dag-node").value)].forest=JSON.parse($("population-dag-forest").value);
  draft.engine=$("population-dag-engine").value;
  $("construction-controls").value=pretty(draft);
  $("population-dag-notice").textContent="Draft retained. Construct validates complete geometry and budgets before drawing a new population; the active dataset remains unchanged until success.";
});
$("population-dag-replay").onclick=()=>act("Replaying exact original-row population…",async()=>show(await request("population_dag_replay",limits())));
$("next").addEventListener("click",event=>{
  const stored=current?.view?.recipe?.recipe;if(!isPopulationDAG(stored))return;
  event.stopImmediatePropagation();event.preventDefault();
  act("Constructing next original-row population…",async()=>{
    if($("recipe").value!==pretty(stored))throw new Error("Unapplied archive edits retained. Use population construction controls to create a new population.");
    const info=await request("population_dag_controls",{}), next=String(BigInt($("task-id").value)+1n);
    const result=isCalibratedPopulation(stored)
      ? await request("population_query",{construction_seed:$("construction-seed").value,seed:$("seed").value,task_id:next,engine:$("population-dag-engine").value || info.controls.engine,...limits()})
      : await request("construct",{controls:info.controls,construction_seed:$("construction-seed").value,seed:$("seed").value,task_id:next,...limits()});
    show(result);
  });
},true);
$("mechanism-load").addEventListener("click",event=>{
  if(!isPopulationDAG(current?.view?.recipe?.recipe))return;
  event.stopImmediatePropagation();event.preventDefault();
  act("Loading row-owned mechanism construction draft…",loadPopulationDraft);
},true);

function populationDAGView(view){
  $("mechanism-load").textContent="Inspect and prepare edit";
  const calibrated=isCalibratedPopulation(view.recipe.recipe);
  $("population-calibrate").disabled=calibrated || !isPopulationDAG(view.recipe.recipe);
  $("population-calibration-details").disabled=!calibrated;
  $("population-calibrate").dataset.unavailable=String($("population-calibrate").disabled);
  $("population-calibration-details").dataset.unavailable=String($("population-calibration-details").disabled);
  $("population-calibration-record").textContent="";
  $("population-calibration-notice").textContent=calibrated
    ? "Independent fit retained. Next creates only new query rows; Replay is exact. Matched/fresh interventions preserve the same fit. Geometry or law edits require explicit new calibration."
    : "Fit an independent same-law population explicitly. Original query rows and their archive remain unchanged.";
  if(!isPopulationDAG(view.recipe.recipe))return;
  $("next").textContent=calibrated ? "Next query · fixed calibration" : "Next population";
  $("mechanism-load").textContent="Load population construction controls";
  $("curve").disabled=true;$("curve").dataset.unavailable="true";
  $("curve-description").textContent="Population DAGs retain row-owned mechanisms. Reference curves remain a required follow-on.";
  $("curve-note").textContent="Population DAG reference curves remain a required follow-on. Inspect original-row forest paths and retained matched/fresh comparisons; no shared frozen response is substituted.";
}

$("population-calibrate").onclick=()=>act("Fitting retained independent population…",async()=>{
  if($("recipe").value!==pretty(current.view.recipe.recipe))throw new Error("Unapplied archive edits retained. Apply changes through a new explicit population before fitting.");
  show(await request("population_calibrate",{calibration_rows:number("population-fit-rows"),calibration_seed:$("population-fit-seed").value,calibration_task_id:$("population-fit-task").value,...limits()}));
});
$("population-calibration-details").onclick=()=>act("Reading retained calibration…",async()=>{
  const info=await request("population_dag_controls",{});
  $("population-calibration-record").textContent=pretty(info.calibration);
});
