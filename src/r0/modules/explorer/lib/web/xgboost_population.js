"use strict";
function isPopulationXGBoost(recipe=current?.view?.recipe?.recipe){return recipe?.version==="population-xgboost-graph-v1";}
function xgbPopulationDraft(){const draft=JSON.parse($("construction-controls").value);if(draft.topology!=="xgboost-population")throw new Error("Choose or load a population XGBoost construction first.");return draft;}
function fillXGBPopulationDraft(draft){
  $("xgb-population-base").value=pretty(draft.base);$("xgb-population-plans").value=pretty(draft.plans);
  $("xgb-population-mode").value=draft.population;$("xgb-population-engine").value=draft.engine;
  $("xgb-population-seed").value=String(draft.fit_seed);$("xgb-population-task").value=String(draft.fit_task_id);
}
$("construction-family").addEventListener("change",()=>{
  const draft=JSON.parse($("construction-controls").value);if(draft.topology!=="xgboost-population")return;
  fillXGBPopulationDraft(draft);
  if(draft.population!=="independent-ancestors"){$("seed").value=String(draft.fit_seed);$("task-id").value=String(draft.fit_task_id);}
});
$("xgb-population-load").onclick=()=>act("Loading retained fitting controls…",async()=>{
  const info=await request("xgboost_population_controls",{});if(!info.xgboost_population)throw new Error("Open a population XGBoost dataset first.");
  $("construction-controls").value=pretty(info.controls);fillXGBPopulationDraft(info.controls);
  $("xgb-population-rows").textContent=pretty(info.fit_rows);$("xgb-population-notice").textContent=info.notice;
});
$("xgb-population-apply").onclick=()=>act("Updating population fitting draft…",async()=>{
  const draft=xgbPopulationDraft();Object.assign(draft,{base:JSON.parse($("xgb-population-base").value),plans:JSON.parse($("xgb-population-plans").value),population:$("xgb-population-mode").value,engine:$("xgb-population-engine").value,fit_seed:$("xgb-population-seed").value,fit_task_id:$("xgb-population-task").value});
  $("construction-controls").value=pretty(draft);$("xgb-population-notice").textContent="Draft updated. Construct validates all fitting plans and identities before drawing targets or fitting models.";
});
$("xgb-population-replay").onclick=()=>act("Replaying retained XGBoost models…",async()=>show(await request("xgboost_population_replay",limits())));
function clearXGBPopulationCapabilities(){document.querySelectorAll('[data-xgb-pending="true"]').forEach(button=>{button.disabled=false;button.dataset.unavailable="false";delete button.dataset.xgbPending;button.removeAttribute("title");});}
function xgbPopulationView(view){
  if(!isPopulationXGBoost(view.recipe.recipe))return;
  const source=view.recipe.recipe,independent=source.population==="independent-ancestors";
  $("xgb-population-notice").textContent=independent?"Independent queries use the retained native models. Query identity must differ from the fitting identity. Next draws a new query without refitting.":"Context and whole-episode fits are tied to their original episode. Replay keeps their models; use an explicit new construction to fit another episode.";
  $("xgb-population-rows").textContent=pretty(source.fits.map(f=>({node:f.node,rows:f.row_ids})));
  const pending=["rows","context","features","active","task-type","classes","difficulty","pair-create","curve","intervention-load","intervention-add","intervention-create","policy-draft","policy-add-location","policy-create","mechanism-load"];
  if(!independent)pending.push("next");
  for(const id of pending){const button=$(id);button.disabled=true;button.dataset.unavailable="true";button.dataset.xgbPending="true";button.title="Retained population fits are immutable. Paired edits, policies and references remain pending; use an explicit fitting construction.";}
  $("next").textContent="Next frozen-model query";
}
for(const id of ["generate","next"])$(id).addEventListener("click",event=>{
  const source=current?.view?.recipe?.recipe;if(!isPopulationXGBoost(source))return;
  event.stopImmediatePropagation();event.preventDefault();
  act(id==="next"?"Constructing a query with frozen models…":"Replaying retained models…",async()=>{
    if($("recipe").value!==pretty(source))throw new Error("Unapplied edits retained. Use the fitting draft to construct another model.");
    if(id==="generate"){
      if($("seed").value!==String(current.view.recipe.seed)||$("task-id").value!==String(current.view.recipe.task_id))throw new Error("Replay requires the saved episode identity. Use Next or a new construction for another episode.");
      show(await request("xgboost_population_replay",limits()));
    }else show(await request("xgboost_population_query",{seed:$("seed").value,task_id:String(BigInt($("task-id").value)+1n),...limits()}));
  });
},true);
