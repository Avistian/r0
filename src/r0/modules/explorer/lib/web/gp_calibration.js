"use strict";
$("gp-fit-create").onclick=()=>act("Fitting targets on an independent GP population…",async()=>{
  show(await request("gp_root_calibrate",{
    calibration_rows:number("gp-fit-rows"),calibration_seed:$("gp-fit-seed").value,
    calibration_task_id:$("gp-fit-task").value,calibration_locations:JSON.parse($("gp-fit-locations").value),...limits()
  }));
});
function gpCalibrationView(view){
  const stored=view.recipe.recipe,source=gpRootEnvelope(stored);
  const calibrated=["gp-root-calibration-v1","gp-root-calibration-query-v1"].includes(stored.version);
  let fit=stored;while(fit.version==="gp-root-calibration-query-v1")fit=fit.source;
  $("graph-population-label").hidden=!calibrated;
  $("gp-query-create").disabled=!calibrated;$("gp-query-create").dataset.unavailable=String(!calibrated);
  $("gp-query-seed").value=String(view.recipe.seed);
  $("gp-query-task").value=String(BigInt(view.recipe.task_id)+1n);
  $("gp-query-notice").textContent=calibrated?"Construct uses these exact identifiers and locations. Next uses the main seed and next task identifier with these location controls. Full history and the original fixed fit remain retained; duplicate identities or exhausted limits fail without changing the active dataset.":"Fit or open a calibrated GP-root dataset first.";
  const categorical=source && source.base.version!=="functional-causal-v1" && source.base.task.task_type!=="regression";
  const available=source && stored.version==="gp-root-dag-v1" && !categorical;
  $("gp-fit-create").disabled=!available;$("gp-fit-create").dataset.unavailable=String(!available);
  if(!source){$("gp-fit-notice").textContent="Generate or open an uncalibrated GP-root dataset first.";return;}
  if(!$("gp-fit-seed").value)$("gp-fit-seed").value=String(view.recipe.seed);
  if(!$("gp-fit-task").value){
    const next=BigInt(view.recipe.task_id)+1n;
    $("gp-fit-task").value=String(next<2n**63n?next:0n);
  }
  $("gp-fit-notice").textContent=calibrated
    ? `Fixed calibration retained from ${fit.calibration.rows} separate GP rows. Replay preserves both populations. Open the original dataset to fit another calibration. Use the generation graph population selector to inspect the independent fitting rows. Next draws an independent query and preserves this fixed fit.`
    : categorical?"Direct categorical targets require category-preserving calibration; numeric scaling is unavailable."
    : "Choose a distinct fit seed/task identity. Both populations and the complete fixed fit are retained; failure leaves the current dataset active.";
}

$("gp-query-create").onclick=()=>act("Constructing a GP query with the retained fixed fit…",async()=>{
  if($("recipe").value!==pretty(current.view.recipe.recipe))throw new Error("Unapplied archive edits retained; reopen the original dataset to change its graph.");
  show(await request("gp_root_query",{seed:$("gp-query-seed").value,task_id:$("gp-query-task").value,locations:JSON.parse($("gp-query-locations").value),...limits()}));
});
