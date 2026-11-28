"use strict";
$("gp-fit-create").onclick=()=>act("Fitting targets on an independent GP population…",async()=>{
  show(await request("gp_root_calibrate",{
    calibration_rows:number("gp-fit-rows"),calibration_seed:$("gp-fit-seed").value,
    calibration_task_id:$("gp-fit-task").value,calibration_locations:JSON.parse($("gp-fit-locations").value),...limits()
  }));
});
function gpCalibrationView(view){
  const stored=view.recipe.recipe,source=gpRootEnvelope(stored);
  $("graph-population-label").hidden=stored.version!=="gp-root-calibration-v1";
  const categorical=source && source.base.version!=="functional-causal-v1" && source.base.task.task_type!=="regression";
  const available=source && stored.version==="gp-root-dag-v1" && !categorical;
  $("gp-fit-create").disabled=!available;$("gp-fit-create").dataset.unavailable=String(!available);
  if(!source){$("gp-fit-notice").textContent="Generate or open an uncalibrated GP-root dataset first.";return;}
  if(!$("gp-fit-seed").value)$("gp-fit-seed").value=String(view.recipe.seed);
  if(!$("gp-fit-task").value){
    const next=BigInt(view.recipe.task_id)+1n;
    $("gp-fit-task").value=String(next<2n**63n?next:0n);
  }
  $("gp-fit-notice").textContent=stored.version==="gp-root-calibration-v1"
    ? `Fixed calibration retained from ${stored.calibration.rows} separate GP rows. Replay preserves both populations. Open the original dataset to fit another calibration. Use the generation graph population selector to inspect the independent fitting rows. Fixed-fit Next remains pending.`
    : categorical?"Direct categorical targets require category-preserving calibration; numeric scaling is unavailable."
    : "Choose a distinct fit seed/task identity. Both populations and the complete fixed fit are retained; failure leaves the current dataset active.";
}
