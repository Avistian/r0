"use strict";
function sourceRootConstructionNote(example) {
  if(example.topology==="network-innovations") example=example.source;
  const active=example.topology==="source-root-network";
  $("source-root-controls").hidden=!active;
  if(!active) return;
  const modes=example.modes || [];
  $("source-root-mode").value=modes.length && modes.every(m=>m===modes[0]) ? modes[0] : "custom";
  $("source-root-rows").value=example.calibration_rows || 256;
  $("source-root-seed").value=String(example.root_seed);
  $("source-root-rows").disabled=$("source-root-mode").value==="identity";
}
$("source-root-mode").onchange=()=>{$("source-root-rows").disabled=$("source-root-mode").value==="identity";};
$("source-root-apply").onclick=()=>act("Applying explicit root controls…",async()=>{
  const controls=JSON.parse($("construction-controls").value);
  const draft=controls.topology==="network-innovations" ? controls.source : controls;
  if(draft.topology!=="source-root-network") throw new Error("Select a source-root network preset first.");
  const mode=$("source-root-mode").value;
  if(mode!=="custom") draft.modes=Array(draft.profile.width).fill(mode);
  if(mode==="identity") draft.calibration_rows=0;
  else draft.calibration_rows=number("source-root-rows");
  draft.root_seed=$("source-root-seed").value;
  $("construction-controls").value=pretty(controls);
});

function showSourceRootEditor() {
  const root=(networkDraft?.base || networkDraft)?.root;
  const active=["source-root-base-v1","sampled-source-root-base-v1"].includes(root?.version);
  $("source-root-editor").hidden=!active;
  if(!active) return;
  $("source-root-edit-policy").value="retain";
  $("source-root-json").readOnly=true;
  $("source-root-json").value=pretty({law:root.law,calibration:root.calibration,version:"source-root-base-v1"});
  $("source-root-provenance").textContent=pretty(root);
}
$("source-root-edit-policy").onchange=()=>{$("source-root-json").readOnly=$("source-root-edit-policy").value!=="authored";};
$("source-root-edit-apply").onclick=()=>act("Validating frozen root draft…",async()=>{
  if($("source-root-edit-policy").value!=="authored") throw new Error("Choose Create an authored root copy to change the frozen law. The original root remains retained.");
  if($("network-recipe").value!==pretty(networkDraft)) throw new Error("Complete recipe has unapplied edits; both drafts retained.");
  const text=$("source-root-json").value, candidate=JSON.parse(JSON.stringify(networkDraft));
  const root=JSON.parse(text);
  if(root.version!=="source-root-base-v1") throw new Error("Authored root edits require source-root-base-v1; sampled receipts cannot be edited.");
  (candidate.base || candidate).root=root;
  const planText=networkNoiseMode ? $("network-noise-plan").value : null;
  const validated=await request(networkNoiseMode ? "network_noise_draft" : "network_draft", networkNoiseMode ? {recipe:candidate,innovation_plan:JSON.parse(planText),...limits()} : networkEditRequest(candidate));
  if((networkNoiseMode && $("network-noise-plan").value!==planText) || $("source-root-json").value!==text || $("network-recipe").value!==pretty(networkDraft)) throw new Error("Draft changed during validation; text retained. Apply again.");
  networkDraft=validated.recipe;
  if(networkNoiseMode) $("network-noise-compiled").textContent=pretty(validated.compiled);
  $("network-recipe").value=pretty(networkDraft);
  $("source-root-provenance").textContent=pretty((networkDraft.base || networkDraft).root);
  $("workbench-note").textContent="Validated authored root draft applied. Generate and save the linked edit; the original sampled root stays in the original archive.";
});
