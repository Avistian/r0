"use strict";
function prototypeConstructionNote(example) {
  if(example.topology==="network-innovations") example=example.source;
  const active=example.topology==="prototype-root-network";
  $("prototype-controls").hidden=!active;
  if(!active) return;
  $("prototype-lifecycle").value=example.lifecycle;
  $("prototype-reference").value=example.reference_rows ?? 32;
  $("prototype-rho").value=example.rho;
  $("prototype-count-rule").value=example.count_rule;
  $("prototype-max").value=example.max_prototypes;
  $("prototype-weight-mode").value=example.weights.mode;
  $("prototype-trials").value=example.weights.trials ?? 16;
  $("prototype-beta").value=example.weights.beta;
  $("prototype-probabilities").value=pretty(example.weights.probabilities);
  $("prototype-seed").value=String(example.bank_seed ?? 13);
  prototypeLifecycleControls();
  $("prototype-trials").disabled=example.weights.mode==="dirichlet-concentration-r0";
}
function prototypeLifecycleControls() {
  const batch=$("prototype-lifecycle").value==="prototype-batch-reference-v1";
  $("prototype-reference").disabled=batch;$("prototype-seed").disabled=batch;
  $("prototype-lifecycle-note").textContent=batch
    ? "Batch reference uses the task row count and seed/task identity. Next population reconstructs prototypes and all row weights; saved archive Replay is exact. Independent bank seed/reference rows are unused and recorded as null."
    : "Frozen Next retains this independent bank and draws fresh row weights. Reference rows and bank seed control its construction independently of task identity.";
}
$("prototype-lifecycle").onchange=prototypeLifecycleControls;
$("prototype-weight-mode").onchange=()=>{$("prototype-trials").disabled=$("prototype-weight-mode").value==="dirichlet-concentration-r0";};
$("prototype-apply").onclick=()=>act("Applying declared prototype conventions…",async()=>{
  const controls=JSON.parse($("construction-controls").value);
  const draft=controls.topology==="network-innovations" ? controls.source : controls;
  if(draft.topology!=="prototype-root-network") throw new Error("Select a prototype root preset first.");
  draft.lifecycle=$("prototype-lifecycle").value;
  const batch=draft.lifecycle==="prototype-batch-reference-v1";
  if(batch && controls.topology==="network-innovations") throw new Error("Batch reference with internal innovations remains required and is not yet supported. Choose a plain prototype network.");
  draft.reference_rows=batch ? null : number("prototype-reference");draft.rho=number("prototype-rho");
  draft.count_rule=$("prototype-count-rule").value;draft.max_prototypes=number("prototype-max");
  draft.weights.mode=$("prototype-weight-mode").value;
  draft.weights.trials=draft.weights.mode==="dirichlet-concentration-r0" ? null : number("prototype-trials");
  draft.weights.beta=number("prototype-beta");draft.weights.probabilities=JSON.parse($("prototype-probabilities").value);
  draft.bank_seed=batch ? null : $("prototype-seed").value;
  $("construction-controls").value=pretty(controls);
});
