"use strict";
function prototypeConstructionNote(example) {
  if(example.topology==="network-innovations") example=example.source;
  const active=example.topology==="prototype-root-network";
  $("prototype-controls").hidden=!active;
  if(!active) return;
  $("prototype-lifecycle").value=example.lifecycle;
  $("prototype-reference").value=example.reference_rows;
  $("prototype-rho").value=example.rho;
  $("prototype-count-rule").value=example.count_rule;
  $("prototype-max").value=example.max_prototypes;
  $("prototype-weight-mode").value=example.weights.mode;
  $("prototype-trials").value=example.weights.trials ?? 16;
  $("prototype-beta").value=example.weights.beta;
  $("prototype-probabilities").value=pretty(example.weights.probabilities);
  $("prototype-seed").value=String(example.bank_seed);
  $("prototype-trials").disabled=example.weights.mode==="dirichlet-concentration-r0";
}
$("prototype-weight-mode").onchange=()=>{$("prototype-trials").disabled=$("prototype-weight-mode").value==="dirichlet-concentration-r0";};
$("prototype-apply").onclick=()=>act("Applying declared prototype conventions…",async()=>{
  const controls=JSON.parse($("construction-controls").value);
  const draft=controls.topology==="network-innovations" ? controls.source : controls;
  if(draft.topology!=="prototype-root-network") throw new Error("Select a prototype root preset first.");
  draft.lifecycle=$("prototype-lifecycle").value;
  draft.reference_rows=number("prototype-reference");draft.rho=number("prototype-rho");
  draft.count_rule=$("prototype-count-rule").value;draft.max_prototypes=number("prototype-max");
  draft.weights.mode=$("prototype-weight-mode").value;
  draft.weights.trials=draft.weights.mode==="dirichlet-concentration-r0" ? null : number("prototype-trials");
  draft.weights.beta=number("prototype-beta");draft.weights.probabilities=JSON.parse($("prototype-probabilities").value);
  draft.bank_seed=$("prototype-seed").value;
  $("construction-controls").value=pretty(controls);
});
