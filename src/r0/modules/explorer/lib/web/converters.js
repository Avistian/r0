"use strict";
const converterModes=["neighbor_id","neighbor_disc","neighbor_func","neighbor_int","softmax_id","softmax_disc","softmax_int"];
function converterModeNote() {
  const mode=$("converter-mode").value;
  $("converter-mode-note").textContent=mode.endsWith("_id") ? "The captured category may change while descendant state stays the original latent slice. A category has no unique inverse latent value." : mode.endsWith("_int") ? "The category index is broadcast across every input-span channel before structural noise. This is not one-hot encoding." : mode==="neighbor_func" ? "A frozen cheap source function transforms selected prototypes. Its fit population retains repeated centers with their row multiplicities." : "Nearest modes use retained prototypes and a frozen p-distance. Softmax modes use frozen standardized logits, explicit emission uniforms, and transformed RandomPoints embeddings. Representations remain distinct from category captures.";
}
function fillConverterControls(controls) {
  const active=controls?.topology==="sampled-converted";
  $("converter-controls").hidden=!active;
  if(!active)return;
  const request=controls.request, modes=request.converter_prior.modes;
  $("converter-mode").value=modes.length===7?"all":modes.length===4&&modes.every(x=>x.startsWith("neighbor_"))?"neighbor":modes.length===3&&modes.every(x=>x.startsWith("softmax_"))?"softmax":modes.length===1?modes[0]:"custom";
  $("converter-fit-rows").value=request.calibration_rows;
  $("converter-features").value=pretty(request.vocabularies);
  $("converter-target").value=pretty(request.target_vocabulary);
  converterModeNote();
}
$("converter-mode").onchange=converterModeNote;
$("converter-apply").onclick=()=>act("Updating typed converter draft…",async()=>{
  const draft=JSON.parse($("construction-controls").value);
  if(draft.topology!=="sampled-converted")throw new Error("Choose a typed converter construction first.");
  const features=JSON.parse($("converter-features").value),target=JSON.parse($("converter-target").value),rows=Number($("converter-fit-rows").value);
  const valid=v=>Array.isArray(v)&&(v.length===0||v.length>=2)&&v.every(x=>typeof x==="string"&&x.length)&&new Set(v).size===v.length;
  if(!Array.isArray(features)||!features.length||!features.every(valid)||!valid(target)||!Number.isSafeInteger(rows)||rows<1)throw new Error("Use valid vocabulary arrays and a positive integer calibration row count.");
  const mode=$("converter-mode").value;
  if(mode!=="custom")draft.request.converter_prior.modes=mode==="all"?converterModes:mode==="neighbor"||mode==="softmax"?converterModes.filter(x=>x.startsWith(mode+"_")):[mode];
  Object.assign(draft.request,{vocabularies:features,target_vocabulary:target,calibration_rows:rows});
  $("construction-controls").value=pretty(draft);
});
