"use strict";
function fillInnovationControls(example) {
  const law=(example?.topology==="sampled-converted"?example.request:example)?.pipeline_prior?.innovation;
  $("innovation-controls").hidden=!law;
  if(!law)return;
  $("innovation-family").value=law.families.length===1?law.families[0]:"all";
  $("innovation-factor").value=law.factors.length===1?law.factors[0]:"both";
  $("innovation-scope").value=law.scale_scope;
  $("innovation-low").value=law.structural_scale[0];$("innovation-high").value=law.structural_scale[1];
  for(const name of ["degrees","strengths","powers"])$("innovation-"+name).value=law[name].join(", ");
  $("innovation-truncation").value=law.truncation.length===1?(law.truncation[0]===null?"none":String(law.truncation[0])):"mixed";
}
$("innovation-apply").onclick=()=>act("Updating innovation draft…",async()=>{
  const draft=JSON.parse($("construction-controls").value),law=(draft.topology==="sampled-converted"?draft.request:draft).pipeline_prior?.innovation;
  if(!law)throw new Error("Choose a sampled innovation pipeline first.");
  const numeric=id=>{const s=$(id).value.trim();if(!s||!Number.isFinite(Number(s)))throw new Error("Enter a finite numeric value.");return Number(s);};
  const catalog=name=>$("innovation-"+name).value.split(",").map(s=>{if(!s.trim()||!Number.isFinite(Number(s)))throw new Error("Catalogs require comma-separated finite numbers.");return Number(s);});
  law.families=$("innovation-family").value==="all"?["gaussian","uniform","student_t","mixture","ball"]:[$("innovation-family").value];
  law.factors=$("innovation-factor").value==="both"?["identity","gaussian"]:[$("innovation-factor").value];
  law.scale_scope=$("innovation-scope").value;law.structural_scale=[numeric("innovation-low"),numeric("innovation-high")];
  for(const name of ["degrees","strengths","powers"])law[name]=catalog(name);
  const bound=$("innovation-truncation").value;law.truncation=bound==="mixed"?[null,2,4,8]:[bound==="none"?null:Number(bound)];
  $("construction-controls").value=pretty(draft);
});
