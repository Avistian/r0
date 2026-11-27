"use strict";
function isTypedPolicy(recipe){return recipe?.version==="sampled-typed-intervention-policy-v1";}
function typedPolicyFields(){
  if(!$("typed-policy-base").options.length)$("typed-policy-base").replaceChildren(...Object.keys(constructionExamples).filter(name=>/Native|native|Fitted structural|Sampled fitted parameters|XGBoost|Typed.*converter/.test(name)).map(name=>new Option(name,name)));
  $("typed-policy-fields").hidden=$("policy-profile").value!=="explicit-typed-policy-r0-v1";
}
function typedPolicyDraft(draft){
  Object.assign(draft,{mode:$("intervention-mode").value,population:$("intervention-population").value,count:1,eligibility:[]});
  draft[draft.kind==="hard"?"locations":"parameter_laws"]=[];
}
$("policy-profile").onchange=typedPolicyFields;
$("policy-law").onchange=()=>{
  if($("policy-law").value==="log-uniform-r0"){$("policy-law-location").value="0.1";$("policy-law-scale").value="2";}
};
$("policy-add-location").onclick=()=>act("Adding explicit typed policy address…",async()=>{
  if(!interventionInfo)throw new Error("Prepare causal intervention controls and select a node first.");
  const draft=JSON.parse($("policy-json").value);
  if(draft.profile!=="explicit-typed-policy-r0-v1")throw new Error("Choose the typed r0 profile and prepare a draft first.");
  const node=Number($("intervention-node").value);
  if(draft.kind==="soft"){
    const family=$("policy-law").value;
    const law={node,path:JSON.parse($("policy-parameter-path").value),family,location:number("policy-law-location"),scale:number("policy-law-scale"),choices:family==="choice-r0"?JSON.parse($("policy-law-choices").value):[]};
    draft.parameter_laws.push(law);
  }else{
    const probabilities=JSON.parse($("policy-probabilities").value);
    let address;
    if(selectedConverter())address={node,kind:"capture",capture:selectedConverter().name,location:$("intervention-converter-location").value,probabilities};
    else if(interventionInfo.native)address={node,kind:"native-variable",variable:interventionInfo.vector?Number($("intervention-variable").value):0,location:"native-value",probabilities};
    else address={node,kind:"numerical",channels:JSON.parse($("intervention-channels").value),location:$("intervention-location").value};
    draft.locations.push(address);
  }
  draft.eligibility=[...new Set([...draft.eligibility,node])];
  $("policy-json").value=pretty(draft);
  $("typed-policy-note").textContent="Draft updated. Review eligibility/count, locations and declared laws before sampling. Worker validation preserves the accepted dataset and previous comparison on any invalid or over-budget request. Quote large stream seeds as decimal strings to retain every digit.";
});

$("typed-policy-load-base").onclick=()=>act("Loading retained base construction…",async()=>{
  const selected=$("typed-policy-base").value;
  if(!constructionExamples[selected])throw new Error("Select an available retained graph construction.");
  $("construction-panel").open=true;$("construction-family").value=selected;$("construction-family").onchange();
  $("typed-policy-note").textContent="Review the loaded construction controls and use Construct mechanism and save dataset. Then prepare causal intervention controls and add the desired typed policy addresses. No dataset or comparison has changed yet.";
});

function policyActiveControls(){
  const active=["sampled-intervention-policy-v1","sampled-typed-intervention-policy-v1"].includes(current?.view?.recipe?.recipe?.version);
  for(const id of ["mechanism-load","intervention-load","policy-create"]){$(id).disabled=active;$(id).dataset.unavailable=String(active);}
  if(active){
    $("workbench-intro").textContent="This frozen policy retains its original graph and realized interventions. Inspect generation stages, replay or advance query rows without resampling the policy. Direct edits to a realized policy envelope remain pending; reopen the original before.zip to edit its mechanism or policy settings and save a new comparison.";
    $("intervention-notice").textContent="Nested policy construction is unavailable. Reopen the comparison's before.zip to edit eligibility, typed addresses, probabilities, parameter laws or stream seeds. The previous saved comparison remains intact.";
  }
}
