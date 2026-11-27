"use strict";
function isGPTargetEnvironment(){return current?.view?.recipe?.recipe?.version==="gp-column-target-recipe-v1";}
function gpTargetEditAvailability(){
  const environment=isGPTargetEnvironment();
  $("gp-edit-section").hidden=!isGPColumns();
  for(const id of ["next","curve"]){
    const button=$(id);
    if(environment){button.disabled=true;button.dataset.unavailable="true";button.dataset.gpEnvironment="true";}
    else if(button.dataset.gpEnvironment==="true"){button.disabled=false;button.dataset.unavailable="false";delete button.dataset.gpEnvironment;}
  }
  if(environment){
    $("next").textContent="Fixed comparison environment";
    $("curve-description").textContent="This comparison retains fixed population identity and both target equations. Dedicated environment reference curves and further linked target editing remain required follow-ons. Reopen the original population to create another comparison or sample a new GP population.";
  }
}
$("gp-edit-original").onclick=()=>act("Loading original target…",async()=>{
  if(!isGPColumns())throw new Error("Select an original GP population.");
  const target=current.view.recipe.recipe.source.target;
  $("gp-edit-family").value=target.family;$("gp-edit-json").value=pretty(target);
});
$("gp-edit-preset").onclick=()=>act("Preparing authored target draft…",async()=>{
  if(!isGPColumns())throw new Error("Select an original GP population.");
  const request=current.view.recipe.recipe.source.request, family=$("gp-edit-family").value;
  const n=request.columns,o=request.outputs,h=family==="neural"?number("gp-edit-hidden"):n;
  if(!Number.isInteger(h)||h<1||h>64)throw new Error("Starting draft hidden width must be 1..64.");
  const weights=Array.from({length:h},(_,i)=>Array.from({length:o},(_,k)=>((i+k)%2?-1:1)/Math.sqrt(h)));
  const target={version:"gp-column-target-r0-v1",family,weights,bias:Array(o).fill(0),quadratic:[],hidden:[],hidden_bias:[]};
  if(family==="quadratic")target.quadratic=Array.from({length:o},(_,k)=>Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>i===j?(k%2?-0.25:0.25):0)));
  if(family==="neural"){
    target.hidden=Array.from({length:n},(_,i)=>Array.from({length:h},(_,j)=>(i===j%n?1:-0.25)/Math.sqrt(n)));
    target.hidden_bias=Array.from({length:h},(_,i)=>0.1*i);
  }
  $("gp-edit-json").value=pretty(target);
});
$("gp-edit-mode").onchange=()=>{
  const fresh=$("gp-edit-mode").value==="fresh";$("gp-edit-seed").disabled=!fresh;
  $("gp-edit-mode-note").textContent=fresh?"Both sides use the same newly keyed feature innovations and adapter draws at the original fixed locations/kernels. Before is the original target on that fresh environment, not the active dataset. Original population identity and calibration inputs remain recorded.":"Before exactly reproduces the retained original inputs, target scores, targets and split. Both sides reuse original adapter noise/label/split streams; after changes only the frozen target equation and its independent calibration.";
};
$("gp-edit-mode").onchange();
$("gp-edit-create").onclick=()=>act("Saving guarded GP target comparison…",async()=>{
  const fresh=$("gp-edit-mode").value==="fresh", seed=$("gp-edit-seed").value;
  if(fresh && !/^[0-9]+$/.test(seed))throw new Error("Fresh seed must be an exact nonnegative decimal integer.");
  const result=await request("gp_target_edit",{target_json:$("gp-edit-json").value,mode:$("gp-edit-mode").value,fresh_seed:fresh?seed:null,expected_episode_id:current.view.episode_id,...limits()});
  current=result;interventionTrace=null;$("intervention-traces").hidden=true;showIntervention();
});
