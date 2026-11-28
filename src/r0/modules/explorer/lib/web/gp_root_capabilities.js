"use strict";
function clearGPRootCapabilities(){
  document.querySelectorAll('[data-gp-root-pending="true"]').forEach(button=>{
    button.disabled=false;button.dataset.unavailable="false";
    delete button.dataset.gpRootPending;button.removeAttribute("title");
  });
  document.querySelectorAll('.gp-root-pending-notice').forEach(note=>note.hidden=true);
}
function pendingGPRootButton(id,message,notice=false){
  const button=$(id);button.disabled=true;button.dataset.unavailable="true";
  button.dataset.gpRootPending="true";button.title=message;
  if(notice){
    let note=$(id+"-gp-root-pending");
    if(!note){note=document.createElement("p");note.id=id+"-gp-root-pending";note.className="muted gp-root-pending-notice";button.before(note);}
    note.hidden=false;note.textContent=message;
  }
}
function gpRootCapabilities(view){
  const recipe=view.recipe.recipe;if(!gpRootEnvelope(recipe))return;
  const intervention="Unavailable for GP-root datasets: interventions require retained-population integration. This remains required work; the current dataset and its GP source are preserved.";
  for(const id of ["intervention-load","intervention-add","intervention-create"])
    pendingGPRootButton(id,intervention,id==="intervention-load");
  $("intervention-controls").hidden=true;
  const policy="Unavailable for GP-root datasets: sampled intervention policies require retained-population integration and remain required work.";
  for(const id of ["policy-draft","policy-add-location","policy-create"])
    pendingGPRootButton(id,policy,id==="policy-draft");
  pendingGPRootButton("curve","Unavailable for GP-root datasets: original-row causal reference curves remain required pending retained-population integration.",true);
  pendingGPRootButton("pair-create","Unavailable for GP-root datasets: generic task-only comparisons do not retain the GP population. Population-aware comparisons remain required pending integration.",true);
  if(recipe.version==="gp-root-calibration-v1")
    pendingGPRootButton("next","Unavailable for calibrated GP-root datasets: fixed-calibration population continuation remains required. Replay preserves the current population and fit.",true);
}
