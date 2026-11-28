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
    if(!note){note=document.createElement("p");note.id=id+"-gp-root-pending";note.className="muted gp-root-pending-notice";if(id==="next")button.parentElement.after(note);else button.before(note);}
    note.hidden=false;note.textContent=message;
  }
}
function gpRootCapabilities(view){
  const recipe=view.recipe.recipe;if(!gpRootEnvelope(recipe))return;
  if(["gp-root-calibration-v1","gp-root-calibration-query-v1","gp-root-intervention-v1"].includes(recipe.version))
    for(const id of ["rows","context","features","active","task-type","classes","difficulty"])
      pendingGPRootButton(id,"The retained fixed fit requires unchanged graph and task geometry. Use an original GP-root construction to define a different task.");
  if(recipe.version==="gp-root-intervention-v1"){
    const message="Open the original GP source to create another change. Nested intervention editing remains pending; replay and inspection retain this result.";
    for(const id of ["next","gp-query-create","gp-root-load","mechanism-load","intervention-load","intervention-add","intervention-create"])
      pendingGPRootButton(id,message,id==="intervention-load");
    $("intervention-controls").hidden=true;
  }
  const policy="Unavailable for GP-root datasets: sampled intervention policies require retained-population integration and remain required work.";
  for(const id of ["policy-draft","policy-add-location","policy-create"])
    pendingGPRootButton(id,policy,id==="policy-draft");
  pendingGPRootButton("pair-create","Unavailable for GP-root datasets: generic task-only comparisons do not retain the GP population. Population-aware comparisons remain required pending integration.",true);

}
