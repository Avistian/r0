"use strict";
function taskControls(recipe) {
  if(recipe.version==="gp-root-intervention-v1")return taskControls(gpRootEnvelope(recipe).base);
  if(["gp-root-calibration-v1","gp-root-calibration-query-v1"].includes(recipe.version))return taskControls(recipe.original);
  if(recipe.version === "gp-root-dag-v1")return taskControls(recipe.base);
  if(recipe.request?.base)return taskControls(recipe.request.base);
  return recipe.runtime?.task || recipe.graph?.runtime?.task || recipe.request?.source?.task || recipe.source?.task || recipe.recipe?.task || recipe.request?.task || recipe.task || recipe;
}
