"use strict";
function taskControls(recipe) {
  if(recipe.version==="population-xgboost-graph-v1")return taskControls(recipe.base);
  if(isCalibratedArchive(recipe))return taskControls(recipe.original);
  if(recipe.version==="recursive-population-graph-intervention-v1")return taskControls(recipe.change.base);
  if(recipe.version === "gp-root-calibration-v1")return taskControls(recipe.original);
  if(recipe.version === "gp-root-dag-v1")return taskControls(recipe.base);
  if(recipe.request?.base)return taskControls(recipe.request.base);
  return recipe.runtime?.task || recipe.graph?.runtime?.task || recipe.request?.source?.task || recipe.source?.task || recipe.recipe?.task || recipe.request?.task || recipe.task || recipe;
}
