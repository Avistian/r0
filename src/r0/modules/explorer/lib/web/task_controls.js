"use strict";
// Resolve the displayed task without rewriting retained construction records.
function taskControls(recipe) { if(isCalibratedPopulation(recipe))return taskControls(recipe.original); if(recipe.request?.base)return taskControls(recipe.request.base); return recipe.runtime?.task || recipe.graph?.runtime?.task || recipe.request?.source?.task || recipe.source?.task || recipe.recipe?.task || recipe.request?.task || recipe.task || recipe; }
