"use strict";
let gpPopulationMatrices=null;
function clearGPPopulationMatrices(){gpPopulationMatrices=null;$("gp-covariance-panel").hidden=true;$("gp-matrix-plot").replaceChildren();$("gp-matrix-table").replaceChildren();}
function showGPPopulationMatrices(graph, selectedRecipe=null, origin="active dataset"){
  clearGPPopulationMatrices();
  if(!["gp_column_population","gp_target_environment","gp_root_dag","gp_root_calibration","gp_root_fit"].includes(graph.kind)||!graph.shared_matrices?.length)return;
  const recipe=selectedRecipe || current.view.recipe.recipe;
  let fitting=recipe;while(fitting.version==="gp-root-calibration-query-v1")fitting=fitting.source;
  let base=["gp-root-calibration-v1","gp-root-calibration-query-v1"].includes(recipe.version)?(graph.population_role==="fit"?fitting.fit:recipe.original):recipe,depth=0;
  while(base.version==="gp-column-target-link-v1" && depth<8){base=base.parent;depth++;}
  const environment=base.version==="gp-column-target-recipe-v1", source=environment?base.comparison:base.source;
  if(!source?.draw)return;
  gpPopulationMatrices={...(graph.population_role?{population_role:graph.population_role}:{}),population_id:graph.population_id,matrices:graph.shared_matrices,locations:source.draw.locations,innovations:source.draw.innovations,values:source.draw.values,...(base.version==="gp-root-dag-v1"?{root_input_bindings:base.bindings}:{}),...(environment?{environment:{origin,mode:source.mode,fresh_seed:source.fresh_seed,side:depth?"linked-replacement":base.side,lineage_depth:depth,recipe_version:recipe.version}}:{})};
  $("gp-matrix-kernel").replaceChildren(...graph.shared_matrices.map((m,i)=>new Option(`Kernel ${i} · feature columns ${m.kernel_columns.join(", ")}`,i)));
  $("gp-matrix-row").value=0;$("gp-matrix-column").value=0;$("gp-matrix-kind").value="covariance";$("gp-covariance-panel").hidden=false;drawGPPopulationMatrix();
}
function drawGPPopulationMatrix(){
  if(!gpPopulationMatrices)return;
  const selected=gpPopulationMatrices.matrices[Number($("gp-matrix-kernel").value)],kind=$("gp-matrix-kind").value;
  const square=["covariance","factor"].includes(kind),matrix=square?selected[kind]:gpPopulationMatrices[kind];
  const bounded=(id,max)=>{const n=Number($(id).value),v=Number.isInteger(n)?Math.max(0,Math.min(max,n)):0;$(id).value=v;$(id).max=max;return v;};
  const start=bounded("gp-matrix-row",matrix.length-1),column=bounded("gp-matrix-column",matrix[0].length-1),rows=matrix.slice(start,start+16).map(r=>r.slice(column,column+16));
  const columns=rows[0].map((_,j)=>column+j),rowIds=rows.map((_,i)=>selected.row_ids[start+i]);
  const heading=square?"Original population row":kind==="locations"?"Latent coordinate":"Feature column";
  const node={label:$("gp-matrix-kind").selectedOptions[0].textContent,values:rows,shape:[matrix.length,matrix[0].length],column_indices:columns,parameters:{axes:`Original population rows ${rowIds.join(", ")}; ${heading.toLowerCase()} positions ${columns.join(", ")}.`}};
  // Explicit axes avoid treating a covariance row as a selected task-row trace.
  const svg=element("svg",{viewBox:"0 0 480 340",role:"img","aria-label":node.label});
  const magnitude=Math.max(...rows.flat().map(Math.abs))||1;
  svg.append(element("text",{x:55,y:22,"font-size":13},node.label));
  rows.forEach((row,i)=>row.forEach((v,j)=>{const alpha=Math.abs(v)/magnitude,color=v>=0?[23,130,107]:[215,137,73];const cell=element("rect",{x:55+j*400/columns.length,y:48+i*250/rows.length,width:400/columns.length,height:250/rows.length,fill:`rgb(${color.map(c=>Math.round(255+alpha*(c-255))).join(",")})`});cell.append(element("title",{},`Original population row ${rowIds[i]}, ${heading} ${columns[j]}: ${v}`));svg.append(cell);}));
  for(const [x,y,text] of [[12,58,rowIds[0]],[12,298,rowIds.at(-1)],[55,320,columns[0]],[440,320,columns.at(-1)]])svg.append(element("text",{x,y,"font-size":11},String(text)));
  $("gp-matrix-plot").replaceChildren(svg);chartExports("gp-matrix-plot",svg,"gp-population-matrix");
  table("gp-matrix-table",["Original population row",...columns.map(j=>`${heading} ${j}`)],rows.map((row,i)=>[rowIds[i],...row]));
  const environment=gpPopulationMatrices.environment,rootNote=gpPopulationMatrices.root_input_bindings?"GP source columns feed the retained root bindings; observed DAG channels are shown in the generation trace. ":"";
  $("gp-matrix-note").textContent=`${gpPopulationMatrices.population_role==="fit"?"Independent fitting population. ":""}${rootNote}${environment?`${environment.origin} · ${environment.mode} innovations · ${environment.side} · lineage ${environment.lineage_depth}. `:""}Population ${gpPopulationMatrices.population_id}. Full shape ${matrix.length} × ${matrix[0].length}; window rows ${rowIds[0]}–${rowIds.at(-1)}, columns ${columns[0]}–${columns.at(-1)}. Change the offsets to inspect every entry. Kernel applies to feature columns ${selected.kernel_columns.join(", ")}. Green positive, orange negative, white zero; scale uses this window. Complete values are in the JSON download.`;
}
for(const id of ["gp-matrix-kernel","gp-matrix-kind","gp-matrix-row","gp-matrix-column"])$(id).onchange=drawGPPopulationMatrix;
$("gp-matrix-json").onclick=()=>{if(gpPopulationMatrices)download("gp-population-matrices.json",new Blob([JSON.stringify(gpPopulationMatrices,null,2)],{type:"application/json"}));};
