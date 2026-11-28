"use strict";
function topologySizeVisibility(){
  const retained=$("topology-size").value==="retained";
  $("topology-parent-label").hidden=!retained;
  for(const id of ["nodes","log-low","log-high","round","shape","rate","components","transform"])$("topology-"+id).parentElement.hidden=retained;
}
$("topology-size").onchange=topologySizeVisibility;
function topologyDraft() {
  const draft=JSON.parse($("construction-controls").value);
  if(!["joint-topology-hyperprior","authored-topology-hyperprior","typed-topology-hyperprior"].includes(draft.topology)) throw new Error("Choose a Topology hyperprior construction first.");
  return draft;
}
function topologyLoad() {
  const draft=topologyDraft(), req=draft.request, top=req.topology;
  $("topology-engine").value=draft.engine;
  const composed=req.version==="typed-parent-topology-request-r0-v1";
  $("topology-size").querySelector('[value="retained"]').disabled=!composed;
  $("topology-size").value=top.version==="fixed-parent-topology-r0-v1"?"retained":top.version==="topology-hyperprior-r0-v1"?"log":"fixed";
  $("topology-parents").value=pretty(top.parents||[]);
  $("topology-parent-composition").hidden=!composed;
  if(composed)$("topology-composition").value=pretty(req.composition);
  topologySizeVisibility();
  for(const [id,key] of [["nodes","nodes"],["log-low","log_nodes_low"],["log-high","log_nodes_high"],["round","rounding"],["shape","gamma_shape"],["rate","gamma_rate"],["components","components"],["transform","probability_transform"]]) if(top[key]!==undefined)$("topology-"+id).value=top[key];
  const typed=draft.topology==="typed-topology-hyperprior", joint=draft.topology==="joint-topology-hyperprior"||typed;
  $("topology-typed-controls").hidden=!typed;
  $("topology-fixed-width").hidden=typed;
  $("topology-noise-control").hidden=typed;
  if(typed){
    $("topology-typed-family").textContent="Mechanism family: "+req.family;
    for(const [id,key] of [["width-prior","width_prior"],["native-roles","native"],["converted-roles","converted"],["pipeline","pipeline"]])$("topology-"+id).value=pretty(req[key]);
    $("topology-subsample").checked=req.subsample;
  }
  $("topology-joint-controls").hidden=!joint;
  if(joint){
    for(const [id,key] of [["mode","mode"],["reuse","categorical_policy"],["global","global_policy"],["width","width"],["calibration","calibration_rows"]])$("topology-"+id).value=req[key];
    $("topology-functions").value=req.function_prior.functions;
    $("topology-function-prior").value=pretty(req.function_prior);
    $("topology-root-prior").value=pretty(req.root_prior);
    $("topology-noise").value=pretty(req.noise);
  }
  $("topology-summary").textContent=pretty({declared_causal_caps:req.limits,raw_caps:req.raw_limits,semantics:req.mechanism_sharing||"joint source sampled functions"});
  $("topology-status").textContent="Draft loaded. Apply changes explicitly; Construct performs guarded validation and saves only on success. Causal caps are not raised automatically.";
}
$("topology-load").onclick=()=>act("Loading topology draft…",async()=>topologyLoad());
$("topology-apply").onclick=()=>act("Updating topology draft…",async()=>{
  const draft=topologyDraft(), req=draft.request, numeric=id=>{
    const value=Number($("topology-"+id).value);
    if(!Number.isFinite(value))throw new Error("Topology controls must be finite.");return value;
  };
  if($("topology-size").value==="retained")req.topology={parents:JSON.parse($("topology-parents").value),version:"fixed-parent-topology-r0-v1"};
  else {
  req.topology={gamma_shape:numeric("shape"),gamma_rate:numeric("rate"),components:numeric("components"),probability_transform:$("topology-transform").value};
  if($("topology-size").value==="fixed")Object.assign(req.topology,{nodes:numeric("nodes"),version:"fixed-size-topology-hyperprior-r0-v1"});
  else Object.assign(req.topology,{log_nodes_low:numeric("log-low"),log_nodes_high:numeric("log-high"),rounding:$("topology-round").value,version:"topology-hyperprior-r0-v1"});
  }
  if(req.version==="typed-parent-topology-request-r0-v1")req.composition=JSON.parse($("topology-composition").value);
  draft.engine=$("topology-engine").value;
  if(["joint-topology-hyperprior","typed-topology-hyperprior"].includes(draft.topology)){
    req.mode=$("topology-mode").value;req.categorical_policy=$("topology-reuse").value;req.global_policy=$("topology-global").value;
    if(draft.topology==="joint-topology-hyperprior")req.width=numeric("width");
    req.calibration_rows=numeric("calibration");
    req.function_prior=JSON.parse($("topology-function-prior").value);req.function_prior.functions=$("topology-functions").value;
    req.root_prior=JSON.parse($("topology-root-prior").value);
    if(draft.topology==="joint-topology-hyperprior")req.noise=JSON.parse($("topology-noise").value);
    else {
      for(const [id,key] of [["width-prior","width_prior"],["native-roles","native"],["converted-roles","converted"],["pipeline","pipeline"]])req[key]=JSON.parse($("topology-"+id).value);
      req.subsample=$("topology-subsample").checked;
    }
  }
  $("construction-controls").value=pretty(draft);$("topology-status").textContent="Draft changed. Construct validates whole finite support before drawing; active task is unchanged until success.";
});
$("topology-adopt-caps").onclick=()=>act("Applying explicit topology caps…",async()=>{
  $("exploration-limits").value=pretty(topologyDraft().request.limits);
  $("topology-status").textContent="Draft causal caps copied to the caller controls. Time and output-byte limits remain separate controls.";
});
$("topology-records").onclick=()=>act("Reading retained topology records…",async()=>{
  if(!current?.recipe_json)throw new Error("Construct or open a topology dataset first.");
  const recipe=JSON.parse(current.recipe_json).recipe;
  if(!recipe.topology?.initial && recipe.topology?.version!=="retained-parent-topology-r0-v1")throw new Error("The active dataset is not a topology hyperprior graph.");
  $("topology-archive").value=current.recipe_json;
  $("topology-summary").textContent=pretty({version:recipe.version,request:recipe.request,topology:recipe.topology.topology||recipe.topology.prior,topology_engine:recipe.topology.initial?.engine||recipe.initial?.[0].engine,composition:recipe.request.composition||null,mechanism_engines:recipe.graph?.initial.map(s=>s.engine),raw_draw_count:recipe.graph?.raw?.length ?? recipe.graph?.raw_draws?.length ?? null,construction:"Exact integer snapshot payloads are retained in the raw text below and ZIP; this summary is descriptive."});
});
