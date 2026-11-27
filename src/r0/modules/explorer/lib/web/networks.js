"use strict";
let networkNoiseMode = false;
let networkMatchingEdit = false;
let networkDraft = null, networkBlockDrafts = new Map(), networkSelectedBlock = null;
function isPrototypeBatch(recipe) { return ["prototype-batch-network-request-v1","prototype-batch-network-realized-v1"].includes(recipe?.version); }
function isStandaloneNetwork(recipe) { return isPrototypeBatch(recipe) || ["standalone-network-innovations-v1","standalone-network-v1","network-input-view-v1","network-input-view-v2"].includes(recipe?.version); }
function networkArchitecture(recipe) { return recipe?.source?.network || recipe?.base?.network || recipe?.network; }
function networkConstructionNote(example) {
  sourceRootConstructionNote(example);
  $("network-noise-controls-note").hidden = example.topology !== "network-innovations";
  $("network-hyperprior-controls-note").hidden = example.topology !== "network-hyperprior";
  $("network-controls-note").hidden = !["network", "network-authored"].includes(example.topology);
}
function resetNetworkWorkbench() {
  networkNoiseMode=false; $("network-noise-edit").hidden=true;
  networkDraft = null; networkMatchingEdit = false;
  networkBlockDrafts.clear(); networkSelectedBlock = null;
  $("network-workbench-content").hidden = true;
  const network = isStandaloneNetwork(current?.view?.recipe?.recipe);
  $("workbench-kind").textContent = network ? "STANDALONE NETWORK WORKBENCH" : "CAUSAL MECHANISM WORKBENCH";
  $("workbench-title").textContent = network ? "Inspect and edit the frozen architecture" : "Inspect and edit the frozen graph";
  $("workbench-intro").textContent = network ? "The architecture below shows ordered inputs, frozen structural masks, gates and projected skips. Select a block to inspect or edit its equations. The generation trace shows original-row values. Edits create linked authored networks." : "The causal structure below is distinct from the generation-process diagram. Select a node to see its channels, function and structural noise. An edit creates a new authored mechanism linked to the original archive; it is not a causal intervention or observational conditioning.";
  $("mechanism-load").textContent = network ? "Inspect architecture and prepare edit" : "Inspect graph and prepare edit";
}
function showNetworkEditor(info) {
  networkNoiseMode=!!info.noise;
  $("network-noise-edit").hidden=!networkNoiseMode;
  if(networkNoiseMode) {$("network-noise-plan").value=pretty(info.innovation_plan); $("network-noise-compiled").textContent=pretty(info.compiled);}
  networkDraft = info.editable_recipe;
  showSourceRootEditor();
  networkMatchingEdit = !!info.matching_edit;
  $("network-matching-policy-label").hidden = !networkMatchingEdit;
  $("network-matching-policy").value = "recalibrate";
  networkBlockDrafts.clear(); networkSelectedBlock = null;
  $("workbench-content").hidden = true;
  $("network-workbench-content").hidden = false;
  $("workbench-note").textContent = info.notice;
  $("network-block").replaceChildren(...networkArchitecture(networkDraft).blocks.map((block,i)=>new Option(`Block ${i+1} · ${block.experts[0].bias.length} units · ${block.experts.length} expert(s)`,i)));
  $("network-recipe").value = pretty(networkDraft);
  selectNetworkBlock();
}
function networkStructure() {
  const fn = networkArchitecture(networkDraft), widths = [fn.input_width,...fn.blocks.map(b=>b.experts[0].bias.length)];
  const width = Math.max(640,widths.length*190), height=250, selected=Number($("network-block").value)+1;
  const svg=element("svg",{viewBox:`0 0 ${width} ${height}`,role:"img","aria-label":"Standalone frozen network architecture with projected skip paths"});
  const ancestors = new Set([selected]);
  for(let state=selected;state>0;state--) if(ancestors.has(state)) {
    const block=fn.blocks[state-1]; [...block.parents,...block.skips.map(s=>s.source)].forEach(p=>ancestors.add(p));
  }
  svg.append(element("rect",{width,height,fill:"white"}));
  fn.blocks.forEach((block,i)=>{
    const destination=i+1;
    for(const [source,skip] of [...block.parents.map(p=>[p,false]),...block.skips.map(s=>[s.source,true])]) {
      const edge=element("path",{d:`M ${75+source*190} 140 Q ${75+(source+destination)*95} ${skip?28:65} ${75+destination*190} 140`,fill:"none",stroke:skip?"#a7632a":"#17826b","stroke-width":2,"stroke-dasharray":skip?"6 4":""});
      edge.append(element("title",{},`${skip?"Projected skip":"Ordered input"}: state ${source} → block ${destination}`));svg.append(edge);
    }
  });
  widths.forEach((units,state)=>{
    const block=state?fn.blocks[state-1]:null, group=element("g",{tabindex:0,role:state?"button":"img","aria-label":state?`Select network block ${state}`:`Input state: ${units} features`,style:state?"cursor:pointer":""});
    group.append(element("rect",{x:20+state*190,y:140,width:155,height:90,rx:10,fill:state===selected?"#d9eee7":ancestors.has(state)?"#eef7f3":"#f1f4f6",stroke:"#78928a"}));
    const lines=block?[`Block ${state} · ${units} units`,`${block.experts.length} expert(s) · ${block.gate?"softmax gate":"one branch"}`,`${block.skips.length} projected skip(s)`]:["Input state 0",`${units} observed features`,"Fixed function per dataset"];
    lines.forEach((text,j)=>group.append(element("text",{x:30+state*190,y:161+j*23,"font-size":11,fill:"#23443b"},text)));
    if(state){group.onclick=()=>{$("network-block").value=String(state-1);selectNetworkBlockSafely();};group.onkeydown=e=>{if(["Enter"," "].includes(e.key)){e.preventDefault();group.onclick();}};}
    svg.append(group);
  });
  $("network-structure").replaceChildren(svg);
  chartExports("network-structure",svg,"network-architecture","Standalone network architecture");
}
function selectNetworkBlock() {
  if (networkSelectedBlock !== null) networkBlockDrafts.set(networkSelectedBlock,$("network-block-json").value);
  const index=Number($("network-block").value), block=networkArchitecture(networkDraft).blocks[index];
  networkSelectedBlock=index;
  $("network-block-json").value=networkBlockDrafts.get(index) ?? pretty(block);
  const masks=block.experts.map((e,i)=>({expert:i,weight_shape:[e.matrix.length,e.matrix[0].length],zero_weights:e.matrix.flat().filter(v=>v===0).length,masked_edges:e.mask.length?e.mask.flat().filter(v=>!v).length:0,total_edges:e.matrix.length*e.matrix[0].length,placement:e.placement,activation:e.activation.kind}));
  $("network-block-info").textContent=pretty({state:index+1,ordered_input_states:block.parents,projected_skip_states:block.skips.map(s=>s.source),gating:block.gate?"input-dependent softmax":"none",masks,note:"Masks and weights are frozen. Zero or disconnected paths are valid; inspect channel ancestry and values in the generation trace below. Orange dashed paths are projected skips, green paths are ordered inputs."});
  networkStructure();
  networkDraftStatus();
}
function networkDraftStatus() {
  const accepted=networkArchitecture(networkDraft)?.blocks[Number($("network-block").value)];
  $("network-draft-status").textContent=$("network-block-json").value===pretty(accepted) ? "Block text matches the accepted draft." : "Unapplied block text retained; it has not been accepted. The diagram and details show the last validated architecture. Apply to validate this text.";
}
$("network-block-json").oninput=networkDraftStatus;
function selectNetworkBlockSafely() { return act("Selecting network block…",async()=>selectNetworkBlock()); }
$("network-block").onchange=selectNetworkBlockSafely;
$("network-apply").onclick=()=>act("Validating network architecture draft…",async()=>{
  const index=Number($("network-block").value), text=$("network-block-json").value;
  networkBlockDrafts.set(index,text);
  networkDraftStatus();
  if ($("network-recipe").value!==pretty(networkDraft)) throw new Error("Complete recipe has unapplied edits. Save that recipe or restore it before applying a block; both drafts are retained.");
  const candidate=JSON.parse(JSON.stringify(networkDraft));
  networkArchitecture(candidate).blocks[index]=JSON.parse(text);
  const planText=$("network-noise-plan").value;
  const validated=await request(networkNoiseMode ? "network_noise_draft" : "network_draft",networkNoiseMode ? {recipe:candidate,innovation_plan:JSON.parse(planText),...limits()} : networkEditRequest(candidate));
  if(networkNoiseMode && $("network-noise-plan").value!==planText) throw new Error("Noise plan changed during validation; all draft text retained. Apply again.");
  if ($("network-block-json").value!==text || Number($("network-block").value)!==index || $("network-recipe").value!==pretty(networkDraft)) throw new Error("Draft changed during validation; edits retained. Apply again.");
  networkDraft=validated.recipe;
  if(networkNoiseMode) {$("network-noise-plan").value=pretty(validated.innovation_plan); $("network-noise-compiled").textContent=pretty(validated.compiled);}
  networkBlockDrafts.delete(index); networkSelectedBlock=null;
  $("network-recipe").value=pretty(networkDraft);
  $("network-block").replaceChildren(...networkArchitecture(networkDraft).blocks.map((block,i)=>new Option(`Block ${i+1} · ${block.experts[0].bias.length} units · ${block.experts.length} expert(s)`,i)));
  $("network-block").value=String(index);
  selectNetworkBlock();
  $("workbench-note").textContent="Validated architecture draft applied. Generate the linked edit to save it. Unapplied block text is retained while selecting other blocks.";
});
$("network-save").onclick=()=>act("Validating, generating and saving standalone network edit…",async()=>{
  const sourceText=$("network-recipe").value, planText=$("network-noise-plan").value;
  let recipe=JSON.parse(sourceText);
  if(networkNoiseMode) {
    const validated=await request("network_noise_draft",{recipe,innovation_plan:JSON.parse(planText),...limits()});
    if($("network-recipe").value!==sourceText || $("network-noise-plan").value!==planText) throw new Error("Draft changed during validation; text retained. Save again.");
    recipe=validated.compiled_recipe;
  }
  if (networkMatchingEdit) recipe=(await request("network_draft",networkEditRequest(recipe))).recipe;
  const result=await request("edit",{recipe,...limits()});
  if(networkNoiseMode && ($("network-recipe").value!==sourceText || $("network-noise-plan").value!==planText)) throw new Error("Draft changed during generation; your text is retained. A linked edit was saved locally; use saved records to reopen it.");
  show(result);
});


function networkEditRequest(recipe) {
  if (!networkMatchingEdit) return {recipe,...limits()};
  const draft=JSON.parse(JSON.stringify(recipe));
  if (draft.version === "network-input-view-v2") {
    delete draft.matching;
    draft.version="network-input-view-v1";
  }
  return {recipe:draft,matching_policy:$("network-matching-policy").value,...limits()};
}
