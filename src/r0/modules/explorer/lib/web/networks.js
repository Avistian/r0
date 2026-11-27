"use strict";
let networkDraft = null;
function isStandaloneNetwork(recipe) { return recipe?.version === "standalone-network-v1"; }
function networkConstructionNote(example) {
  $("network-controls-note").hidden = !["network", "network-authored"].includes(example.topology);
}
function resetNetworkWorkbench() {
  networkDraft = null;
  $("network-workbench-content").hidden = true;
  const network = isStandaloneNetwork(current?.view?.recipe?.recipe);
  $("workbench-kind").textContent = network ? "STANDALONE NETWORK WORKBENCH" : "CAUSAL MECHANISM WORKBENCH";
  $("workbench-title").textContent = network ? "Inspect and edit the frozen architecture" : "Inspect and edit the frozen graph";
  $("workbench-intro").textContent = network ? "The architecture below shows ordered inputs, frozen structural masks, gates and projected skips. Select a block to inspect or edit its equations. The generation trace shows original-row values. Edits create linked authored networks." : "The causal structure below is distinct from the generation-process diagram. Select a node to see its channels, function and structural noise. An edit creates a new authored mechanism linked to the original archive; it is not a causal intervention or observational conditioning.";
  $("mechanism-load").textContent = network ? "Inspect architecture and prepare edit" : "Inspect graph and prepare edit";
}
function showNetworkEditor(info) {
  networkDraft = info.editable_recipe;
  $("workbench-content").hidden = true;
  $("network-workbench-content").hidden = false;
  $("workbench-note").textContent = info.notice;
  $("network-block").replaceChildren(...networkDraft.network.blocks.map((block,i)=>new Option(`Block ${i+1} · ${block.experts[0].bias.length} units · ${block.experts.length} expert(s)`,i)));
  $("network-recipe").value = pretty(networkDraft);
  selectNetworkBlock();
}
function networkStructure() {
  const fn = networkDraft.network, widths = [fn.input_width,...fn.blocks.map(b=>b.experts[0].bias.length)];
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
    if(state){group.onclick=()=>{$("network-block").value=String(state-1);selectNetworkBlock();};group.onkeydown=e=>{if(["Enter"," "].includes(e.key)){e.preventDefault();group.onclick();}};}
    svg.append(group);
  });
  $("network-structure").replaceChildren(svg);
  chartExports("network-structure",svg,"network-architecture","Standalone network architecture");
}
function selectNetworkBlock() {
  const index=Number($("network-block").value), block=networkDraft.network.blocks[index];
  $("network-block-json").value=pretty(block);
  const masks=block.experts.map((e,i)=>({expert:i,weight_shape:[e.matrix.length,e.matrix[0].length],zero_weights:e.matrix.flat().filter(v=>v===0).length,masked_edges:e.mask.length?e.mask.flat().filter(v=>!v).length:0,total_edges:e.matrix.length*e.matrix[0].length,placement:e.placement,activation:e.activation.kind}));
  $("network-block-info").textContent=pretty({state:index+1,ordered_input_states:block.parents,projected_skip_states:block.skips.map(s=>s.source),gating:block.gate?"input-dependent softmax":"none",masks,note:"Masks and weights are frozen. Zero or disconnected paths are valid; inspect channel ancestry and values in the generation trace below. Orange dashed paths are projected skips, green paths are ordered inputs."});
  networkStructure();
}
$("network-block").onchange=selectNetworkBlock;
$("network-apply").onclick=()=>act("Updating network architecture draft…",async()=>{
  networkDraft.network.blocks[Number($("network-block").value)]=JSON.parse($("network-block-json").value);
  $("network-recipe").value=pretty(networkDraft);
  $("workbench-note").textContent="Architecture draft changed. Generate the linked edit to validate widths, masks, gates and parent order.";
});
$("network-save").onclick=()=>act("Validating, generating and saving standalone network edit…",async()=>{
  show(await request("edit",{recipe:JSON.parse($("network-recipe").value),...limits()}));
});
