"use strict";
let constructionExamples = {}, mechanismDraft = null, mechanismInfo = null;
function explorationLimits() {
  const caps = JSON.parse($("exploration-limits").value);
  if (!caps || Array.isArray(caps) || typeof caps !== "object") throw new Error("Exploration caps must be a JSON object.");
  return Object.keys(caps).length ? {causal_limits: caps} : {};
}
function resetWorkbench() {
  mechanismDraft = null; mechanismInfo = null;
  $("workbench-content").hidden = true;
  $("workbench-note").textContent = "The original dataset remains active while you create a linked mechanism edit.";
  $("edit-results").hidden = !current?.edit;
  if (current?.edit) {
    const edit = current.edit;
    $("edit-path").value = edit.path;
    $("edit-semantics").textContent = edit.card.semantics;
    $("edit-card").textContent = pretty({path: edit.path, after_archive: edit.after_archive, ...edit.card});
    const rows = [current.view.rows, edit.after.rows];
    const xs = rows.flatMap(rs => rs.map(r => r.inputs[0])), ys = rows.flatMap(rs => rs.map(r => r.target));
    ["before", "after"].forEach((side, i) => {
      const {svg, x, y} = plotFrame(`edit-${side}-plot`, `${side} mechanism`, "x0", "Observed target", Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys));
      for (const row of rows[i]) {
        const dot = element("circle", {cx:x(row.inputs[0]), cy:y(row.target), r:3, fill: row.role === "context" ? "#17826b" : "#d78949"});
        dot.append(element("title", {}, `Row ${row.index}: x0=${row.inputs[0]}, target=${row.target}`)); svg.append(dot);
      }
    });
  }
}
function causalStructure() {
  const nodes = mechanismDraft.nodes, width = Math.max(540, nodes.length * 180), height = 210;
  const svg = element("svg", {viewBox:`0 0 ${width} ${height}`, role:"img", "aria-label":"Causal structure and channel assignments"});
  svg.append(element("rect", {width, height, fill:"white"}));
  const selected = Number($("mechanism-node").value), ancestors = new Set([selected]);
  for (let i = selected; i >= 0; i--) if (ancestors.has(i)) nodes[i].parents.forEach(p => ancestors.add(p));
  nodes.forEach((node, i) => node.parents.forEach(parent => {
    const edge = element("path", {d:`M ${50+parent*180} 105 Q ${50+(parent+i)*90} ${20+10*(i-parent)} ${50+i*180} 105`, stroke:ancestors.has(i)&&ancestors.has(parent)?"#17826b":"#b4c0c8", fill:"none", "stroke-width":2});
    edge.append(element("title", {}, `Node ${parent} → node ${i}`)); svg.append(edge);
    svg.append(element("path", {d:`M ${44+i*180} 95 L ${50+i*180} 105 L ${54+i*180} 94`, stroke:"#637482", fill:"none"}));
  }));
  nodes.forEach((node,i) => {
    const info = mechanismInfo.nodes[i], group = element("g", {tabindex:0, role:"button", "aria-label":`Inspect causal node ${i}`, style:"cursor:pointer"});
    group.append(element("rect",{x:20+i*180,y:105,width:150,height:85,rx:10,fill:i===selected?"#d9eee7":ancestors.has(i)?"#eef7f3":"#f1f4f6",stroke:"#78928a"}));
    const observed = mechanismDraft.observed.filter(([n]) => n===i).map(([,c]) => c), targets = mechanismDraft.target.filter(([n]) => n===i).map(([,c])=>c);
    const hidden = info.width - new Set([...observed,...targets]).size;
    for (const [j,text] of [`Node ${i} · ${info.width} channels`,`Observed: ${observed.join(",") || "none"}`,`Target: ${targets.join(",") || "none"} · hidden ${hidden}`].entries()) group.append(element("text",{x:30+i*180,y:126+j*22,"font-size":11,fill:"#23443b"},text));
    group.onclick = () => {$("mechanism-node").value=String(i); selectMechanismNode();};
    group.onkeydown = event => {if (["Enter"," "].includes(event.key)) {event.preventDefault();group.onclick();}};
    svg.append(group);
  });
  $("causal-structure").replaceChildren(svg);
  chartExports("causal-structure", svg, "causal-structure", "Causal structure");
}
function selectMechanismNode() {
  const index = Number($("mechanism-node").value), node = mechanismDraft.nodes[index];
  $("mechanism-parents").value = pretty(node.parents);
  $("mechanism-noise").value = pretty(node.noise_scale);
  $("mechanism-function").value = pretty(node.function);
  $("mechanism-node-info").textContent = pretty({...mechanismInfo.nodes[index], geometry:mechanismInfo.geometry, note:"Green highlights the selected node and its ancestors. Hidden channels are not selected as observed features or targets. Values and equations are available in the generation-process drill-down below."});
  causalStructure();
}
$("construction-family").onchange = () => {$("construction-controls").value=pretty(constructionExamples[$("construction-family").value]);};
$("construct").onclick = () => act("Sampling, generating and saving graph…", async () => show(await request("construct", {controls:JSON.parse($("construction-controls").value), construction_seed:$("construction-seed").value, seed:$("seed").value, task_id:$("task-id").value, ...limits()})));
$("mechanism-load").onclick = () => act("Verifying mechanism archive…", async () => {
  mechanismInfo = await request("mechanism"); mechanismDraft = mechanismInfo.editable_recipe;
  $("workbench-content").hidden = false; $("workbench-note").textContent = mechanismInfo.notice;
  $("mechanism-node").replaceChildren(...mechanismInfo.nodes.map(n=>new Option(`Node ${n.index} · ${n.function}`,n.index)));
  $("mechanism-recipe").value=pretty(mechanismDraft); $("mechanism-observed").value=pretty(mechanismDraft.observed); $("mechanism-target").value=pretty(mechanismDraft.target);
  selectMechanismNode();
});
$("mechanism-node").onchange=selectMechanismNode;
$("mechanism-apply").onclick=()=>act("Updating mechanism draft…",async()=>{
  const node=mechanismDraft.nodes[Number($("mechanism-node").value)];
  node.parents=JSON.parse($("mechanism-parents").value); node.noise_scale=JSON.parse($("mechanism-noise").value); node.function=JSON.parse($("mechanism-function").value);
  $("mechanism-recipe").value=pretty(mechanismDraft); $("workbench-note").textContent="Draft changed. Generate the edit to validate dimensions and save a linked comparison.";
});
$("mechanism-save").onclick=()=>act("Validating, generating and saving edit…",async()=>{
  const recipe=JSON.parse($("mechanism-recipe").value); recipe.observed=JSON.parse($("mechanism-observed").value); recipe.target=JSON.parse($("mechanism-target").value);
  show(await request("edit",{recipe,...limits()}));
});
$("edit-open").onclick=()=>act("Verifying saved mechanism edit…",async()=>show(await request("open_edit",{path:$("edit-path").value,...explorationLimits()})));
$("edit-export").onclick=()=>act("Exporting mechanism edit…",async()=>{
  const result=await request("edit_export",{max_bytes:number("bytes")*1024**2,expected_edit_id:current.edit.card.after_episode_id});
  download(result.name,new Blob([Uint8Array.from(atob(result.data),c=>c.charCodeAt(0))],{type:result.mime}));
});
request("construction_examples").then(result=>{
  constructionExamples=result.examples;
  $("construction-family").replaceChildren(...Object.keys(constructionExamples).map(name=>new Option(name,name)));
  $("construction-family").onchange();
}).catch(error=>{$("status").className="error";$("status").textContent=error.message;});
