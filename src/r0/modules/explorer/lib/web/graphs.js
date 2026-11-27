"use strict";
let generationGraph = null, graphNode = null, graphScale = 1;
function clearGraph() {
  clearGPPopulationMatrices();
  generationGraph = null; graphNode = null;
  $("generation-graph").replaceChildren(); $("graph-downloads").replaceChildren();
  $("graph-detail").hidden = true; $("graph-tools").hidden = true;
  $("graph-note").textContent = "Built on demand from saved parameters. Dataset values remain unchanged.";
}
function resetGraph() {
  $("graph-rows").value = "0,3"; $("graph-column").value = 0;
  $("graph-component").replaceChildren(new Option("Whole task", "")); clearGraph();
}
function graphRequest() {
  const parts = $("graph-rows").value.split(",").map(value => value.trim());
  if (parts.some(value => !/^\d+$/.test(value))) throw new Error("Enter comma-separated original row indices.");
  return {...limits(), indices: parts.map(Number), component: $("graph-component").value === "" ? null : number("graph-component"), column_start: number("graph-column")};
}
function graphStyle(node) {
  if (node.id === "inputs") return {color:"#3674c5",light:"#eef5ff",kind:"INPUTS",icon:"x"};
  if (["weights","normals","noise","labels"].includes(node.id)) return {color:"#b7791f",light:"#fff8e8",kind:node.id === "weights" ? "PARAMETERS" : "RANDOM DRAW",icon:node.id === "noise" ? "ε" : "z"};
  if (["signal","target","logits","probabilities"].includes(node.id)) return {color:"#138470",light:"#eaf8f3",kind:node.id === "target" ? "TARGET" : "SIGNAL",icon:node.id === "target" ? "y" : "s"};
  if (node.id === "split") return {color:"#64748b",light:"#f1f5f9",kind:"SPLIT",icon:"↗"};
  return {color:"#8060bd",light:"#f5f1fc",kind:node.component === undefined ? "TRANSFORM" : "COMPONENT",icon:"ƒ"};
}
function setGraphZoom(scale) {
  const svg = $("generation-graph").querySelector("svg"); if (!svg) return;
  graphScale = Math.max(0.03, Math.min(2, scale));
  svg.style.width = `${svg.viewBox.baseVal.width * graphScale}px`;
  svg.style.height = `${svg.viewBox.baseVal.height * graphScale}px`;
  $("graph-zoom-label").textContent = `${Math.round(graphScale * 100)}%`;
}
function fitGraph() {
  const svg = $("generation-graph").querySelector("svg"); if (!svg) return;
  setGraphZoom(Math.min(1, ($("generation-graph").clientWidth - 4) / svg.viewBox.baseVal.width, 580 / svg.viewBox.baseVal.height));
}
function drawGenerationGraph(graph) {
  generationGraph = graph;
  showGPPopulationMatrices(graph);
  const selected = graph.component === null ? "" : String(graph.component);
  $("graph-component").replaceChildren(new Option("Whole task", ""), ...Array.from({length: graph.component_count}, (_, i) => new Option(`Component ${i}`, i)));
  $("graph-component").value = selected;
  $("graph-note").textContent = `${graph.nodes.length} steps · original rows ${graph.rows.join(", ")} · select any step to follow its inputs and outputs. Values show at most 32 columns; parameter lists show their first 32 items. Full parameters are in the dataset export.`;
  const positions = new Map(), counts = new Map();
  for (const node of graph.nodes) {
    const level = node.parents.length ? Math.max(...node.parents.map(id => positions.get(id).level)) + 1 : 0;
    const slot = counts.get(level) || 0; counts.set(level, slot + 1);
    positions.set(node.id, {level,slot});
  }
  const levels = Math.max(...counts.keys()) + 1, slots = Math.max(...counts.values());
  const vertical = slots <= 4, nodeWidth = 224, nodeHeight = 104;
  const baseWidth = vertical ? slots * 264 + 16 : levels * 280 + 16;
  const baseHeight = vertical ? levels * 144 + 16 : slots * 144 + 16;
  for (const node of graph.nodes) {
    const point = positions.get(node.id);
    point.x = vertical ? 24 + point.slot * 264 + (slots - counts.get(point.level)) * 132 : 24 + point.level * 280;
    point.y = vertical ? 24 + point.level * 144 : 24 + point.slot * 144;
  }
  const longEdges = graph.nodes.flatMap(node => node.parents.filter(parent => positions.get(node.id).level > positions.get(parent).level + 1)).length;
  const width = baseWidth + (vertical ? longEdges * 18 + 24 : 24), height = baseHeight + (vertical ? 24 : longEdges * 18 + 24);
  const svg = element("svg", {viewBox:`0 0 ${width} ${height}`,width,height,role:"group","aria-label":"Generation steps"});
  const defs = element("defs"), grid = element("pattern", {id:"graph-grid",width:24,height:24,patternUnits:"userSpaceOnUse"});
  grid.append(element("circle",{cx:1,cy:1,r:.65,fill:"#dce4ec"})); defs.append(grid);
  const marker = element("marker",{id:"graph-arrow",viewBox:"0 0 10 10",refX:9,refY:5,markerWidth:6,markerHeight:6,orient:"auto"});
  marker.append(element("path",{d:"M1,1 L9,5 L1,9",fill:"none",stroke:"#8fa7b6","stroke-width":1.8,"stroke-linecap":"round","stroke-linejoin":"round"})); defs.append(marker);
  const shadow = element("filter",{id:"graph-shadow",x:"-15%",y:"-20%",width:"130%",height:"150%"});
  shadow.append(element("feDropShadow",{dx:0,dy:3,stdDeviation:4,"flood-color":"#1b3048","flood-opacity":.07})); defs.append(shadow);
  svg.append(element("title",{},"Generation steps reconstructed from saved provenance"),defs,element("rect",{width,height,fill:"#f8fafc"}),element("rect",{width,height,fill:"url(#graph-grid)"}));
  let laneIndex = 0;
  for (const node of graph.nodes) for (const parent of node.parents) {
    const from = positions.get(parent), to = positions.get(node.id), long = to.level > from.level + 1;
    let path;
    if (vertical) {
      const x1 = from.x + nodeWidth / 2, y1 = from.y + nodeHeight, x2 = to.x + nodeWidth / 2;
      const lane = baseWidth + laneIndex * 18;
      path = long ? `M${x1},${y1} V${y1+14} H${lane} V${to.y-14} H${x2} V${to.y}` : `M${x1},${y1} C${x1},${y1+20} ${x2},${to.y-20} ${x2},${to.y}`;
    } else {
      const x1 = from.x + nodeWidth, y1 = from.y + nodeHeight / 2, y2 = to.y + nodeHeight / 2;
      const lane = baseHeight + laneIndex * 18;
      path = long ? `M${x1},${y1} H${x1+18} V${lane} H${to.x-18} V${y2} H${to.x}` : `M${x1},${y1} C${x1+28},${y1} ${to.x-28},${y2} ${to.x},${y2}`;
    }
    if (long) laneIndex++;
    const edge = element("path",{d:path,fill:"none",stroke:"#a7b8c6","stroke-width":1.5,"stroke-linejoin":"round","marker-end":"url(#graph-arrow)","data-from":parent,"data-to":node.id});
    edge.append(element("title",{},`${parent} → ${node.id}`)); svg.append(edge);
  }
  for (const node of graph.nodes) {
    const {x,y} = positions.get(node.id), style = graphStyle(node);
    const group = element("g",{"data-node":node.id,tabindex:0,role:"button","aria-label":node.label,class:"graph-node",style:"cursor:pointer"});
    group.append(element("title",{},node.label),element("rect",{x,y,width:nodeWidth,height:nodeHeight,rx:13,fill:"white",stroke:"#dce4ec","stroke-width":1,filter:"url(#graph-shadow)",class:"graph-card"}));
    group.append(element("rect",{x:x+14,y:y+12,width:26,height:26,rx:8,fill:style.light}));
    group.append(element("text",{x:x+27,y:y+31,fill:style.color,"font-size":17,"font-family":"sans-serif","font-weight":600,"text-anchor":"middle"},style.icon));
    group.append(element("text",{x:x+49,y:y+29,fill:style.color,"font-size":9,"font-family":"sans-serif","font-weight":700,"letter-spacing":1.2},style.kind));
    const lines = [""];
    for (const word of node.label.replaceAll("_"," ").split(" ")) {if ((lines.at(-1)+word).length>27) lines.push(""); lines[lines.length-1]+=word+" ";}
    lines.slice(0,2).forEach((line,i)=>group.append(element("text",{x:x+14,y:y+58+i*17,fill:"#25374a","font-size":13,"font-family":"sans-serif","font-weight":600},line.trim())));
    group.append(element("text",{x:x+14,y:y+92,fill:"#80909f","font-size":10,"font-family":"sans-serif"},`${node.shape.join(" × ")} preview`));
    group.append(element("text",{x:x+nodeWidth-14,y:y+92,fill:style.color,"font-size":10,"font-family":"sans-serif","text-anchor":"end"},"Inspect →"));
    group.onclick=()=>selectGraphNode(node.id);
    group.onkeydown=event=>{if(event.key==="Enter"||event.key===" "){event.preventDefault();selectGraphNode(node.id);}};
    svg.append(group);
  }
  $("generation-graph").replaceChildren(svg); $("graph-tools").hidden=false; $("graph-downloads").replaceChildren();
  setGraphZoom(Math.max(.7,Math.min(1,($("generation-graph").clientWidth-4)/width)));
  chartExports("graph-downloads",svg,"generation-graph");
  const json=document.createElement("button");json.textContent="Graph JSON";
  json.onclick=()=>download("generation-graph.json",new Blob([pretty(generationGraph)],{type:"application/json"}));
  $("graph-downloads").append(json); $("graph-values").textContent=pretty(graph);
  selectGraphNode(graph.nodes.some(node=>node.id==="target")?"target":graph.nodes.at(-1).id);
}
function matrixContext(node, rowCount = node.values.length, originalRows = generationGraph.rows) {
  const p = node.parameters, columns = node.column_indices;
  let rowLabels = originalRows, columnLabels = columns, rowHeading = "Original row", columnHeading = "Column";
  let count = `${rowCount} of ${node.shape[0]} selected rows · ${columns.length} of ${node.shape[1]} columns`;
  let axes = `Original row order: ${originalRows.join(", ")}`;
  if (p.support_indices_shown) {
    rowLabels = p.support_indices_shown; rowHeading = "Support row";
    count = `${rowCount} of ${p.full_shape[0]} support rows · ${columns.length} of ${p.full_shape[1]} output columns`;
    axes = "Original joint standard normals; first support rows, not observation rows.";
  } else if (p.value_axes || p.display === "selected task-row covariance submatrix") {
    rowLabels = p.row_indices; columnLabels = columns.map(i => p.row_indices[i]); columnHeading = "Original row";
    axes = `Covariance on original rows ${p.row_indices.join(", ")}`;
    if (p.full_shape) axes += `; full joint support ${p.full_shape.join(" × ")}`;
  } else if (typeof p.axes === "string") {
    rowLabels = node.values.map((_, i) => i); rowHeading = "Matrix row";
    count = `${rowCount} of ${node.shape[0]} matrix rows · ${columns.length} of ${node.shape[1]} columns`;
    axes = p.axes;
  } else if (p.scope?.includes("not an emitted dataset row") || node.values.length !== originalRows.length) {
    rowLabels = node.values.map((_, i) => i); rowHeading = "Shared draw";
    count = `${rowCount} shared construction / environment rows · ${columns.length} columns`;
    axes = p.scope || "Shared values; not indexed by selected dataset rows.";
  }
  return {count, axes, rowLabels, columnLabels, rowHeading, columnHeading};
}
function graphMatrix(container, node) {
  const rows = node.values.slice(0, 16), columns = node.column_indices, context = matrixContext(node, Math.min(16, node.values.length));
  const magnitude = Math.max(...rows.flat().map(Math.abs)), scale = magnitude || 1;
  const svg = element("svg", {viewBox: "0 0 480 334", role: "img", "aria-label": `${node.label} values`});
  svg.append(element("title", {}, node.label), element("rect", {width: 480, height: 334, fill: "white"}));
  const label = (x, y, text, anchor = "start") => svg.append(element("text", {x, y, "font-family": "sans-serif", "font-size": 11, fill: "#637482", "text-anchor": anchor}, text));
  label(62, 18, node.label); label(62, 34, context.count);
  const axisLines = [""];
  for (const word of context.axes.split(" ")) {
    if ((axisLines.at(-1) + word).length > 61) axisLines.push("");
    axisLines[axisLines.length - 1] += word + " ";
  }
  axisLines.forEach((line, i) => label(62, 52 + i * 14, line.trim()));
  rows.forEach((row, i) => row.forEach((value, j) => {
    const alpha = Math.abs(value) / scale, color = value >= 0 ? [23, 130, 107] : [215, 137, 73];
    const cell = element("rect", {x: 62 + j * 394 / columns.length, y: 100 + i * 172 / rows.length, width: 394 / columns.length, height: 172 / rows.length, fill: `rgb(${color.map(channel => Math.round(255 + alpha * (channel - 255))).join(",")})`});
    cell.append(element("title", {}, `${context.rowHeading} ${context.rowLabels[i]}, ${context.columnHeading} ${context.columnLabels[j]}: ${value}`)); svg.append(cell);
  }));
  label(54, 109, context.rowLabels[0], "end"); label(54, 272, context.rowLabels[rows.length - 1], "end");
  label(62, 292, context.columnLabels[0]); label(456, 292, context.columnLabels.at(-1), "end");
  label(62, 314, "Orange < 0 · white = 0 · green > 0");
  label(456, 314, `Max |value| ${format(magnitude)}`, "end");
  $(container).replaceChildren(svg); chartExports(container, svg, container);
}
function selectGraphNode(id) {
  graphNode = generationGraph.nodes.find(node => node.id === id);
  document.querySelectorAll("#generation-graph [data-node]").forEach(node => {
    const selected = node.getAttribute("data-node") === id;
    node.setAttribute("aria-pressed", String(selected));
    const card=node.querySelector(".graph-card"), style=graphStyle(graphNode);
    card.setAttribute("stroke-width",selected?2.5:1);card.setAttribute("stroke",selected?style.color:"#dce4ec");card.setAttribute("fill",selected?style.light:"white");
  });
  document.querySelectorAll("#generation-graph [data-from]").forEach(edge=>{
    const active=edge.getAttribute("data-from")===id||edge.getAttribute("data-to")===id;
    edge.setAttribute("opacity",active?1:.35);edge.setAttribute("stroke",active?graphStyle(graphNode).color:"#a7b8c6");edge.setAttribute("stroke-width",active?2:1.5);
  });
  $("graph-detail").hidden = false; $("graph-node-title").textContent = graphNode.label;
  $("graph-formula").textContent = graphNode.formula; $("graph-parameters").textContent = pretty(graphNode.parameters);
  $("graph-expand").hidden = graphNode.component === undefined;
  $("graph-input").replaceChildren(...graphNode.parents.map(parent => new Option(generationGraph.nodes.find(node => node.id === parent).label, parent)));
  showGraphInput(); graphMatrix("graph-output-plot", graphNode);
  const context = matrixContext(graphNode);
  $("graph-matrix-note").textContent = `${context.count}. ${context.axes}. Heatmaps show the first 16 returned rows; the table contains all returned rows. Column positions ${graphNode.column_indices[0]}–${graphNode.column_indices.at(-1)}.`;
  table("graph-output-table", [context.rowHeading, ...context.columnLabels.map(i => `${context.columnHeading} ${i}`)], graphNode.values.map((row, i) => [context.rowLabels[i], ...row]));
}
function showGraphInput() {
  const input = generationGraph.nodes.find(node => node.id === $("graph-input").value);
  if (input) graphMatrix("graph-input-plot", input);
  else $("graph-input-plot").textContent = "Source step: inspect its parameters and recorded values.";
}
$("graph").onclick = () => act("Reconstructing generation steps…", async () => {
  const data = graphRequest(); clearGraph();
  const result = await request("graph", data);
  if (pretty(data) === pretty(graphRequest())) drawGenerationGraph(result.graph);
});
for (const id of ["graph-rows", "graph-column", "graph-component"]) $(id).onchange = clearGraph;
$("graph-input").onchange = showGraphInput;
$("graph-expand").onclick = () => {$("graph-component").value = graphNode.component; $("graph").click();};

$("graph-fit").onclick=fitGraph;
$("graph-zoom-in").onclick=()=>setGraphZoom(graphScale*1.25);
$("graph-zoom-out").onclick=()=>setGraphZoom(graphScale/1.25);
$("graph-zoom-label").onclick=()=>setGraphZoom(1);
