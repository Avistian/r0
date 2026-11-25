"use strict";
let generationGraph = null, graphNode = null;
function clearGraph() {
  generationGraph = null; graphNode = null;
  $("generation-graph").replaceChildren(); $("graph-downloads").replaceChildren();
  $("graph-detail").hidden = true;
  $("graph-note").textContent = "Built on demand from saved parameters. Dataset values remain unchanged.";
}
function resetGraph() {
  $("graph-rows").value = "0,3"; $("graph-column").value = 0;
  $("graph-component").replaceChildren(new Option("Whole task", "")); clearGraph();
}
function graphRequest() {
  const parts = $("graph-rows").value.split(",").map(value => value.trim());
  if (parts.some(value => !/^\d+$/.test(value))) throw new Error("Enter comma-separated original row indices.");
  return {indices: parts.map(Number), component: $("graph-component").value === "" ? null : number("graph-component"), column_start: number("graph-column")};
}
function drawGenerationGraph(graph) {
  generationGraph = graph;
  const selected = graph.component === null ? "" : String(graph.component);
  $("graph-component").replaceChildren(new Option("Whole task", ""), ...Array.from({length: graph.component_count}, (_, i) => new Option(`Component ${i}`, i)));
  $("graph-component").value = selected;
  $("graph-note").textContent = `${graph.nodes.length} steps · original rows ${graph.rows.join(", ")} · up to 32 value columns per node, with explicit indices. Parameter lists show their first 32 items; full parameters are in the dataset export. Diagram PNG is limited to 4096 px on its longest side.`;
  const positions = new Map(), counts = new Map();
  for (const node of graph.nodes) {
    const level = node.parents.length ? Math.max(...node.parents.map(id => positions.get(id).level)) + 1 : 0;
    const row = counts.get(level) || 0; counts.set(level, row + 1);
    positions.set(node.id, {level, x: 20 + level * 220, y: 24 + row * 80});
  }
  const width = (Math.max(...counts.keys()) + 1) * 220 + 20;
  const baseHeight = Math.max(150, Math.max(...counts.values()) * 80 + 24);
  const longEdges = graph.nodes.flatMap(node => node.parents.filter(parent => positions.get(node.id).level > positions.get(parent).level + 1));
  const height = baseHeight + longEdges.length * 18 + 20;
  const svg = element("svg", {viewBox: `0 0 ${width} ${height}`, width, height, role: "group", "aria-label": "Generation steps"});
  svg.append(element("title", {}, "Generation steps reconstructed from saved provenance"), element("rect", {width, height, fill: "white"}));
  const defs = element("defs"), marker = element("marker", {id: "graph-arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse"});
  marker.append(element("path", {d: "M0,0 L10,5 L0,10 z", fill: "#9eb4b2"})); defs.append(marker); svg.append(defs);
  let longEdge = 0;
  for (const node of graph.nodes) {
    const to = positions.get(node.id);
    for (const parent of node.parents) {
      const from = positions.get(parent), x = from.x + 190;
      const lane = baseHeight + longEdge * 18;
      const path = to.level > from.level + 1
        ? `M${x},${from.y + 26} H${x + 12} V${lane} H${to.x - 12} V${to.y + 26} H${to.x}`
        : `M${x},${from.y + 26} C${x + 20},${from.y + 26} ${to.x - 20},${to.y + 26} ${to.x},${to.y + 26}`;
      if (to.level > from.level + 1) longEdge++;
      const edge = element("path", {d: path, fill: "none", stroke: "#b4c6c4", "marker-end": "url(#graph-arrow)", "data-from": parent, "data-to": node.id});
      edge.append(element("title", {}, `${parent} → ${node.id}`)); svg.append(edge);
    }
  }
  for (const node of graph.nodes) {
    const {x, y} = positions.get(node.id);
    const group = element("g", {"data-node": node.id, tabindex: 0, role: "button", "aria-label": node.label, style: "cursor:pointer"});
    group.append(element("title", {}, node.label), element("rect", {x, y, width: 190, height: 52, rx: 8, fill: "#f2f7f5", stroke: "#b6cec4", "stroke-width": 1}));
    const words = node.label.split(" "), lines = [""];
    for (const word of words) { if ((lines.at(-1) + word).length > 26) lines.push(""); lines[lines.length - 1] += word + " "; }
    lines.slice(0, 2).forEach((line, i) => group.append(element("text", {x: x + 12, y: y + (lines.length === 1 ? 30 : 22) + i * 16, fill: "#294c42", "font-size": 11, "font-family": "sans-serif"}, line.trim())));
    group.onclick = () => selectGraphNode(node.id);
    group.onkeydown = event => {if (event.key === "Enter" || event.key === " ") {event.preventDefault(); selectGraphNode(node.id);}};
    svg.append(group);
  }
  $("generation-graph").replaceChildren(svg); $("graph-downloads").replaceChildren();
  chartExports("graph-downloads", svg, "generation-graph");
  const json = document.createElement("button"); json.textContent = "Graph JSON";
  json.onclick = () => download("generation-graph.json", new Blob([pretty(generationGraph)], {type: "application/json"}));
  $("graph-downloads").append(json);
  $("graph-values").textContent = pretty(graph);
  selectGraphNode(graph.nodes.some(node => node.id === "target") ? "target" : graph.nodes.at(-1).id);
}
function matrixContext(node, rowCount = node.values.length) {
  const p = node.parameters, columns = node.column_indices;
  let rowLabels = generationGraph.rows, columnLabels = columns, rowHeading = "Original row", columnHeading = "Column";
  let count = `${rowCount} of ${node.shape[0]} selected rows · ${columns.length} of ${node.shape[1]} columns`;
  let axes = `Original row order: ${generationGraph.rows.join(", ")}`;
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
    node.querySelector("rect").setAttribute("stroke-width", selected ? 3 : 1);
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
