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
function graphExports(container, svg, name) {
  const controls = document.createElement("div"); controls.className = "chart-actions";
  for (const kind of ["SVG", "PNG"]) {
    const button = document.createElement("button"); button.textContent = kind;
    button.setAttribute("aria-label", `Export ${name} as ${kind}`);
    button.onclick = () => saveChart(svg, name, kind); controls.append(button);
  }
  $(container).append(controls);
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
  graphExports("graph-downloads", svg, "generation-graph");
  const json = document.createElement("button"); json.textContent = "Graph JSON";
  json.onclick = () => download("generation-graph.json", new Blob([pretty(generationGraph)], {type: "application/json"}));
  $("graph-downloads").append(json);
  $("graph-values").textContent = pretty(graph);
  selectGraphNode(graph.nodes.some(node => node.id === "target") ? "target" : graph.nodes.at(-1).id);
}
function graphMatrix(container, node) {
  const rows = node.values.slice(0, 16), columns = node.column_indices;
  const magnitude = Math.max(...rows.flat().map(Math.abs)), scale = magnitude || 1;
  const svg = element("svg", {viewBox: "0 0 480 270", role: "img", "aria-label": `${node.label} values`});
  svg.append(element("title", {}, node.label), element("rect", {width: 480, height: 270, fill: "white"}));
  const label = (x, y, text, anchor = "start") => svg.append(element("text", {x, y, "font-family": "sans-serif", "font-size": 11, fill: "#637482", "text-anchor": anchor}, text));
  label(62, 19, `${rows.length} of ${node.shape[0]} matrix rows · ${columns.length} of ${node.shape[1]} columns`);
  rows.forEach((row, i) => row.forEach((value, j) => {
    const alpha = Math.abs(value) / scale, color = value >= 0 ? [23, 130, 107] : [215, 137, 73];
    const cell = element("rect", {x: 62 + j * 394 / columns.length, y: 38 + i * 172 / rows.length, width: 394 / columns.length, height: 172 / rows.length, fill: `rgb(${color.map(channel => Math.round(255 + alpha * (channel - 255))).join(",")})`});
    cell.append(element("title", {}, `Matrix row ${i}, column ${columns[j]}: ${value}`)); svg.append(cell);
  }));
  label(54, 47, "0", "end"); label(54, 210, rows.length - 1, "end");
  label(62, 230, columns[0]); label(456, 230, columns.at(-1), "end");
  label(62, 252, "Orange < 0 · white = 0 · green > 0");
  label(456, 252, `Max |value| ${format(magnitude)}`, "end");
  $(container).replaceChildren(svg); graphExports(container, svg, container);
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
  $("graph-matrix-note").textContent = `Output shape ${graphNode.shape.join(" × ")}; displayed columns ${graphNode.column_indices[0]}–${graphNode.column_indices.at(-1)}. Original observation rows, where applicable: ${generationGraph.rows.join(", ")}. Coefficient, covariance and normal matrices use the axes stated in step parameters. Heatmaps show the first 16 matrix rows; the table contains all returned rows.`;
  table("graph-output-table", ["Matrix row", ...graphNode.column_indices.map(i => `Column ${i}`)], graphNode.values.map((row, i) => [i, ...row]));
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
