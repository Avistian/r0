"use strict";
const $ = id => document.getElementById(id);
let examples = {}, current = null, busy = false;
const number = id => Number($(id).value);
const pretty = value => JSON.stringify(value, null, 2);
const format = value => {
  if (value === null) return "unavailable";
  const number = Number(value), magnitude = Math.abs(number);
  return magnitude > 0 && (magnitude < 0.001 || magnitude >= 1e6)
    ? number.toExponential(2)
    : number.toLocaleString(undefined, {maximumSignificantDigits: 4});
};

async function request(action, data = {}) {
  if (current && ["row", "parameters", "export", "curve", "gp_curve", "graph", "pair", "pair_export", "mechanism", "edit", "edit_export"].includes(action)) data = {...data, expected_episode_id: current.view.episode_id};
  if (["graph", "pair", "open_pair", "mechanism", "edit"].includes(action)) data = {...data, ...explorationLimits()};
  if (window.r0request) return window.r0request(action, data);
  const response = await fetch("/api", {
    method: "POST", headers: {"Content-Type": "application/json", "X-r0-explorer": window.r0token},
    body: JSON.stringify({action, data})
  });
  const result = await response.json();
  if (!response.ok || result.error) throw new Error(result.error || "Request failed");
  return result;
}

async function act(message, work) {
  if (busy) return;
  busy = true;
  $("status").className = "";
  $("status").textContent = message;
  document.querySelectorAll("button").forEach(button => button.disabled = true);
  try {
    await work();
    $("status").textContent = "Ready.";
  } catch (error) {
    $("status").className = "error";
    $("status").textContent = error.message;
  } finally {
    busy = false;
    document.querySelectorAll("button").forEach(button => button.disabled = false);
  }
}

function taskControls(recipe) { return recipe.recipe?.task || recipe.request?.task || recipe.task || recipe; }
function parseRecipe() {
  return JSON.parse($("recipe").value, (key, value) => {
    if (key === "split" && typeof value === "number" && !Number.isSafeInteger(value)) throw new Error("Write large split identifiers as quoted decimal strings to preserve every digit.");
    return value;
  });
}
function fillControls(recipe) {
  const task = taskControls(recipe);
  const native=["native-causal-v1","sampled-native-graph-v1"].includes(recipe.version);
  $("difficulty-label").hidden=native;
  for(const id of ["features","active","task-type","classes"]) $(id).disabled=native;
  for (const [id, key] of Object.entries({rows: "rows", context: "context_rows", features: "features", active: "active_features", "task-type": "task_type", classes: "classes"})) $(id).value = task[key];
  $("difficulty").value = task.task_type === "regression" ? task.noise_ratio : task.separation;
  $("difficulty-label").firstChild.textContent = task.task_type === "regression" ? "Noise / signal" : "Class separation";
  $("recipe").value = pretty(recipe);
}
function updateRecipe(changed) {
  const recipe = parseRecipe(), task = taskControls(recipe);
  if (recipe.version?.startsWith("sampled-") || recipe.version === "fitted-functional-causal-v1") throw new Error("Frozen construction archives are immutable. Use the mechanism editor below to create a linked edit, or sample a new graph.");
  const fields = {rows: "rows", context: "context_rows", features: "features", active: "active_features", classes: "classes"};
  if (fields[changed]) task[fields[changed]] = number(changed);
  if (changed === "difficulty") task[task.task_type === "regression" ? "noise_ratio" : "separation"] = number("difficulty");
  if (changed === "features" || changed === "active") task.signal_features = Math.max(task.active_features, Math.min(task.features, task.signal_features));
  if (changed === "active" && recipe.components) for (const component of recipe.components) {
    if (!component.axes?.length) {
      const child = taskControls(component.recipe);
      child.active_features = task.active_features;
      child.signal_features = Math.max(child.signal_features, child.active_features);
      child.features = Math.max(child.features, child.signal_features);
    }
  }
  if (changed === "task-type") {
    Object.assign(task, {task_type: $("task-type").value, classes: number("classes"), noise_ratio: 0, separation: 1, class_offsets: []});
    if (recipe.components) for (const component of recipe.components) {
      Object.assign(taskControls(component.recipe), {task_type: task.task_type, classes: task.classes, noise_ratio: 0, separation: 1, class_offsets: []});
    }
  } else if (changed === "classes" && recipe.components) {
    for (const component of recipe.components) taskControls(component.recipe).classes = task.classes;
  }
  fillControls(recipe);
}
function limits() { return {...explorationLimits(), device: $("device").value, max_seconds: number("seconds"), max_bytes: number("bytes") * 1024 ** 2}; }
async function generate() {
  const stored=current?.view?.recipe?.recipe;
  const frozen=stored && (stored.version?.startsWith("sampled-") || stored.version==="fitted-functional-causal-v1") && $("recipe").value===pretty(stored);
  const identity={seed:$("seed").value,task_id:$("task-id").value};
  const payload=frozen ? {record_json:current.recipe_json,frozen_identity:identity} : {recipe:parseRecipe(),...identity};
  const result = await request("generate", {...payload,...limits()});
  show(result);
}
function download(name, blob) {
  const url = URL.createObjectURL(blob), link = document.createElement("a");
  link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function table(container, columns, rows) {
  const table = document.createElement("table"), head = table.createTHead().insertRow();
  for (const name of columns) { const cell = document.createElement("th"); cell.textContent = name; head.append(cell); }
  const body = table.createTBody();
  for (const values of rows) {
    const row = body.insertRow();
    for (const value of values) { const cell = row.insertCell(); cell.textContent = typeof value === "number" ? format(value) : value; }
  }
  $(container).replaceChildren(table);
}
function element(name, attributes = {}, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}
function position(value, low, high) {
  if (low === high) return 0.5;
  const scale = Math.max(Math.abs(low), Math.abs(high)) || 1;
  return (value / scale - low / scale) / (high / scale - low / scale);
}
function plotFrame(container, title, xLabel, yLabel, low, high, yLow, yHigh) {
  const svg = element("svg", {viewBox: "0 0 480 270", role: "img", "aria-label": title});
  svg.append(element("title", {}, title), element("rect", {width: 480, height: 270, fill: "white"}));
  const label = (x, y, text, anchor = "middle") => svg.append(element("text", {x, y, fill: "#637482", "font-size": 11, "font-family": "sans-serif", "text-anchor": anchor}, text));
  for (const y of [34, 78, 122, 166, 210]) svg.append(element("line", {x1: 62, x2: 456, y1: y, y2: y, stroke: "#e8edf0"}));
  label(62, 231, format(low)); label(456, 231, format(high));
  label(54, 214, format(yLow), "end"); label(54, 38, format(yHigh), "end");
  label(260, 258, xLabel); label(62, 18, yLabel, "start");
  $(container).replaceChildren(svg);
  chartExports(container, svg, container, title);
  return {svg, x: value => 62 + 394 * position(value, low, high), y: value => 210 - 176 * position(value, yLow, yHigh)};
}
function chartExports(container, svg, name, title = name) {
  const controls = document.createElement("div"); controls.className = "chart-actions";
  for (const kind of ["SVG", "PNG"]) {
    const button = document.createElement("button"); button.textContent = kind;
    button.setAttribute("aria-label", `Export ${title} as ${kind}`);
    button.onclick = () => saveChart(svg, name, kind);
    controls.append(button);
  }
  $(container).append(controls);
}
function saveChart(svg, name, kind) {
  const xml = new XMLSerializer().serializeToString(svg), blob = new Blob([xml], {type: "image/svg+xml"});
  if (kind === "SVG") { download(`${name}.svg`, blob); return; }
  const image = new Image(), url = URL.createObjectURL(blob);
  image.onload = () => {
    const canvas = document.createElement("canvas"), {width, height} = svg.viewBox.baseVal;
    const scale = Math.min(2, 4096 / width, 4096 / height);
    canvas.width = Math.ceil(width * scale); canvas.height = Math.ceil(height * scale);
    canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(data => download(`${name}.png`, data), "image/png");
    URL.revokeObjectURL(url);
  };
  image.src = url;
}
function histogram(container, summary, name) {
  const {counts, edges} = summary.histogram, maximum = Math.max(...counts);
  const {svg} = plotFrame(container, `${name} distribution`, name, "Row count", summary.min, summary.max, 0, maximum);
  counts.forEach((count, i) => {
    const height = 176 * count / maximum;
    const bar = element("rect", {x: 62 + i * 394 / counts.length, y: 210 - height, width: Math.max(1, 394 / counts.length - 1), height, fill: "#25836b", rx: 1});
    bar.append(element("title", {}, `${format(edges[i])} to ${format(edges[i + 1])}: ${count} rows`)); svg.append(bar);
  });
}
function scatter(container, points, xName, yName) {
  const xs = points.map(point => point.x), ys = points.map(point => point.y);
  const {svg, x, y} = plotFrame(container, `${yName} versus ${xName}`, xName, yName, Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys));
  for (const point of points) {
    const circle = element("circle", {cx: x(point.x), cy: y(point.y), r: 3, fill: point.role === "context" ? "#17826b" : "#d78949", opacity: .65});
    circle.append(element("title", {}, `Row ${point.index} · ${point.role}\n${xName}: ${point.x}\n${yName}: ${point.y}`));
    circle.onclick = () => inspectRow(point.index); svg.append(circle);
  }
  const legend = document.createElement("p"); legend.className = "legend"; legend.textContent = "Green: context · orange: query · click a point to inspect its row"; $(container).append(legend);
}
function charts() {
  if (!current) return;
  const view = current.view, feature = number("feature"), channel = number("channel");
  if(view.target_origin === "native-variable-v1") {nativeCharts(view,feature,channel);return;}
  histogram("feature-hist", view.features[feature].summary, `x${feature}`);
  histogram("target-hist", view.target, "Target");
  scatter("signal-plot", view.rows.map(row => ({...row, x: row.inputs[feature], y: row.scores[Math.min(channel, row.scores.length - 1)]})), `x${feature}`, "Latent signal");
  if (view.task_type === "regression") {
    $("diagnostic-title").textContent = "Signal and observation";
    $("diagnostic-note").textContent = `Observed target versus noiseless signal · realized noise std ${format(view.noise.std)}`;
    scatter("diagnostic-plot", view.rows.map(row => ({...row, x: row.scores[0], y: row.target})), "Latent signal", "Observed target");
  } else {
    $("diagnostic-title").textContent = `Probability of ${view.classes[channel]}`;
    $("diagnostic-note").textContent = "Generated probabilities · labels are sampled from these probabilities";
    scatter("diagnostic-plot", view.rows.map(row => ({...row, x: row.inputs[feature], y: row.probabilities[channel]})), `x${feature}`, "Class probability");
  }
  table("table", ["Row", "Split", ...view.features.map(f => `${f.name} (${f.role})`), "Target", "Signal", "Probability / noise"], view.rows.slice(0, 64).map(row => [row.index, row.role, ...row.inputs, row.target, row.scores[Math.min(channel, row.scores.length - 1)], row.probabilities ? row.probabilities[channel] : row.noise]));
}
function show(result) {
  current = result; const view = result.view;
  $("empty").hidden = true; $("results").hidden = false;
  $("family-title").textContent = view.family;
  $("task-title").textContent = view.task_type === "regression" ? "Regression dataset" : `${view.classes.length}-class dataset`;
  $("metrics").replaceChildren();
  for (const [label, value] of [["TOTAL ROWS", view.total_rows], ["FEATURES", view.features.length], ["CONTEXT ROWS", view.split_counts.context], ["QUERY ROWS", view.split_counts.query]]) {
    const box = document.createElement("div"); box.className = "metric";
    const strong = document.createElement("strong"), caption = document.createElement("span"); strong.textContent = format(value); caption.textContent = label; box.append(strong, caption); $("metrics").append(box);
  }
  $("sampling").textContent = `Charts show ${view.shown_rows} of ${view.total_rows} rows: ${view.sampling}. The table previews the first ${Math.min(64, view.shown_rows)} sampled rows. Distribution summaries use all ${view.total_rows} rows.`;
  $("identity").textContent = pretty({source_task_id: view.source_task_id, episode_id: view.episode_id, archive: result.archive, generation: result.run?.summary, elapsed_seconds: result.run?.elapsed_seconds, execution: view.recipe.execution, ...(view.feature_schema ? {feature_schema:view.feature_schema,target_origin:view.target_origin} : {})});
  $("feature").replaceChildren(...view.features.map(feature => new Option(`${feature.name} · ${feature.role}`, feature.index)));
  $("channel").replaceChildren(...(view.classes.length ? view.classes : ["Signal"]).map((name, i) => new Option(name, i)));
  $("class-panel").hidden = !view.classification;
  if (view.classification) {
    const stats = view.classification;
    table("class-counts", ["Class", "Observed", "Context", "Query", "Expected"], view.classes.map((name, i) => [name, stats.observed_counts[i], stats.context_counts[i], stats.query_counts[i], stats.expected_counts[i]]));
  }
  $("selected-row").hidden = true; $("parameter-values").textContent = "";
  $("row-index").max = view.total_rows - 1; $("row-index").value = 0;
  $("archive-path").value = result.archive; $("saved-id").value = view.recipe.task_id;
  $("replay-record").value = result.recipe_json; $("seed").value = view.recipe.seed; $("task-id").value = view.recipe.task_id;
  $("curve-row").value = 0; $("curve-row").max = view.total_rows - 1;
  $("gp-approximate").checked = false;
  $("construction-details").hidden=!view.construction_details;
  $("construction-detail-notice").textContent=view.construction_details?.notice || "";
  $("construction-detail-record").textContent=pretty(view.construction_details || {});
  fillControls(view.recipe.recipe); charts(); clearCurve(); resetGraph(); resetPair(); resetWorkbench(); nativeViewControls(view);
}
async function inspectRow(index) {
  await act("Loading row…", async () => {
    const result = await request("row", {index}); $("selected-row").hidden = false; $("selected-row").textContent = pretty(result.row); $("row-index").value = index;
  });
}
$("generator").onsubmit = event => {event.preventDefault(); act("Generating and saving dataset…", generate);};
$("next").onclick = () => act("Generating next task…", async () => {$("task-id").value = String(BigInt($("task-id").value) + 1n); await generate();});
$("family").onchange = () => fillControls(examples[$("family").value]);
for (const id of ["rows", "context", "features", "active", "classes", "difficulty"]) $(id).onchange = () => act("Updating recipe…", async () => updateRecipe(id));
$("recipe").onchange = () => act("Updating controls…", async () => fillControls(parseRecipe()));
$("task-type").onchange = () => {
  const kind = $("task-type").value; $("classes").value = kind === "regression" ? 0 : kind === "binary" ? 2 : 3;
  $("difficulty").value = kind === "regression" ? 0 : 1;
  $("difficulty-label").firstChild.textContent = kind === "regression" ? "Noise / signal" : "Class separation";
  act("Updating recipe…", async () => updateRecipe("task-type"));
};
$("feature").onchange = () => {charts(); clearCurve();};
$("channel").onchange = () => {charts(); drawCurve();};
$("inspect-row").onclick = () => inspectRow(number("row-index"));
$("replay").onclick = () => act("Replaying saved recipe…", async () => show(await request("generate", {record_json: $("replay-record").value, ...limits()})));
$("open").onclick = () => act("Verifying saved dataset…", async () => show(await request("open", {archive: $("archive-path").value, task_id: $("saved-id").value})));
$("parameters").onclick = () => act("Loading sampled parameters…", async () => {$("parameter-values").textContent = pretty((await request("parameters")).parameters);});
$("recipe-download").onclick = () => download("recipe.json", new Blob([current.recipe_json], {type: "application/json"}));
$("export").onclick = () => act("Exporting every row…", async () => {
  const result = await request("export", {max_bytes: number("bytes") * 1024 ** 2});
  download(result.name, new Blob([Uint8Array.from(atob(result.data), c => c.charCodeAt(0))], {type: result.mime}));
});
request("examples").then(result => {
  examples = result.examples; $("family").replaceChildren(...Object.keys(examples).map(name => new Option(name, name)));
  $("output-dir").textContent = `Output folder: ${result.output_dir}`; fillControls(examples[$("family").value]);
}).catch(error => {$("status").className = "error"; $("status").textContent = error.message;});
