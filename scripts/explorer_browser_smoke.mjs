// Browser proof using an already-running Chromium CDP endpoint, with no npm packages.
// node scripts/explorer_browser_smoke.mjs <debug-port> <explorer-url> <evidence-directory>
import fs from "node:fs/promises";
import path from "node:path";
import assert from "node:assert/strict";
import net from "node:net";

const [port = "9224", url = "http://127.0.0.1:8767/", directory = "/tmp/r0-explorer-browser"] = process.argv.slice(2);
await fs.mkdir(directory, {recursive: true});
const address = new URL(url);
const idle = net.createConnection({host: address.hostname, port: Number(address.port || 80)});
await new Promise(resolve => idle.once("connect", resolve));
await new Promise(resolve => setTimeout(resolve, 100));
try {
  assert.equal((await fetch(url, {signal: AbortSignal.timeout(1500)})).status, 200);
} finally {
  idle.end();
}
const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const socket = new WebSocket(targets.find(target => target.type === "page").webSocketDebuggerUrl);
await new Promise(resolve => socket.addEventListener("open", resolve, {once: true}));
const pending = new Map(), exceptions = [], downloads = [];
let counter = 0;
socket.addEventListener("message", event => {
  const message = JSON.parse(event.data);
  if (message.method === "Runtime.exceptionThrown") exceptions.push(message.params);
  if (message.method === "Browser.downloadWillBegin") downloads.push({...message.params,state:"started"});
  if (message.method === "Browser.downloadProgress") {
    const download = downloads.find(item => item.guid === message.params.guid);
    if (download) download.state = message.params.state;
  }
  if (message.id) {
    const {resolve, reject} = pending.get(message.id);
    pending.delete(message.id);
    message.error ? reject(new Error(JSON.stringify(message.error))) : resolve(message.result);
  }
});
function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++counter;
    pending.set(id, {resolve, reject});
    socket.send(JSON.stringify({id, method, params}));
  });
}
async function evaluate(expression) {
  const result = await send("Runtime.evaluate", {expression, awaitPromise: true, returnByValue: true});
  if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
async function waitFor(expression, timeout = 60000) {
  const start = Date.now();
  while (!await evaluate(expression)) {
    if (Date.now() - start > timeout) throw new Error(`Timed out: ${expression}`);
    await new Promise(resolve => setTimeout(resolve, 100));
  }
}
async function downloaded(name, count = 1) {
  const start = Date.now();
  while (downloads.filter(item => item.suggestedFilename === name && item.state === "completed").length < count) {
    if (Date.now() - start > 30000) throw new Error(`Download did not complete: ${name}`);
    await new Promise(resolve => setTimeout(resolve, 100));
  }
}
async function screenshot(name) {
  const result = await send("Page.captureScreenshot", {captureBeyondViewport: true});
  await fs.writeFile(path.join(directory, name), Buffer.from(result.data, "base64"));
}
async function click(id) {
  await evaluate(`document.getElementById(${JSON.stringify(id)}).click()`);
  await waitFor("!busy");
  const error = await evaluate("document.getElementById('status').className === 'error' ? document.getElementById('status').textContent : null");
  if (error) throw new Error(error);
}
await send("Page.enable");
await send("Runtime.enable");
await send("Emulation.setDeviceMetricsOverride", {width: 1440, height: 1100, deviceScaleFactor: 1, mobile: false});
await send("Browser.setDownloadBehavior", {behavior: "allow", eventsEnabled: true, downloadPath: path.resolve(directory)});
await send("Page.navigate", {url});
await waitFor("document.getElementById('family')?.options.length === 10");
assert.equal(await evaluate("Boolean(document.getElementById('curve'))"), true, "response-curve controls must exist");
assert.equal(await evaluate("Boolean(document.getElementById('graph'))"), true, "generation graph controls must exist");
await screenshot("empty.png");
await click("generate");
const initial = await evaluate("({id: current.view.source_task_id, rows: current.view.total_rows, plotted: document.querySelectorAll('svg circle').length, archive: current.archive, run: current.run.summary})");
assert.equal(initial.rows, 64);
assert.equal(initial.plotted, 128);
await screenshot("regression.png");
await evaluate("document.getElementById('row-index').value = 63");
await click("inspect-row");
assert.equal(await evaluate("JSON.parse(document.getElementById('selected-row').textContent).index"), 63);
await click("replay");
assert.equal(await evaluate("current.view.source_task_id"), initial.id);
assert.equal(await evaluate("current.run.summary.generated"), 0);
await click("open");
assert.equal(await evaluate("current.view.source_task_id"), initial.id);
await click("export");
await evaluate("document.querySelector('#feature-hist .chart-actions button').click()");
await evaluate("document.querySelectorAll('#feature-hist .chart-actions button')[1].click()");
await Promise.all([downloaded(`dataset-${initial.id}.zip`), downloaded("feature-hist.svg"), downloaded("feature-hist.png")]);
const files = await fs.readdir(directory);
assert(files.some(name => name.endsWith('.zip')));
assert(files.includes('feature-hist.svg'));
assert(files.includes('feature-hist.png'));
await evaluate("document.getElementById('task-type').value='multiclass'; document.getElementById('task-type').dispatchEvent(new Event('change'))");
await waitFor("!busy");
await evaluate("{const recipe=JSON.parse(document.getElementById('recipe').value); recipe.class_offsets=[0,-2,1]; recipe.separation=7; recipe.features=10; document.getElementById('recipe').value=JSON.stringify(recipe); document.getElementById('rows').value=80; document.getElementById('rows').dispatchEvent(new Event('change'));}");
await waitFor("!busy");
assert.deepEqual(await evaluate("JSON.parse(document.getElementById('recipe').value).class_offsets"), [0, -2, 1]);
assert.equal(await evaluate("JSON.parse(document.getElementById('recipe').value).separation"), 7);
assert.equal(await evaluate("JSON.parse(document.getElementById('recipe').value).features"), 10);
await click("generate");
assert.equal(await evaluate("current.view.classes.length"), 3);
assert.equal(await evaluate("document.querySelectorAll('#class-counts tbody tr').length"), 3);
await screenshot("classification.png");
assert.equal(await evaluate("[...document.querySelectorAll('.plot svg text')].every(text => {const box=text.getBBox(); return box.x>=0 && box.x+box.width<=480;})"), true);
const families = await evaluate("Object.keys(examples)"), familyRuns = [], curveRuns = [], graphRuns = [];
for (const family of families) {
  await evaluate(`document.getElementById('family').value=${JSON.stringify(family)}; document.getElementById('family').dispatchEvent(new Event('change'))`);
  await click("generate");
  assert.equal(await evaluate("current.view.total_rows"), 64);
  familyRuns.push({family, ...await evaluate("({identity: current.view.source_task_id, run: current.run.summary})")});
  await evaluate("document.getElementById('feature').value = current.view.features.find(f => f.role === 'active').index; document.getElementById('feature').dispatchEvent(new Event('change'))");
  if (family === "Exact Gaussian process") {
    await evaluate("document.getElementById('curve').click()"); await waitFor("!busy");
    assert.match(await evaluate("document.getElementById('status').textContent"), /conditional/);
    assert.equal(await evaluate("document.querySelectorAll('#response-plot polyline').length"), 0);
    curveRuns.push({family, result: "explicit conditional-inspection requirement"});
  } else {
    await click("curve");
    assert.equal(await evaluate("document.querySelector('#response-plot polyline').points.numberOfItems"), 129);
    assert.match(await evaluate("document.getElementById('curve-note').textContent"), /fixed to row 0/);
    curveRuns.push({family, ...await evaluate("({identity:curve.episode_id,points:curve.points,feature:curve.feature,bounds:curve.bounds})")});
  }
  await click("graph");
  assert.equal(await evaluate("document.querySelectorAll('#generation-graph [data-node]').length"), await evaluate("generationGraph.nodes.length"));
  assert.equal(await evaluate("graphNode.id"), "target");
  assert.equal(await evaluate("generationGraph.target_max_error"), 0);
  assert.equal(await evaluate("[...document.querySelectorAll('#graph-detail .plot svg text')].every(text => {const box=text.getBBox(); return box.x>=0 && box.x+box.width<=480;})"), true);
  graphRuns.push({family, ...await evaluate("({identity:generationGraph.episode_id,nodes:generationGraph.nodes.length,rows:generationGraph.rows,signal_error:generationGraph.signal_max_error,target_error:generationGraph.target_max_error})")});
  if (family === "Exact Gaussian process") {
    await evaluate("document.querySelector('[data-node=kernel-0]').dispatchEvent(new MouseEvent('click'))");
    assert.match(await evaluate("document.getElementById('graph-formula').textContent"), /exp/);
    await screenshot("gp-graph.png");
    await evaluate("document.querySelector('[data-node=normals]').dispatchEvent(new MouseEvent('click'))");
    const normalRows = await evaluate("graphNode.parameters.full_shape[0]");
    assert.match(await evaluate("document.getElementById('graph-output-plot').textContent"), new RegExp(`16 of ${normalRows} support rows`));
    graphRuns.at(-1).normal_preview = {shown_rows:16,full_support_rows:normalRows};
    await evaluate("document.querySelector('#graph-output-plot .chart-actions button').click(); document.querySelectorAll('#graph-output-plot .chart-actions button')[1].click()");
    await Promise.all([downloaded("graph-output-plot.svg"),downloaded("graph-output-plot.png")]);
    await screenshot("gp-normals.png");
  }
}
await screenshot("composition.png");
await evaluate("document.querySelector('[data-node=component-1]').dispatchEvent(new MouseEvent('click'))");
assert.equal(await evaluate("document.getElementById('graph-expand').hidden"), false);
await click("graph-expand");
assert.equal(await evaluate("generationGraph.component"), 1);
assert.equal(await evaluate("generationGraph.nodes.some(node => node.id==='target')"), false);
await screenshot("component-graph.png");
await evaluate("document.getElementById('graph-component').value=''; document.getElementById('graph-component').dispatchEvent(new Event('change')); document.getElementById('graph-rows').value='0,3,63'");
await click("graph");
await evaluate("document.querySelector('[data-node=expression-0]').focus()");
await send("Input.dispatchKeyEvent", {type:"keyDown",key:"Enter",code:"Enter",windowsVirtualKeyCode:13});
await send("Input.dispatchKeyEvent", {type:"keyUp",key:"Enter",code:"Enter",windowsVirtualKeyCode:13});
assert.equal(await evaluate("graphNode.id"), "expression-0");
await evaluate("document.querySelector('#graph-downloads button').click(); document.querySelectorAll('#graph-downloads button')[1].click()");
await Promise.all([downloaded("generation-graph.svg"), downloaded("generation-graph.png")]);
assert((await fs.readdir(directory)).includes('generation-graph.svg'));
assert((await fs.readdir(directory)).includes('generation-graph.png'));
await fs.copyFile(path.join(directory,"generation-graph.png"),path.join(directory,"composition-diagram.png"));
await fs.copyFile(path.join(directory,"generation-graph.svg"),path.join(directory,"composition-diagram.svg"));
await click("graph-fit"); assert(await evaluate("graphScale") < 1);
await click("graph-zoom-label"); assert.equal(await evaluate("graphScale"),1);
await screenshot("composition-graph.png");
await evaluate("{const recipe=JSON.parse(document.getElementById('recipe').value); recipe.components=Array.from({length:64},()=>structuredClone(recipe.components[0])); recipe.expression={operation:'sum',children:Array.from({length:64},(_,i)=>i),strength:1}; document.getElementById('recipe').value=JSON.stringify(recipe);}");
await click("generate"); await click("graph");
assert.equal(await evaluate("generationGraph.component_count"), 64);
assert.equal(await evaluate("document.querySelectorAll('#generation-graph [data-node]').length"), await evaluate("generationGraph.nodes.length"));
await evaluate("document.querySelectorAll('#graph-downloads button')[1].click()");
await downloaded("generation-graph.png", 2);
const largePng = await fs.readFile(path.join(directory, "generation-graph.png"));
await fs.copyFile(path.join(directory,"generation-graph.png"),path.join(directory,"large-composition-diagram.png"));
assert.equal(Math.max(largePng.readUInt32BE(16), largePng.readUInt32BE(20)), 4096);
const largeGraph = await evaluate("({nodes:generationGraph.nodes.length,components:generationGraph.component_count})");
await evaluate("document.getElementById('family').value='Composition'; document.getElementById('family').dispatchEvent(new Event('change'))");
await click("generate");


await evaluate("document.getElementById('curve-row').value=63; document.getElementById('curve-row').dispatchEvent(new Event('change')); document.getElementById('curve-points').value=65");
assert.equal(await evaluate("document.querySelectorAll('#response-plot polyline').length"), 0);
await click("curve");
assert.equal(await evaluate("curve.anchor_row"), 63);
assert.equal(await evaluate("document.querySelector('#response-plot polyline').points.numberOfItems"), 65);
await evaluate("document.querySelector('#response-plot .chart-actions button').click(); document.querySelectorAll('#response-plot .chart-actions button')[1].click()");
await Promise.all([downloaded("response-plot.svg"), downloaded("response-plot.png")]);
assert((await fs.readdir(directory)).includes('response-plot.svg'));
assert((await fs.readdir(directory)).includes('response-plot.png'));
await screenshot("composition-response.png");
await evaluate("document.getElementById('feature').value=current.view.features.find(f => f.role==='irrelevant').index; document.getElementById('feature').dispatchEvent(new Event('change'))");
assert.equal(await evaluate("document.querySelectorAll('#response-plot polyline').length"), 0);
await click("curve");
assert.equal(await evaluate("curve.scores.every(row => row.every((value,i) => value===curve.scores[0][i]))"), true);

await evaluate("document.getElementById('active').value=3; document.getElementById('active').dispatchEvent(new Event('change'))");
await waitFor("!busy");
await click("generate");
assert.equal(await evaluate("current.view.recipe.recipe.components[0].recipe.active_features"), 3);
await evaluate("{const recipe=JSON.parse(document.getElementById('recipe').value); recipe.components[0].axes=[0,1]; recipe.components[0].recipe.active_features=2; document.getElementById('recipe').value=JSON.stringify(recipe); document.getElementById('active').value=4; document.getElementById('active').dispatchEvent(new Event('change'));}");
await waitFor("!busy");
assert.deepEqual(await evaluate("JSON.parse(document.getElementById('recipe').value).components[0].axes"), [0, 1]);
assert.equal(await evaluate("JSON.parse(document.getElementById('recipe').value).components[0].recipe.active_features"), 2);
await click("generate");
await evaluate("document.getElementById('family').value='Linear'; document.getElementById('family').dispatchEvent(new Event('change')); document.getElementById('seed').value='9223372036854775807'; document.getElementById('task-id').value='9007199254741115'");
await click("generate");
assert.equal(await evaluate("document.getElementById('seed').value"), "9223372036854775807");
assert.equal(await evaluate("document.getElementById('task-id').value"), "9007199254741115");
const largeIdentity = await evaluate("current.view.source_task_id");
await click("replay");
assert.equal(await evaluate("current.view.source_task_id"), largeIdentity);
await evaluate("document.getElementById('seed').value='27'; document.getElementById('task-id').value='0'");
await evaluate("{const recipe=JSON.parse(document.getElementById('recipe').value); Object.assign(recipe,{rows:20000,context_rows:15000,features:256,signal_features:256,active_features:32}); document.getElementById('recipe').value=JSON.stringify(recipe);}");
await click("generate");
assert.equal(await evaluate("current.view.total_rows"), 20000);
assert.equal(await evaluate("current.view.features.length"), 256);
assert.equal(await evaluate("document.querySelectorAll('#table tbody tr').length"), 64);
await screenshot("large-linear.png");
const largeTask = await evaluate("({rows:current.view.total_rows, features:current.view.features.length, shown:current.view.shown_rows, table_rows:document.querySelectorAll('#table tbody tr').length, run:current.run.summary})");
await evaluate("request('generate',{recipe:examples.Linear,seed:27,task_id:999})");
await evaluate("document.getElementById('parameters').click()");
await waitFor("!busy");
assert.match(await evaluate("document.getElementById('status').textContent"), /active dataset changed/);
await click("replay");
await evaluate("document.getElementById('family').value='Smooth radial'; document.getElementById('family').dispatchEvent(new Event('change')); document.getElementById('rows').value=513; document.getElementById('rows').dispatchEvent(new Event('change'))");
await waitFor("!busy");
await click("generate");
assert.equal(await evaluate("current.view.total_rows"), 513);
assert.equal(await evaluate("current.view.shown_rows"), 512);
const retained = await evaluate("current.view.source_task_id");
await evaluate("document.getElementById('recipe').value='{\"rows\":0}'; document.getElementById('generate').click()");
await waitFor("!busy");
assert.equal(await evaluate("document.getElementById('status').className"), "error");
assert.equal(await evaluate("current.view.source_task_id"), retained);
await send("Emulation.setDeviceMetricsOverride", {width: 390, height: 844, deviceScaleFactor: 1, mobile: true});
await click("graph");
await screenshot("mobile.png");
assert.equal(await evaluate("document.documentElement.scrollWidth > innerWidth"), false);
assert.deepEqual(exceptions, []);
await fs.writeFile(path.join(directory, "receipt.json"), JSON.stringify({url, browser: await send("Browser.getVersion"), node: process.version, initial, familyRuns, curveRuns, graphRuns, largeGraph, largeIdentity, largeTask, files, downloads, checks: ["generation graph all ten families", "component expansion", "graph node keyboard selection", "graph SVG/PNG download", "graph numerical label bounds", "analytic curves for nine families", "GP explicitly rejects analytic curve", "curve anchor and point controls", "irrelevant feature constant curve", "curve SVG/PNG exports", "generate all ten starting recipes", "row selection", "replay zero-new-generation", "open saved task", "ZIP/SVG/PNG downloads", "classification counts", "advanced recipe preservation", "whole-axis and subset composition controls", "63-bit identity replay", "20000 rows and 256 features", "full vs sampled rows", "failed generation preserves view", "mobile no horizontal overflow"], exceptions}, null, 2));
socket.close();
console.log(`Browser checks passed; evidence in ${directory}`);
