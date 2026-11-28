// Actual installed Chromium walkthrough; owns neither the server nor browser process.
import fs from "node:fs/promises";
import path from "node:path";
import assert from "node:assert/strict";
import {openBrowser} from "./explorer_browser.mjs";
const [port,url,directory]=process.argv.slice(2);
const b=await openBrowser(Number(port),url,directory), results=[];
async function construct(name) {
  await b.evaluate(`$('construction-panel').open=true;$('construction-family').value=${JSON.stringify(name)};$('construction-family').onchange();{const c=JSON.parse($('construction-controls').value);const t=c.skeleton?.task || c.task;t.rows=16;t.context_rows=8;if(c.fits)c.fits.forEach(f=>f.plan.rows=32);else if(c.topology!=='native')c.calibration_rows=8;if(c.width_prior){c.width_prior.low=1;c.width_prior.high=3;}if(c.pipeline_prior){c.pipeline_prior.converter.allow_activation=false;c.pipeline_prior.converter.allow_kumaraswamy=false;}$('construction-controls').value=JSON.stringify(c);}$('seed').value='9007199254740997';$('task-id').value='9007199254740999';$('seconds').value='45';`);
  await b.click("construct");await b.click("intervention-load");
  return await b.evaluate("({episode:current.view.episode_id,raw:current.recipe_json})");
}
async function hard(node,value,location="state") {
  await b.evaluate(`$('intervention-node').value=${JSON.stringify(String(node))};$('intervention-node').onchange();$('intervention-kind').value='hard';$('intervention-kind').onchange();$('intervention-json').value='[]';$('intervention-values').value=${JSON.stringify(JSON.stringify([value]))};$('intervention-location').value=${JSON.stringify(location)};$('intervention-location').onchange();`);
  await b.click("intervention-add");await b.click("intervention-create");
}
try {
  const original=await construct("Explicit causal graph");
  await hard(2,4);
  assert.equal(await b.evaluate("current.recipe_json"),original.raw);
  assert.equal(await b.evaluate("current.intervention.after.rows.every(r=>r.target===4)"),true);
  assert.equal(await b.evaluate("current.intervention.card.row_identity.seed"),"9007199254740997");
  await b.evaluate("$('intervention-rows').value='15,0,15'");await b.click("intervention-inspect");
  assert.deepEqual(await b.evaluate("interventionTrace.after.rows"),[15,0,15]);
  await b.evaluate("$('intervention-stage').value='after:'+interventionTrace.after.nodes.findIndex(n=>n.id==='intervention-2-state');$('intervention-stage').onchange()");
  assert.equal(await b.evaluate("$('intervention-trace-table').textContent.includes('4')"),true);
  await b.click("intervention-export");await b.downloaded("causal-intervention-pair.zip");
  const saved=await b.evaluate("({path:current.intervention.path,id:current.intervention.after_id})");
  await b.click("intervention-open");assert.equal(await b.evaluate("current.intervention.after_id"),saved.id);
  await b.evaluate("$('intervention-json').value='broken';$('intervention-create').click()");await b.waitFor("!busy");
  assert.equal(await b.evaluate("$('status').className"),"error");assert.equal(await b.evaluate("current.intervention.after_id"),saved.id);
  await b.click("intervention-load");await b.evaluate("$('intervention-node').value='1';$('intervention-node').onchange();$('intervention-kind').value='soft';$('intervention-kind').onchange();{const f=JSON.parse($('intervention-function').value);f.matrix=f.matrix.map(row=>row.map(v=>-v));$('intervention-function').value=JSON.stringify(f);}");
  await b.click("intervention-add");await b.click("intervention-create");
  assert.equal(await b.evaluate("current.intervention.card.changes[0].version"),"soft-causal-intervention-v1");
  await b.evaluate("$('intervention-mode').value='population';$('intervention-mode').onchange();$('intervention-population').value='9007199254740993'");await b.click("intervention-create");
  assert.equal(await b.evaluate("current.intervention.after.recipe.recipe.population"),"9007199254740993");
  assert.equal(await b.evaluate("current.intervention.card.mode"),"population");
  const population=await b.evaluate("current.intervention.after_id");
  await b.evaluate("$('seconds').value='0.000001';$('intervention-create').click()");await b.waitFor("!busy");
  assert.equal(await b.evaluate("$('status').className"),"error");assert.equal(await b.evaluate("current.intervention.after_id"),population);
  await b.evaluate("$('seconds').value='45';$('intervention-panel').scrollIntoView()");await b.screenshot("intervention-comparison.png");results.push(saved);
  await construct("Numerical pipeline: readout-only");
  const target=await b.evaluate("current.view.recipe.recipe.graph.runtime.target[0][0]");
  assert.equal(await b.evaluate(`interventionInfo.nodes[${target}].captured_channels.includes(0)`),true);
  await b.evaluate("$('intervention-mode').value='matched';$('intervention-mode').onchange()");await hard(target,4);
  assert.equal(await b.evaluate("current.intervention.after.rows.every((r,i)=>r.target===current.view.rows[i].target)"),true);
  await hard(target,4,"state_and_readout");assert.equal(await b.evaluate("current.intervention.after.rows.every(r=>r.target===4)"),true);
  await b.screenshot("intervention-pipeline-locations.png");
  await construct("Native mixed graph (authored)");
  await b.evaluate("$('intervention-node').value='3';$('intervention-node').onchange();$('intervention-kind').value='hard';$('intervention-kind').onchange();$('intervention-json').value='[]';$('intervention-category').value='blue'");
  assert.deepEqual(await b.evaluate("Array.from($('intervention-category').options).map(o=>o.value)"),["red","green","blue"]);
  await b.click("intervention-add");await b.click("intervention-create");
  assert.equal(await b.evaluate("current.intervention.after.rows.every(r=>r.target_label==='blue' && r.probabilities[2]===1)"),true);
  await b.click("intervention-inspect");await b.click("intervention-export");await b.downloaded("causal-intervention-pair.zip",2);
  await b.screenshot("intervention-native-category.png");
  const fitted=await construct("Fitted structural tree: dt");await hard(2,4);
  assert.equal(await b.evaluate("current.recipe_json"),fitted.raw);
  assert.equal(await b.evaluate("current.intervention.after.recipe.recipe.base.version"),"fitted-functional-causal-v1");
  await b.click("intervention-export");await b.downloaded("causal-intervention-pair.zip",3);
  await b.send("Emulation.setDeviceMetricsOverride",{width:390,height:844,deviceScaleFactor:1,mobile:true});
  assert.equal(await b.evaluate("document.documentElement.scrollWidth>innerWidth+2"),false);
  await b.evaluate("$('intervention-panel').scrollIntoView()");await b.screenshot("intervention-mobile.png");
  assert.deepEqual(b.exceptions,[]);
  await fs.writeFile(path.join(directory,"green.json"),JSON.stringify({results,downloads:b.downloads,exceptions:b.exceptions,fittedOriginal:fitted.raw},null,2));
} finally {b.close();}
