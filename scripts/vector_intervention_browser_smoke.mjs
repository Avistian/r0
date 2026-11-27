// Real Chromium coverage of logical-variable clamps and domain changes.
import fs from "node:fs/promises";
import path from "node:path";
import assert from "node:assert/strict";
import {openBrowser} from "./explorer_browser.mjs";
const [port,url,directory]=process.argv.slice(2), b=await openBrowser(Number(port),url,directory);
const results=[];
try {
  for(const family of ["Native heterogeneous vector graph","Native vector parent composition"]) {
    await b.evaluate(`$('construction-panel').open=true;$('construction-family').value=${JSON.stringify(family)};$('construction-family').onchange();{const c=JSON.parse($('construction-controls').value);c.rows=16;c.context_rows=8;c.calibration_rows=8;$('construction-controls').value=JSON.stringify(c);}$('seconds').value='45';`);
    await b.click("construct");await b.click("intervention-load");
    const original=await b.evaluate("current.recipe_json");
    await b.evaluate("$('intervention-node').value='1';$('intervention-node').onchange();$('intervention-variable').value='0';$('intervention-variable').onchange();$('intervention-kind').value='hard';$('intervention-kind').onchange();$('intervention-category').selectedIndex=4;");
    assert.equal(await b.evaluate("$('intervention-variable').options.length"),3);
    const category=await b.evaluate("$('intervention-category').value");
    await b.click("intervention-add");await b.click("intervention-create");
    assert.equal(await b.evaluate("current.recipe_json"),original);
    assert.equal(await b.evaluate(`current.intervention.after.rows.every(r=>r.target_label===${JSON.stringify(category)} && r.probabilities[4]===1)`),true);
    await b.evaluate("$('intervention-rows').value='15,0,15'");await b.click("intervention-inspect");
    assert.deepEqual(await b.evaluate("interventionTrace.after.rows"),[15,0,15]);
    assert.equal(await b.evaluate("interventionTrace.after.nodes.some(n=>n.id==='intervention-1-forced-probability:0')"),true);
    await b.click("intervention-export");await b.downloaded("causal-intervention-pair.zip",results.length+1);
    const saved=await b.evaluate("current.intervention.after_id");await b.click("intervention-open");
    assert.equal(await b.evaluate("current.intervention.after_id"),saved);
    await b.click("intervention-load");
    await b.evaluate("$('intervention-node').value='1';$('intervention-node').onchange();$('intervention-variable').value='1';$('intervention-variable').onchange();$('intervention-kind').value='hard';$('intervention-kind').onchange();$('intervention-values').value='[7]'");
    await b.click("intervention-add");await b.click("intervention-create");
    assert.equal(await b.evaluate("current.intervention.after.rows.every(r=>r.inputs[3]===7)"),true);
    assert.equal(await b.evaluate("current.intervention.after.target_emission"),"softmax");
    await b.evaluate("$('intervention-json').value='[]';$('intervention-variable').value='0';$('intervention-variable').onchange();$('intervention-kind').value='domain';$('intervention-kind').onchange();{const d=JSON.parse($('intervention-domain-json').value);d.emission='argmax';d.temperature=1;$('intervention-domain-json').value=JSON.stringify(d)}");
    await b.click("intervention-add");await b.click("intervention-create");
    assert.equal(await b.evaluate("current.intervention.after.target_emission"),"argmax");
    assert.equal(await b.evaluate("current.intervention.after.original_target_domain.emission"),"softmax");
    const active=await b.evaluate("current.intervention.after_id");
    await b.evaluate("$('intervention-json').value='[{\"version\":\"native-vector-hard-intervention-v1\",\"node\":1,\"variable\":99,\"value\":4,\"category\":null}]';$('intervention-create').click()");await b.waitFor("!busy");
    assert.equal(await b.evaluate("$('status').className"),"error");
    assert.equal(await b.evaluate("current.intervention.after_id"),active);
    results.push({family,original,category,after:active});
  }
  await b.evaluate("$('intervention-panel').scrollIntoView()");await b.screenshot("vector-intervention.png");
  await b.send("Emulation.setDeviceMetricsOverride",{width:390,height:844,deviceScaleFactor:1,mobile:true});
  assert.equal(await b.evaluate("document.documentElement.scrollWidth>innerWidth+2"),false);
  await b.screenshot("vector-intervention-mobile.png");assert.deepEqual(b.exceptions,[]);
  await fs.writeFile(path.join(directory,"receipt.json"),JSON.stringify({results,downloads:b.downloads,exceptions:b.exceptions},null,2));
} finally {b.close();}
