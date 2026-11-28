import fs from "node:fs/promises";
import path from "node:path";
import assert from "node:assert/strict";
import {openBrowser} from "./explorer_browser.mjs";
const [port,url,out]=process.argv.slice(2), b=await openBrowser(Number(port),url,out), results=[];
try {
 for (const classification of [false,true]) {
  await b.evaluate(`$('construction-panel').open=true;$('construction-family').value='Standalone network: dense';$('construction-family').onchange();{const c=JSON.parse($('construction-controls').value);c.task.rows=16;c.task.context_rows=8;if(${classification}){c.task.task_type='multiclass';c.task.classes=3;c.network_prior.widths[c.network_prior.widths.length-1]=3;}$('construction-controls').value=JSON.stringify(c);}`);
  await b.click("construct");
  const original=await b.evaluate("current.recipe_json");
  const choices=await b.evaluate("Array.from($('pair-intervention').options,o=>o.value)");
  assert.deepEqual(choices,[classification?"separation":"noise","irrelevant","sparsity"]);
  assert.ok(await b.evaluate("$('pair-policy-note').textContent.includes('saved network')"));
  await b.evaluate(`$('pair-value').value='${classification?0:.5}'`);
  await b.click("pair-create");
  assert.equal(await b.evaluate("current.recipe_json"),original);
  const card=await b.evaluate("current.pair.card");
  assert.equal(card.intervention,classification?"separation":"noise");
  await b.click("pair-open");
  assert.equal(await b.evaluate("current.recipe_json"),original);
  await b.screenshot(classification?"classification.png":"regression.png");
  results.push({classification,choices,card});
 }
 assert.deepEqual(b.exceptions,[]);
 await fs.writeFile(path.join(out,"receipt.json"),JSON.stringify({results,exceptions:b.exceptions},null,2));
} finally {b.close();}
