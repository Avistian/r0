import fs from "node:fs/promises";
import path from "node:path";
import assert from "node:assert/strict";
import {openBrowser} from "./explorer_browser.mjs";
const [port,url,out]=process.argv.slice(2), b=await openBrowser(Number(port),url,out);
try {
 await b.evaluate("$('construction-panel').open=true;$('construction-family').value='Standalone network: sparse + gated + skip (authored)';$('construction-family').onchange();{const c=JSON.parse($('construction-controls').value);c.task.rows=16;c.task.context_rows=8;c.network.blocks=[{parents:[0],experts:[{matrix:[[2,3]],bias:[0.1],mask:[],multiplier:1,activation:{kind:'identity',parameters:[]},placement:'after'}],gate:null,skips:[]}];$('construction-controls').value=JSON.stringify(c);}");
 await b.click("construct");
 await b.evaluate("$('pair-intervention').value='sparsity';pairControls();$('pair-value').value='1';$('pair-estimate').checked=true");
 await b.click("pair-create");
 assert.ok(await b.evaluate("$('pair-variance').textContent.includes('baseline mean')"));
 const matched=await b.evaluate("current.pair.after.recipe_json");
 await b.evaluate(`$('replay-record').value=${JSON.stringify(matched)}`);
 await b.click("replay");
 assert.equal(await b.evaluate("$('intervention-load').hidden"),true);
 await b.click("next");
 const source=JSON.parse(await b.evaluate("current.recipe_json")).recipe;
 assert.deepEqual(source,JSON.parse(matched).recipe);
 await b.click("curve");
 for (const policy of ["recalibrate","raw"]) {
  await b.click("mechanism-load");
  assert.equal(await b.evaluate("$('network-matching-policy-label').hidden"),false);
  await b.evaluate(`$('network-matching-policy').value='${policy}';{const block=JSON.parse($('network-block-json').value);block.experts[0].matrix=[[4,5]];$('network-block-json').value=JSON.stringify(block);}`);
  await b.click("network-apply");
  await b.click("network-save");
  const edit=await b.evaluate("current.edit.after.recipe.recipe");
  if(policy==="recalibrate"){
   assert.equal(edit.version,"network-input-view-v2");
   assert.deepEqual(edit.matching.population,source.matching.population);
   assert.notDeepEqual(edit.matching.before_scores,source.matching.before_scores);
  } else assert.equal(edit.version,"network-input-view-v1");
  assert.deepEqual(edit.active_inputs,source.active_inputs);
 }
 assert.deepEqual(b.exceptions,[]);
 await b.screenshot("matched-edit.png");
 await fs.writeFile(path.join(out,"receipt.json"),JSON.stringify({status:"PASS",matching:true,frozenNext:true,curve:true,policies:["recalibrate","raw"],populationPreserved:true,exceptions:b.exceptions},null,2));
} finally {b.close();}
