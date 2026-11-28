import fs from "node:fs/promises";
import path from "node:path";
import assert from "node:assert/strict";
import {openBrowser} from "./explorer_browser.mjs";

const [port,url,out]=process.argv.slice(2),b=await openBrowser(Number(port),url,out);
try {
  const preset="Prototype root: dense / multinomial-trial-temperature-r0 / ceil-r0";
  assert.ok(await b.evaluate(`Array.from($('construction-family').options).some(o=>o.value===${JSON.stringify(preset)})`),"prototype preset missing");
  await b.evaluate(`$('construction-panel').open=true;$('construction-family').value=${JSON.stringify(preset)};$('construction-family').onchange();$('seed').value='11';$('task-id').value='18014398509481987';{const c=JSON.parse($('construction-controls').value);c.source.source.task.rows=16;c.source.source.task.context_rows=8;$('construction-controls').value=JSON.stringify(c);}`);
  assert.equal(await b.evaluate("$('prototype-controls').hidden"),false);
  await b.evaluate("$('prototype-seed').value='18014398509481985';$('prototype-probabilities').value='[0.1,0.2,0.3,0.4]';$('prototype-beta').value='2'");
  await b.click("prototype-apply");await b.click("construct");
  const raw=await b.evaluate("current.recipe_json"), root=JSON.parse(raw).recipe.root;
  assert.equal(root.version,"prototype-root-base-v1");assert.equal(root.bank.root.weights.beta,2);
  assert.equal(root.bank.root.prototypes.length,4);
  assert.equal(await b.evaluate("current.view.construction_details.source_root.kind"),"prototype-convex-mixture-root");
  await b.screenshot("prototype-controls.png");
  await b.click("next");assert.equal(await b.evaluate("$('task-id').value"),"18014398509481988");
  assert.deepEqual(JSON.parse(await b.evaluate("current.recipe_json")).recipe.root,root);
  await b.evaluate(`$('replay-record').value=${JSON.stringify(raw)}`);await b.click("replay");
  await b.evaluate("$('prototype-lifecycle').value='prototype-batch-reference-v1';$('prototype-rho').value='0.25'");await b.click("prototype-apply");
  await b.click("construct");
  assert.equal(JSON.parse(await b.evaluate("current.recipe_json")).recipe.version,"prototype-batch-network-realized-v1");
  await b.evaluate(`$('replay-record').value=${JSON.stringify(raw)}`);await b.click("replay");
  let rejected=false;
  assert.equal(await b.evaluate("current.recipe_json"),raw);
  await b.evaluate("$('prototype-lifecycle').value='frozen-independent-prototype-bank-r0-v1';$('prototype-rho').value='0.125'");await b.click("prototype-apply");
  await b.click("mechanism-load");assert.equal(await b.evaluate("$('source-root-editor').hidden"),false);
  assert.equal(await b.evaluate("$('source-root-json').readOnly"),true);
  await b.evaluate("$('source-root-edit-policy').value='authored';$('source-root-edit-policy').onchange();$('source-root-json').value='{}'");
  const diagram=await b.evaluate("$('network-structure').innerHTML");
  rejected=false;try{await b.click("source-root-edit-apply");}catch{rejected=true;}assert.ok(rejected);
  assert.equal(await b.evaluate("$('source-root-json').value"),"{}");assert.equal(await b.evaluate("$('network-structure').innerHTML"),diagram);
  const authored={version:"prototype-root-base-v1",bank:structuredClone(root.bank.root)};
  authored.bank.prototypes[0][0]+=.125;
  await b.evaluate(`$('source-root-json').value=${JSON.stringify(JSON.stringify(authored))}`);
  await b.click("source-root-edit-apply");await b.click("network-save");
  assert.deepEqual(await b.evaluate("current.edit.after.recipe.recipe.root"),authored);
  assert.equal(await b.evaluate("current.recipe_json"),raw);
  await b.click("edit-open");await b.click("edit-export");await b.downloaded("mechanism-edit.zip",1);
  const trace=await b.evaluate("request('graph',{indices:[15,0,15],...limits()})");assert.deepEqual(trace.graph.rows,[15,0,15]);
  const curve=await b.evaluate("request('curve',{feature:0,points:7,...limits()})");assert.equal(curve.curve.scores.length,7);
  assert.ok(await b.evaluate("request('graph',{indices:[0],...limits(),max_seconds:0.000001}).then(()=>false,()=>true)"));
  assert.equal(await b.evaluate("current.recipe_json"),raw);
  for(const mode of ["multinomial-probability-temperature-r0","dirichlet-concentration-r0"]){
    await b.evaluate(`$('prototype-weight-mode').value=${JSON.stringify(mode)};$('prototype-weight-mode').onchange()`);
    await b.click("prototype-apply");await b.click("construct");
    const retained=JSON.parse(await b.evaluate("current.recipe_json")).recipe.root.bank.root.weights;
    assert.equal(retained.mode,mode);if(mode.startsWith("dirichlet")) assert.equal(retained.trials,null);
  }
  await b.screenshot("prototype-dirichlet.png");assert.deepEqual(b.exceptions,[]);
  await fs.writeFile(path.join(out,"receipt.json"),JSON.stringify({status:"PASS",raw,authored,allWeightLaws:true,batchReferenceSeparate:true,exactIdentity:true,traceRows:trace.graph.rows,exceptions:b.exceptions},null,2));
} finally {b.close();}
