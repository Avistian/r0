import fs from "node:fs/promises";
import path from "node:path";
import assert from "node:assert/strict";
import {openBrowser} from "./explorer_browser.mjs";

const [port,url,out]=process.argv.slice(2),b=await openBrowser(Number(port),url,out);
try {
  const cases=[
    ["P1 paper","dense","shared width","p1-paper-tnlu-v1"],
    ["I1 pinned code","gated","individual widths","tabicl-9530bd6-tnlu-v1"]
  ];
  const records=[];
  for (const [index,[label,family,scope,law]] of cases.entries()) {
    const preset=`Neural hyperpriors: ${label} / ${family} / ${scope} (r0 architecture)`;
    await b.evaluate(`$('construction-panel').open=true;$('construction-family').value=${JSON.stringify(preset)};$('construction-family').onchange();$('construction-seed').value='0';$('seed').value='71';$('task-id').value='18014398509481987';{const c=JSON.parse($('construction-controls').value);c.task.rows=16;c.task.context_rows=8;$('construction-controls').value=JSON.stringify(c);}`);
    assert.equal(await b.evaluate("$('network-hyperprior-controls-note').hidden"),false);
    await b.click("construct");
    const raw=await b.evaluate("current.recipe_json");records.push(raw);
    assert.equal(JSON.parse(raw).recipe.network.profile.depth.family,law);
    assert.deepEqual(await b.evaluate("current.view.construction_details.stream_roles"),["hyper","named","raw"]);
    await b.click("next");
    assert.equal(await b.evaluate("$('task-id').value"),"18014398509481988");
    assert.deepEqual(JSON.parse(await b.evaluate("current.recipe_json")).recipe,JSON.parse(raw).recipe);
    await b.evaluate(`$('replay-record').value=${JSON.stringify(raw)}`);await b.click("replay");
    await b.click("mechanism-load");
    await b.evaluate("{const block=JSON.parse($('network-block-json').value);block.experts[0].bias[0]+=.25;$('network-block-json').value=JSON.stringify(block);}");
    await b.click("network-apply");await b.click("network-save");
    assert.equal(await b.evaluate("current.edit.after.recipe.recipe.network.version"),"frozen-structural-network-v1");
    assert.equal(await b.evaluate("current.recipe_json"),raw);
    await b.click("edit-open");await b.click("edit-export");
    await b.downloaded("mechanism-edit.zip",index+1);
    await fs.rename(path.join(out,"mechanism-edit.zip"),path.join(out,`profile-${index}-mechanism-edit.zip`));
    await b.waitFor("!busy");
  }
  const baseline=await b.evaluate("current.recipe_json");
  const good=await b.evaluate("$('construction-controls').value");
  await b.evaluate("{const c=JSON.parse($('construction-controls').value);c.hyperprior.depth={family:'fixed',low:100,high:100,offset:0,rounded:false,choices:[],version:'network-scalar-prior-v1'};$('construction-controls').value=JSON.stringify(c);}");
  let rejected=false;try{await b.click("construct");}catch{rejected=true;}assert.ok(rejected);
  assert.equal(await b.evaluate("current.recipe_json"),baseline);
  await b.evaluate(`$('construction-controls').value=${JSON.stringify(good)};$('seconds').value='0.001'`);
  let timedOut=false;try{await b.click("construct");}catch{timedOut=true;}assert.ok(timedOut);
  assert.equal(await b.evaluate("current.recipe_json"),baseline);
  await b.evaluate("$('seconds').value='60';$('construction-details').open=true");
  await b.screenshot("network-hyperpriors.png");
  assert.deepEqual(b.exceptions,[]);
  await fs.writeFile(path.join(out,"receipt.json"),JSON.stringify({status:"PASS",profiles:cases,records,frozenNext:true,linkedAuthoredEdits:true,capAndTimeoutPreserveOriginal:true,exceptions:b.exceptions},null,2));
} finally {b.close();}
