import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {openBrowser} from './explorer_browser.mjs';
const [port,url,out]=process.argv.slice(2), b=await openBrowser(Number(port),url,out), results=[];
try {
 for(const family of ['student_t','mixture','ball']) {
  await b.evaluate(`{$('construction-panel').open=true;$('construction-family').value='Pipeline innovations: ${family} (r0 extension)';$('construction-family').onchange();const c=JSON.parse($('construction-controls').value);c.task.rows=16;c.task.context_rows=8;c.calibration_rows=8;c.width_prior.low=1;c.width_prior.high=3;$('construction-controls').value=JSON.stringify(c);$('seed').value='9007199254740997';$('task-id').value='9007199254740999';$('innovation-factor').value='gaussian';$('innovation-degrees').value='1';$('innovation-strengths').value='0.1';$('innovation-powers').value='1';$('innovation-truncation').value='none';$('innovation-low').value='0.2';$('innovation-high').value='0.2';}`);
  assert.equal(await b.evaluate("$('innovation-controls').hidden"),false);
  await b.click('innovation-apply');await b.click('construct');
  const original=await b.evaluate('({json:current.recipe_json,archive:current.archive,source:current.view.source_task_id,id:current.view.episode_id})');
  assert.equal(await b.evaluate('current.view.construction_details.kind'),'r0-innovation-pipelines');
  assert.ok(await b.evaluate('current.view.construction_details.innovations.every(n=>n.scales.every(v=>v===0.2))'));
  await b.evaluate("$('graph-rows').value='15,0,15'");await b.click('graph');
  assert.deepEqual(await b.evaluate('generationGraph.rows'),[15,0,15]);
  assert.ok(await b.evaluate('generationGraph.nodes.some(n=>n.id.startsWith("exogenous-"))'));
  await b.click('export');await b.downloaded(`dataset-${original.source}.zip`);
  await b.click('mechanism-load');
  await b.evaluate("{const r=JSON.parse($('mechanism-recipe').value);r.nodes[0].noise_scale=r.nodes[0].noise_scale.map(()=>0);$('mechanism-recipe').value=JSON.stringify(r)}");
  await b.click('mechanism-save');assert.equal(await b.evaluate('current.recipe_json'),original.json);
  await b.click('edit-open');await b.click('edit-export');await b.downloaded('mechanism-edit.zip',results.length+1);
  await b.evaluate(`$('archive-path').value=${JSON.stringify(original.archive)};$('saved-id').value='9007199254740999'`);await b.click('open');
  assert.equal(await b.evaluate('current.recipe_json'),original.json);
  await b.screenshot(`innovation-${family}.png`);results.push(original);
 }
 await b.evaluate("$('innovation-degrees').value='0'");await b.click('innovation-apply');
 await b.evaluate("$('construct').click()");await b.waitFor('!busy');
 assert.equal(await b.evaluate("$('status').className"),'error');
 assert.equal(await b.evaluate('current.view.episode_id'),results.at(-1).id);
 await b.send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 assert.equal(await b.evaluate('document.documentElement.scrollWidth>innerWidth+2'),false);
 await b.screenshot('innovation-mobile.png');assert.deepEqual(b.exceptions,[]);
 await fs.writeFile(path.join(out,'receipt.json'),JSON.stringify({results,downloads:b.downloads,exceptions:b.exceptions},null,2));
} finally {b.close();}
