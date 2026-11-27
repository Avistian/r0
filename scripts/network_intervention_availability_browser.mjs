import fs from "node:fs/promises";
import path from "node:path";
import assert from "node:assert/strict";
import {openBrowser} from "./explorer_browser.mjs";
const [port,url,out]=process.argv.slice(2),b=await openBrowser(Number(port),url,out);
try {
 await b.evaluate("$('construction-family').value='Standalone network: dense';$('construction-family').onchange();{const c=JSON.parse($('construction-controls').value);c.task.rows=16;c.task.context_rows=8;$('construction-controls').value=JSON.stringify(c)}");
 await b.click("construct");
 assert.equal(await b.evaluate("$('intervention-load').hidden"),true);
 assert.ok(await b.evaluate("$('intervention-notice').textContent.includes('explicit causal graph')"));
 await b.click("mechanism-load");
 assert.equal(await b.evaluate("$('intervention-load').hidden"),true);
 await b.screenshot("standalone.png");
 const recipe={"task": {"rows": 16, "features": 2, "signal_features": 2, "active_features": 2, "context_rows": 8, "split": 0, "noise_ratio": 0.0, "task_type": "regression", "classes": 0, "separation": 1.0, "class_offsets": [], "version": "linear-v1"}, "nodes": [{"parents": [], "function": {"matrix": [[1.0, 0.0], [0.0, 1.0]], "version": "frozen-linear-function-v1"}, "root": {"family": "gaussian", "width": 2, "gaussian_factor": null, "weights": null, "factor": null, "covariance_base": null, "version": "joint-root-base-v1"}, "noise_scale": [], "exogenous": {"inputs": {"distribution": "gaussian", "location": 0.0, "scale": 1.0, "df": 5.0, "mixture_gap": 2.0, "bound": null, "version": "inputs-v1"}, "version": "exogenous-v1", "factor": [], "geometry": "independent"}, "noise_response": {"offset": 1.0, "strength": 0.0, "power": 1.0}, "pipeline": null, "version": "functional-causal-node-v1"}, {"parents": [0], "function": {"input_width": 2, "blocks": [{"parents": [0], "experts": [{"matrix": [[2.0, -1.0]], "bias": [0.5], "mask": [], "multiplier": 1.0, "activation": {"kind": "identity", "parameters": []}, "placement": "after"}], "gate": null, "skips": []}], "version": "frozen-structural-network-v1"}, "root": null, "noise_scale": [], "exogenous": {"inputs": {"distribution": "gaussian", "location": 0.0, "scale": 1.0, "df": 5.0, "mixture_gap": 2.0, "bound": null, "version": "inputs-v1"}, "version": "exogenous-v1", "factor": [], "geometry": "independent"}, "noise_response": {"offset": 1.0, "strength": 0.0, "power": 1.0}, "pipeline": null, "version": "functional-causal-node-v1"}], "observed": [[0, 0], [0, 1]], "target": [[1, 0]], "limits": {"nodes": 1024, "node_channels": 1024, "total_channels": 8192, "parameters": 1000000, "working_bytes": 268435456}, "version": "functional-causal-v1"};
 await b.evaluate(`(async()=>{show(await request('generate',{recipe:${JSON.stringify(recipe)},seed:3,task_id:4,max_seconds:30}));})()`);
 assert.equal(await b.evaluate("$('intervention-load').hidden"),false);
 await b.click("intervention-load");
 await b.evaluate("$('intervention-json').value=JSON.stringify([{version:'hard-causal-intervention-v1',node:0,channels:[0,1],values:[3,4],category:null,location:'state'}])");
 await b.click("intervention-create");
 assert.ok(await b.evaluate("current.intervention.after.rows.every(r=>r.target===2.5)"));
 await b.screenshot("causal-network.png");assert.deepEqual(b.exceptions,[]);
 await fs.writeFile(path.join(out,"receipt.json"),JSON.stringify({status:"PASS",network_target:2.5,exceptions:b.exceptions}));
} finally {b.close();}
