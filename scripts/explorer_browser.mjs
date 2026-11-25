// Shared CDP transport for the actual explorer browser walkthroughs.
import fs from "node:fs/promises";
import path from "node:path";

export async function openBrowser(port, url, directory) {
  await fs.mkdir(directory, {recursive: true});
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
  return {send, evaluate, waitFor, downloaded, screenshot, click, exceptions, downloads, close: () => socket.close()};
}
