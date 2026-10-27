// Optional browser QA without npm dependencies:
// node tests/check_mpc_explainer_browser.mjs /absolute/path/to/chrome
// Uses Chrome's debugging pipe; no web server or hosted page is required.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';

const binary = process.argv[2];
if (!binary) throw new Error('Pass the path to an already-installed Chrome/Chromium binary');
const directory = await mkdtemp(join(tmpdir(), 'fed-skdpp-explainer-qa-'));
const child = spawn(binary, ['--headless', '--no-sandbox', '--disable-dev-shm-usage',
  '--disable-gpu', '--disable-background-networking', '--disable-component-update',
  '--no-first-run', '--no-default-browser-check', '--remote-debugging-pipe',
  `--user-data-dir=${join(directory, 'browser-profile')}`, 'about:blank'],
  {stdio: ['ignore', 'ignore', 'pipe', 'pipe', 'pipe']});
let sequence = 0, buffer = '', stderr = '';
const pending = new Map(), exceptions = [], network = [];
child.stderr.on('data', value => { stderr = (stderr + value.toString()).slice(-4000); });
child.on('error', error => { for (const {reject} of pending.values()) reject(error); });
child.on('exit', () => { for (const {reject} of pending.values()) reject(new Error(`Chrome exited: ${stderr}`)); });
for (const pipe of [child.stdio[3], child.stdio[4]]) pipe.on('error', error => {
  for (const {reject} of pending.values()) reject(new Error(`Chrome debugging pipe: ${error.message}; ${stderr}`));
});
child.stdio[4].on('data', chunk => {
  buffer += chunk.toString();
  let delimiter;
  while ((delimiter = buffer.indexOf('\0')) >= 0) {
    const raw = buffer.slice(0, delimiter); buffer = buffer.slice(delimiter + 1);
    if (!raw) continue;
    const message = JSON.parse(raw);
    if (message.id && pending.has(message.id)) {
      const request = pending.get(message.id); pending.delete(message.id);
      clearTimeout(request.timeout);
      if (message.error) request.reject(new Error(JSON.stringify(message.error)));
      else request.resolve(message.result);
    }
    if (message.method === 'Runtime.exceptionThrown') exceptions.push(message.params.exceptionDetails);
    if (message.method === 'Network.requestWillBeSent') network.push(message.params.request.url);
  }
});
function send(method, params = {}, sessionId) {
  const id = ++sequence;
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => { pending.delete(id); reject(new Error(`CDP timeout: ${method}; ${stderr}`)); }, 15000);
    pending.set(id, {resolve, reject, timeout});
    child.stdio[3].write(JSON.stringify({id, method, params, ...(sessionId ? {sessionId} : {})}) + '\0');
  });
}
let session;
async function evaluate(expression) {
  const response = await send('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true}, session);
  if (response.exceptionDetails) throw new Error(JSON.stringify(response.exceptionDetails));
  return response.result.value;
}
async function screenshot(name) {
  await evaluate('new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))');
  const image = await send('Page.captureScreenshot', {format: 'png', captureBeyondViewport: false}, session);
  await writeFile(join(directory, name), Buffer.from(image.data, 'base64'));
}
try {
  const {targetId} = await send('Target.createTarget', {url: 'about:blank'});
  ({sessionId: session} = await send('Target.attachToTarget', {targetId, flatten: true}));
  await send('Runtime.enable', {}, session);
  await send('Page.enable', {}, session);
  await send('Network.enable', {}, session);
  await send('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false}, session);
  const url = new URL('../supplements/mpc_euclidean_explainer.html', import.meta.url).href;
  await send('Page.navigate', {url}, session);
  for (let attempt = 0; attempt < 100; attempt++) {
    if (await evaluate('Boolean(window.MPCExplainerMath && document.getElementById("search-points")?.children.length === 18)')) break;
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  assert.equal(await evaluate('document.getElementById("search-points").children.length'), 18);
  assert.equal(await evaluate('document.querySelectorAll("#kernel-svg rect").length'), 584);
  assert.ok(await evaluate('document.documentElement.scrollWidth <= innerWidth + 1'));
  await screenshot('desktop-overview.png');

  assert.ok(await evaluate(`(() => {
    const byId = id => document.getElementById(id);
    byId('secret-slider').value = 81; byId('secret-slider').dispatchEvent(new Event('input'));
    byId('collusion-button').click();
    return byId('sharing-explanation').textContent.includes('= 81') && byId('secret-output').textContent === '81';
  })()`));
  await evaluate(`document.getElementById('reshare-button').click(); document.querySelector('[data-multiply-stage="2"]').click()`);
  assert.ok((await evaluate('document.getElementById("multiplication-note").textContent')).includes('encode 63'));

  await evaluate('document.getElementById("matrix-view-button").click()');
  assert.ok((await evaluate('document.getElementById("matrix-readout").textContent')).includes('Secret-shared'));
  assert.ok((await evaluate('document.getElementById("conditional-caption").textContent')).includes('placeholders'));
  await evaluate(`document.getElementById('matrix-column').value = 6; document.getElementById('matrix-column').dispatchEvent(new Event('change'))`);
  assert.ok(!(await evaluate('document.getElementById("matrix-readout").textContent')).includes('Secret-shared'));
  await evaluate('document.getElementById("matrix-view-button").click()');
  for (const gamma of [.2, 3]) {
    await evaluate(`document.getElementById('gamma-slider').value = ${gamma}; document.getElementById('gamma-slider').dispatchEvent(new Event('input'))`);
    assert.ok(Number.isFinite(Number(await evaluate('document.getElementById("joint-private-logdet").textContent'))));
  }
  await evaluate(`document.getElementById('gamma-slider').value = 1; document.getElementById('gamma-slider').dispatchEvent(new Event('input')); document.getElementById('kernel').scrollIntoView({behavior:'instant',block:'start'})`);
  await screenshot('desktop-kernel.png');

  await evaluate(`document.getElementById('distance-slider').value = .02; document.getElementById('distance-slider').dispatchEvent(new Event('input'))`);
  const thin = Number(await evaluate('document.getElementById("determinant-value").textContent'));
  await evaluate(`document.getElementById('distance-slider').value = 2; document.getElementById('distance-slider').dispatchEvent(new Event('input'))`);
  assert.ok(Number(await evaluate('document.getElementById("determinant-value").textContent')) > thin);
  await evaluate('for (let i = 0; i < 3; i++) document.getElementById("pivot-step-button").click()');
  assert.ok((await evaluate('document.getElementById("pivot-stage").textContent')).includes('0.536'));
  await evaluate('document.getElementById("search-next").click()');
  assert.equal(await evaluate('document.getElementById("search-output").textContent'), '1 / 200');
  await evaluate('document.getElementById("search-back").click()');
  assert.equal(await evaluate('document.getElementById("search-output").textContent'), '0 / 200');
  await evaluate(`document.getElementById('search-slider').value = 200; document.getElementById('search-slider').dispatchEvent(new Event('input'))`);
  assert.ok((await evaluate('document.getElementById("search-decision").textContent')).includes('frozen'));
  assert.equal(await evaluate('document.querySelectorAll(".point-chip.selected").length'), 6);
  assert.equal(await evaluate('document.querySelectorAll("#onehot-grid .on").length'), 6);
  assert.equal(await evaluate('document.getElementById("search-stalled").textContent'), '50 / 50');
  await evaluate('document.getElementById("search-reset").click(); document.getElementById("search-play").click()');
  await new Promise(resolve => setTimeout(resolve, 800));
  await evaluate('document.getElementById("search-play").click()');
  assert.ok((await evaluate('Number(document.getElementById("search-slider").value)')) > 0);
  await evaluate('document.getElementById("search").scrollIntoView({behavior:"instant",block:"start"})');
  await screenshot('desktop-search.png');

  await send('Emulation.setDeviceMetricsOverride', {width: 390, height: 844, deviceScaleFactor: 1, mobile: true}, session);
  await evaluate('window.scrollTo({top:0,behavior:"instant"})');
  await screenshot('mobile-overview.png');
  assert.ok(await evaluate('document.documentElement.scrollWidth <= innerWidth + 1'), 'Mobile document must not overflow horizontally');
  await evaluate('document.getElementById("search").scrollIntoView({behavior:"instant",block:"start"})');
  await screenshot('mobile-search.png');
  assert.equal(exceptions.length, 0, JSON.stringify(exceptions));
  assert.equal(network.filter(url => /^https?:/.test(url)).length, 0, 'Page must not load network resources');
  console.log(JSON.stringify({browser_checks_passed: true, desktop: '1440 × 1000', mobile: '390 × 844',
    uncaught_errors: exceptions.length, network_requests: network.filter(url => /^https?:/.test(url)).length,
    screenshot_directory: directory}, null, 2));
} finally {
  if (child.exitCode === null) await send('Browser.close').catch(() => {});
  if (child.exitCode === null) child.kill();
  for (const request of pending.values()) clearTimeout(request.timeout);
}
