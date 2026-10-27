// Run with node tests/check_mpc_explainer.mjs. No browser or npm packages needed.
import assert from 'node:assert/strict';
import {readFileSync, existsSync} from 'node:fs';
import vm from 'node:vm';

const page = new URL('../supplements/mpc_euclidean_explainer.html', import.meta.url);
const html = readFileSync(page, 'utf8');
const script = html.match(/<script id="explainer-script">([\s\S]*?)<\/script>/)[1];
const context = {};
vm.runInNewContext(script, context, {timeout: 10000});
const math = context.MPCExplainerMath;
assert.ok(math);
const close = (a, b, tolerance = 1e-9) => assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b}`);

for (let secret = 0; secret < 101; secret++) {
  for (let slope = 0; slope < 101; slope++) assert.equal(math.recoverToy(math.shareToy(secret, slope)), secret);
  const fixedShare = 49;
  assert.equal(math.shareToy(secret, math.mod(fixedShare - secret))[0], fixedShare);
}
const multiplication = math.multiplyToy();
assert.equal(math.recoverToy(multiplication.result), 63);
assert.equal(math.mod(multiplication.result[1] - multiplication.result[0]), math.mod(multiplication.result[2] - multiplication.result[1]));

const points = math.makeTeachingPoints();
assert.equal(points.length, 24);
assert.ok(points.every(row => row.length === 4 && row.every(value => value >= 0 && value <= 1)));
for (const gamma of [.2, 1, 3]) {
  const kernel = math.rbf(points, gamma), conditional = math.condition(kernel);
  assert.equal(conditional.q.length, 18);
  for (let i = 0; i < 24; i++) {
    assert.equal(kernel[i][i], 1);
    for (let j = 0; j < 24; j++) close(kernel[i][j], kernel[j][i], 1e-15);
  }
  for (const ids of [[0, 1, 2, 3, 4, 5], [3, 6, 8, 11, 14, 17]]) {
    const joint = [0, 1, 2, 3, 4, 5, ...ids.map(i => i + 6)];
    close(math.logdet(math.submatrix(kernel, joint)), conditional.contextLogdet + math.logdet(math.submatrix(conditional.q, ids)));
  }
}
close(Math.exp(math.logdet([[1, .6, .2], [.6, 1, .4], [.2, .4, 1]])), .536, 1e-14);
assert.throws(() => math.logdet([[1, 1], [1, 1]]));
for (const distance of [.02, .75, 2]) {
  const rho = Math.exp(-(distance ** 2));
  close(math.logdet([[1, rho], [rho, 1]]), Math.log(1 - rho ** 2), 1e-10);
}

const simulation = math.simulateSearch();
assert.equal(simulation.trace.length, 201);
assert.ok(simulation.stopStep > 0 && simulation.stopStep < 200, 'Demo must illustrate private stopping and padding');
for (const [i, row] of simulation.trace.entries()) {
  assert.equal(row.selected.length, 6);
  assert.equal(new Set(row.selected).size, 6);
  assert.ok(row.selected.every(id => id >= 0 && id < 18));
  const joint = [0, 1, 2, 3, 4, 5, ...row.selected.map(id => id + 6)];
  close(row.current, math.logdet(math.submatrix(simulation.kernel, joint)));
  if (i === 0) continue;
  assert.ok(row.current >= simulation.trace[i - 1].current);
  if (row.active) {
    close(row.ratio, Math.exp(row.delta), 1e-14);
    assert.equal(row.accepted, row.delta >= 0);
  } else {
    assert.equal(row.stalled, 50);
    assert.equal(row.activeCount, simulation.stopStep);
    assert.equal(JSON.stringify(row.selected), JSON.stringify(simulation.trace[simulation.stopStep].selected));
    assert.equal(row.current, simulation.trace[simulation.stopStep].current);
  }
}
assert.equal(JSON.stringify(simulation.trace), JSON.stringify(math.simulateSearch().trace));

const data = JSON.parse(html.match(/<script id="experiment-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
assert.equal(data.patience, 50);
assert.equal(data.trials.length, 3);
for (const row of data.trials) {
  assert.equal(JSON.stringify(row.plain_ids), JSON.stringify(row.mpc_ids));
  assert.equal(new Set(row.mpc_ids).size, 6);
  assert.ok(Math.abs(row.plain_logdet - row.mpc_logdet) < 2e-13);
}
const record = new URL(`../results/${data.run}/results.json`, import.meta.url);
if (existsSync(record)) {
  const results = JSON.parse(readFileSync(record, 'utf8'));
  for (const row of data.trials) {
    const measured = results.trials[row.trial - 1];
    assert.equal(row.mpc_logdet, measured.final_joint_logdet);
    assert.equal(row.iterations, measured.iterations);
    assert.equal(JSON.stringify(row.mpc_ids), JSON.stringify(measured.selected_public_ids));
  }
}
console.log(`Explainer checks passed: sharing, multiplication, RBF/Schur identity, geometry, 200-step SLS, padding and measured results. Teaching stop: ${simulation.stopStep}.`);
