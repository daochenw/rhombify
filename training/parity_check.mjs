// Compare neural.js forward pass against PyTorch dumps (parity_cases.json).
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const dir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(dir, '..');

// Stub fetch so loadNet can read the local weights file.
globalThis.fetch = async (url) => ({
  json: async () => JSON.parse(fs.readFileSync(path.resolve(root, url), 'utf8')),
});

// neural.js is ESM but lives in a dir with no package.json; import a .mjs copy.
const mjs = path.join(dir, 'neural_copy.mjs');
fs.copyFileSync(path.resolve(root, 'neural.js'), mjs);
const { loadNet, _test } = await import(mjs);
await loadNet(path.resolve(root, 'hexnet.json'));

const cases = JSON.parse(fs.readFileSync(path.join(dir, 'parity_cases.json'), 'utf8'));

let maxPolicyErr = 0, maxValueErr = 0;
for (const cse of cases) {
  const board = Int8Array.from(cse.board);
  const g = new _test.Hex(7, board, cse.to_move);
  const { planes } = g.encode();
  const { policy, value } = _test.forward(planes);
  for (let i = 0; i < 49; i++) {
    maxPolicyErr = Math.max(maxPolicyErr, Math.abs(policy[i] - cse.policy[i]));
  }
  maxValueErr = Math.max(maxValueErr, Math.abs(value - cse.value));
}
console.log(`cases ${cases.length}  maxPolicyErr ${maxPolicyErr.toExponential(2)}  maxValueErr ${maxValueErr.toExponential(2)}`);
console.log(maxPolicyErr < 1e-3 && maxValueErr < 1e-3 ? 'PARITY OK' : 'PARITY FAIL');
