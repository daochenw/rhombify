// Neural strategy: a trained AlphaZero-style net (fully-convolutional, so the
// same weights work for any board size) running entirely in the browser as
// inference only. Weights are produced offline by training/train.py.
//
// Play uses the policy head directly (one forward pass per move). The value
// head trained well enough to learn the game but, in the ~1h training budget,
// stayed too noisy for MCTS to improve on raw policy (search actually scored
// worse vs a random opponent than policy alone) -- so we play the policy, which
// also makes each move instant. To re-enable search later, train the value head
// longer and add MCTS back on top of forward().
//
// Conventions mirror board.js and training/hex.py:
//   idx = row * n + col;  red connects top<->bottom and moves first;
//   neighbours (-1,0)(-1,1)(0,-1)(0,1)(1,-1)(1,0).

// ---- weight loading -------------------------------------------------------

function b64ToFloat32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

let net = null; // { n, channels, blocks, layers: {name -> {shape,w,b}} }

export async function loadNet(url = 'hexnet.json') {
  const blob = await (await fetch(url)).json();
  const layers = {};
  for (const l of blob.layers) {
    layers[l.name] = { shape: l.shape, w: b64ToFloat32(l.w), b: b64ToFloat32(l.b) };
  }
  net = { n: blob.n, channels: blob.channels, blocks: blob.blocks, layers };
  return net;
}

export function netReady() {
  return net !== null;
}

// ---- forward pass ---------------------------------------------------------
// Activations are Float32Array laid out as [channel][row][col] -> c*A + r*n + col,
// where A = n*n.

// Pre-pad input to (cin, n+2, n+2) so the inner kernel loop needs no bounds
// checks (the dominant cost of the forward pass).
function conv3x3(inp, cin, w, b, cout, n, relu, residual) {
  const A = n * n, p = n + 2, PA = p * p;
  const pad = new Float32Array(cin * PA);
  for (let ic = 0; ic < cin; ic++) {
    const src = ic * A, dst = ic * PA;
    for (let r = 0; r < n; r++) {
      pad.set(inp.subarray(src + r * n, src + r * n + n), dst + (r + 1) * p + 1);
    }
  }
  const out = new Float32Array(cout * A);
  for (let oc = 0; oc < cout; oc++) {
    const wBase = oc * cin * 9, oBase = oc * A, bias = b[oc];
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        let acc = bias;
        const top = r * p + c; // top-left of 3x3 window in padded coords
        for (let ic = 0; ic < cin; ic++) {
          const pBase = ic * PA, wIc = wBase + ic * 9;
          const r0 = pBase + top, r1 = r0 + p, r2 = r1 + p;
          acc += pad[r0] * w[wIc] + pad[r0 + 1] * w[wIc + 1] + pad[r0 + 2] * w[wIc + 2]
               + pad[r1] * w[wIc + 3] + pad[r1 + 1] * w[wIc + 4] + pad[r1 + 2] * w[wIc + 5]
               + pad[r2] * w[wIc + 6] + pad[r2 + 1] * w[wIc + 7] + pad[r2 + 2] * w[wIc + 8];
        }
        const oi = oBase + r * n + c;
        if (residual) acc += residual[oi];
        out[oi] = relu && acc < 0 ? 0 : acc;
      }
    }
  }
  return out;
}

// 1x1 conv: cout x cin pointwise.
function conv1x1(inp, cin, w, b, cout, n) {
  const A = n * n;
  const out = new Float32Array(cout * A);
  for (let oc = 0; oc < cout; oc++) {
    const wBase = oc * cin;
    const oBase = oc * A;
    for (let p = 0; p < A; p++) {
      let acc = b[oc];
      for (let ic = 0; ic < cin; ic++) acc += inp[ic * A + p] * w[wBase + ic];
      out[oBase + p] = acc;
    }
  }
  return out;
}

// Returns { policy: Float32Array(n*n) logits, value: number in (-1,1) }.
function forward(planes) {
  const n = net.n, C = net.channels, L = net.layers, A = n * n;
  let x = conv3x3(planes, 2, L.stem.w, L.stem.b, C, n, true, null);
  for (let i = 0; i < net.blocks; i++) {
    const a = L['res' + i + '_a'], bk = L['res' + i + '_b'];
    const y = conv3x3(x, C, a.w, a.b, C, n, true, null);
    x = conv3x3(y, C, bk.w, bk.b, C, n, false, x); // skip + relu(x+y)
    for (let j = 0; j < x.length; j++) if (x[j] < 0) x[j] = 0;
  }
  // policy: 1x1 conv -> 1 plane
  const policy = conv1x1(x, C, L.policy.w, L.policy.b, 1, n);
  // value: 1x1 conv (16) -> global avg pool -> fc1 relu -> fc2 tanh
  const vc = conv1x1(x, C, L.value_conv.w, L.value_conv.b, 16, n);
  const pooled = new Float32Array(16);
  for (let ch = 0; ch < 16; ch++) {
    let s = 0;
    for (let p = 0; p < A; p++) s += vc[ch * A + p];
    pooled[ch] = s / A;
  }
  const f1 = L.value_fc1, f2 = L.value_fc2;
  const h = new Float32Array(32);
  for (let o = 0; o < 32; o++) {
    let acc = f1.b[o];
    for (let i = 0; i < 16; i++) acc += pooled[i] * f1.w[o * 16 + i];
    h[o] = acc < 0 ? 0 : acc;
  }
  let v = f2.b[0];
  for (let i = 0; i < 32; i++) v += h[i] * f2.w[i];
  return { policy, value: Math.tanh(v) };
}

// ---- canonical encoding (mirrors hex.py) ----------------------------------
// The net always sees the position from the to-move player's view as if they
// were the vertical (red) player; for blue we transpose and swap colours.

function encodeBoard(board, toMove) {
  const n = net.n, A = n * n, planes = new Float32Array(2 * A);
  if (toMove === 0) {
    for (let i = 0; i < A; i++) {
      if (board[i] === 1) planes[i] = 1;
      else if (board[i] === 2) planes[A + i] = 1;
    }
    return { planes, transposed: false };
  }
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const i = r * n + c, ti = c * n + r; // transpose
    if (board[i] === 2) planes[ti] = 1;
    else if (board[i] === 1) planes[A + ti] = 1;
  }
  return { planes, transposed: true };
}

// ---- public API -----------------------------------------------------------

// Sample the first few plies from a softened policy so games diverge (the
// trained policy is sharp, so greedy play would repeat the same game); after
// that, play the strongest move.
const OPENING_PLIES = 3;
const OPENING_TEMP = 1.4;

export function chooseMove(stones, color) {
  if (!net) return null;
  const n = net.n, A = n * n;
  const board = new Int8Array(A);
  const legal = [];
  let filled = 0;
  for (let i = 0; i < A; i++) {
    if (stones[i] === 'red') { board[i] = 1; filled++; }
    else if (stones[i] === 'blue') { board[i] = 2; filled++; }
    else legal.push(i);
  }
  if (!legal.length) return null;

  const { planes, transposed } = encodeBoard(board, color === 'red' ? 0 : 1);
  const { policy } = forward(planes);
  // real cell -> canonical policy index
  const logitOf = (a) => transposed ? policy[(a % n) * n + ((a / n) | 0)] : policy[a];

  if (filled < OPENING_PLIES) {
    let mx = -Infinity;
    for (const a of legal) mx = Math.max(mx, logitOf(a));
    let tot = 0;
    const w = legal.map((a) => { const e = Math.exp((logitOf(a) - mx) / OPENING_TEMP); tot += e; return e; });
    let r = Math.random() * tot;
    for (let i = 0; i < legal.length; i++) { r -= w[i]; if (r < 0) return legal[i]; }
    return legal[legal.length - 1];
  }

  let best = legal[0], bestLogit = logitOf(legal[0]);
  for (const a of legal) { const l = logitOf(a); if (l > bestLogit) { bestLogit = l; best = a; } }
  return best;
}

// No learning at play time; kept so the game loop's call is a harmless no-op.
export function reward() {}

// Test hook (parity check; not used by the app).
export const _test = { forward, encodeBoard };
