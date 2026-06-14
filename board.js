// Hex board rendering and game state. Exports `stones`, `nextColor`, and
// `place(idx)` for the rest of the app. The board is a right-leaning rhombus of
// SIZE×SIZE pointy-top hexagons; idx = row * SIZE + col.
// Red connects top<->bottom, blue connects left<->right.

const SIZE = 7;
const board = document.getElementById('board');

const s = 26;            // hex size: center to vertex
const w = Math.sqrt(3) * s; // hex width (flat-to-flat)
const pad = 6;           // breathing room around the rhombus

export const stones = new Array(SIZE * SIZE).fill(null);
export let nextColor = 'red';

// Cell-center pixel coordinates, indexed by idx.
const centers = new Array(SIZE * SIZE);
for (let row = 0; row < SIZE; row++) {
  for (let col = 0; col < SIZE; col++) {
    centers[row * SIZE + col] = {
      x: pad + w / 2 + w * (col + row / 2),
      y: pad + s + 1.5 * s * row,
    };
  }
}

// Bounding box of the whole rhombus.
const boardW = pad * 2 + w * (SIZE + (SIZE - 1) / 2);
const boardH = pad * 2 + 1.5 * s * (SIZE - 1) + 2 * s;
board.style.width = boardW + 'px';
board.style.height = boardH + 'px';

const SVG_NS = 'http://www.w3.org/2000/svg';
const svg = document.createElementNS(SVG_NS, 'svg');
svg.setAttribute('width', boardW);
svg.setAttribute('height', boardH);
board.appendChild(svg);

// Six vertices of a pointy-top hexagon centered at (cx, cy):
// 0 top, 1 upper-right, 2 lower-right, 3 bottom, 4 lower-left, 5 upper-left.
function verts(cx, cy) {
  const v = [];
  for (let i = 0; i < 6; i++) {
    const a = Math.PI / 180 * (60 * i - 90);
    v.push({ x: cx + s * Math.cos(a), y: cy + s * Math.sin(a) });
  }
  return v;
}

// Pass 1: cell outlines.
centers.forEach(c => {
  const v = verts(c.x, c.y);
  const cell = document.createElementNS(SVG_NS, 'polygon');
  cell.setAttribute('points', v.map(p => p.x.toFixed(2) + ',' + p.y.toFixed(2)).join(' '));
  cell.setAttribute('class', 'cell');
  svg.appendChild(cell);
});

// Pass 2: colored player edges, tracing the outer hexagon edges of the
// boundary cells (drawn over the outlines; blue first so red wins the corners).
function colorEdge(v, i, j, cls) {
  const line = document.createElementNS(SVG_NS, 'line');
  line.setAttribute('x1', v[i].x);
  line.setAttribute('y1', v[i].y);
  line.setAttribute('x2', v[j].x);
  line.setAttribute('y2', v[j].y);
  line.setAttribute('class', 'edge ' + cls);
  svg.appendChild(line);
}

const last = SIZE - 1;
for (let row = 0; row < SIZE; row++) {
  for (let col = 0; col < SIZE; col++) {
    const v = verts(centers[row * SIZE + col].x, centers[row * SIZE + col].y);
    if (col === 0)    { colorEdge(v, 4, 5, 'blue'); colorEdge(v, 3, 4, 'blue'); } // left
    if (col === last) { colorEdge(v, 1, 2, 'blue'); colorEdge(v, 0, 1, 'blue'); } // right
    if (row === 0)    { colorEdge(v, 5, 0, 'red');  colorEdge(v, 0, 1, 'red');  } // top
    if (row === last) { colorEdge(v, 2, 3, 'red');  colorEdge(v, 3, 4, 'red');  } // bottom
  }
}

// Keep the fixed -30deg orientation but scale so the whole board fits the
// viewport (capped at 1 so it never grows past natural size).
function fit() {
  const m = 16;
  const rad = 30 * Math.PI / 180;
  const bw = boardW * Math.cos(rad) + boardH * Math.sin(rad);
  const bh = boardW * Math.sin(rad) + boardH * Math.cos(rad);
  const scale = Math.min(1, (window.innerWidth - m) / bw, (window.innerHeight - m) / bh);
  board.style.transform = 'translate(-50%, -50%) rotate(-30deg) scale(' + scale + ')';
}
fit();
window.addEventListener('resize', fit);

export function place(idx) {
  if (stones[idx]) return; // occupied

  stones[idx] = nextColor;

  const c = centers[idx];
  const stone = document.createElement('div');
  stone.className = 'stone ' + nextColor;
  stone.style.left = c.x + 'px';
  stone.style.top = c.y + 'px';
  board.appendChild(stone);

  nextColor = nextColor === 'red' ? 'blue' : 'red';
}
