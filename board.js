// Board rendering and game state. Exports `stones`, `nextColor`, and `place(idx)`
// for the rest of the app.

const SIZE = 7;
const board = document.getElementById('board');
const cell = 36;
const margin = 36;
const stars = new Set(['2,2', '2,4', '4,2', '4,4', '3,3']); // 3-3 points and center

export const stones = new Array(SIZE * SIZE).fill(null);
export let nextColor = 'black';

const at = index => margin + index * cell + 'px';

export function place(idx) {
  if (stones[idx]) return; // occupied

  const col = idx % SIZE;
  const row = Math.floor(idx / SIZE);

  stones[idx] = nextColor;

  const stone = document.createElement('div');
  stone.className = 'stone ' + nextColor;
  stone.style.left = at(col);
  stone.style.top = at(row);
  board.appendChild(stone);

  nextColor = nextColor === 'black' ? 'white' : 'black';
}

// Grid lines
for (let i = 0; i < SIZE; i++) {
  const h = document.createElement('div');
  h.className = 'line h';
  h.style.top = at(i);
  board.appendChild(h);

  const v = document.createElement('div');
  v.className = 'line v';
  v.style.left = at(i);
  board.appendChild(v);
}

// Star points
stars.forEach(key => {
  const [col, row] = key.split(',').map(Number);
  const s = document.createElement('div');
  s.className = 'star';
  s.style.left = at(col);
  s.style.top = at(row);
  board.appendChild(s);
});
