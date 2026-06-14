// Move-selection strategies. A strategy takes a read-only view of the board
// (`stones`: array of null | 'red' | 'blue') and the color about to move,
// and returns the chosen cell index, or null if no move is available.
//
// This is the seam for more sophisticated methods later — add new functions
// here and swap which one the game loop calls.

const weights = { red: [], blue: [] };
const played = { red: [], blue: [] };

function learnable(color, n) {
  while (weights[color].length < n) weights[color].push(1);
}

function symmetric(i, n) {
  return i % n * n + Math.floor(i / n);
}

// Uniformly at random among the still-empty cells.
// `color` is unused for now but kept in the signature for future strategies.
export function random(stones, color) {
  const empty = [];
  for (let i = 0; i < stones.length; i++) {
    if (!stones[i]) empty.push(i);
  }
  if (!empty.length) return null;
  return empty[Math.floor(Math.random() * empty.length)];
}

// Weighted random by color/cell, nudged upward after winning games.
export function learn0(stones, color) {
  learnable(color, stones.length);
  let total = 0;
  const empty = [];
  for (let i = 0; i < stones.length; i++) {
    if (!stones[i]) {
      empty.push(i);
      total += weights[color][i];
    }
  }
  if (!empty.length) return null;

  let r = Math.random() * total;
  for (const i of empty) {
    r -= weights[color][i];
    if (r < 0) {
      played[color].push(i);
      return i;
    }
  }
  played[color].push(empty[empty.length - 1]);
  return empty[empty.length - 1];
}

export function reward(winner) {
  const loser = winner === 'red' ? 'blue' : 'red';
  const size = weights[winner].length;
  const n = Math.sqrt(size);
  learnable(loser, size);
  for (const i of played[winner]) {
    weights[winner][i]++;
    weights[loser][symmetric(i, n)]++;
  }
  played.red = [];
  played.blue = [];
}

export const chooseMove = learn0;
