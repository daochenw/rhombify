// Move-selection strategies. A strategy takes a read-only view of the board
// (`stones`: array of null | 'red' | 'blue') and the color about to move,
// and returns the chosen cell index, or null if no move is available.
//
// This is the seam for more sophisticated methods later — add new functions
// here and swap which one the game loop calls.

// Uniformly at random among the still-empty cells.
// `color` is unused for now but kept in the signature for future strategies.
export function chooseMove(stones, color) {
  const empty = [];
  for (let i = 0; i < stones.length; i++) {
    if (!stones[i]) empty.push(i);
  }
  if (!empty.length) return null;
  return empty[Math.floor(Math.random() * empty.length)];
}
