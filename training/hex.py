"""Hex engine for n x n boards, matching rhombify's board.js conventions.

  - idx = row * n + col
  - RED (player 0) connects top <-> bottom (rows), moves first.
  - BLUE (player 1) connects left <-> right (cols).
  - Hex 6-neighborhood: (-1,0)(-1,+1)(0,-1)(0,+1)(+1,-1)(+1,0)  [from board.js]

Hex can never draw: exactly one player connects. We exploit that.

Canonical view (for the net): the position is always presented from the
perspective of the player-to-move as if THEY are the vertical (red) player.
Because transposing the board (r,c)->(c,r) preserves hex adjacency and swaps
the two connection axes, a blue-to-move position is shown transposed with
colours swapped. One 2-plane net then handles both colours and both axes,
which is also why it generalises across board sizes.
"""

import numpy as np

NEIGHBORS = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))

RED, BLUE = 0, 1


class Hex:
    def __init__(self, n=7):
        self.n = n
        self.board = np.zeros(n * n, dtype=np.int8)  # 0 empty, 1 red, 2 blue
        self.to_move = RED  # red moves first
        self._winner = None

    def clone(self):
        g = Hex.__new__(Hex)
        g.n = self.n
        g.board = self.board.copy()
        g.to_move = self.to_move
        g._winner = self._winner
        return g

    def legal_moves(self):
        return np.flatnonzero(self.board == 0)

    def play(self, idx):
        assert self.board[idx] == 0 and self._winner is None
        color = self.to_move + 1  # 1 red, 2 blue
        self.board[idx] = color
        if self._connects(idx, self.to_move):
            self._winner = self.to_move
        self.to_move ^= 1

    @property
    def winner(self):
        return self._winner  # None, RED, or BLUE

    # --- win detection: flood-fill the just-moved colour between its edges ---
    def _connects(self, idx, player):
        n = self.n
        color = player + 1
        board = self.board
        r0, c0 = divmod(idx, n)

        # Which edges this colour must join.
        if player == RED:      # rows 0 .. n-1
            touch_lo = r0 == 0
            touch_hi = r0 == n - 1
        else:                  # cols 0 .. n-1
            touch_lo = c0 == 0
            touch_hi = c0 == n - 1

        stack = [idx]
        seen = {idx}
        while stack:
            cur = stack.pop()
            r, c = divmod(cur, n)
            for dr, dc in NEIGHBORS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    nb = rr * n + cc
                    if nb not in seen and board[nb] == color:
                        if player == RED:
                            touch_lo |= rr == 0
                            touch_hi |= rr == n - 1
                        else:
                            touch_lo |= cc == 0
                            touch_hi |= cc == n - 1
                        seen.add(nb)
                        stack.append(nb)
        return touch_lo and touch_hi

    # --- canonical encoding for the network ---
    def encode(self):
        """Return (2, n, n) float planes from the to-move player's view, and a
        transform tag so we can map policy indices back to real cells."""
        n = self.n
        b = self.board.reshape(n, n)
        if self.to_move == RED:
            mine = (b == 1).astype(np.float32)
            theirs = (b == 2).astype(np.float32)
            transposed = False
        else:
            # show blue-to-move as if vertical: transpose + swap colours
            mine = (b == 2).T.astype(np.float32)
            theirs = (b == 1).T.astype(np.float32)
            transposed = True
        return np.stack([mine, theirs]), transposed

    def canon_to_real(self, action, transposed):
        if not transposed:
            return action
        n = self.n
        r, c = divmod(action, n)
        return c * n + r  # inverse transpose


def play_random_until_end(g, rng):
    while g.winner is None:
        moves = g.legal_moves()
        g.play(int(rng.choice(moves)))
    return g.winner
