"""Batched PUCT MCTS + self-play for AlphaZero-style training.

Two throughput tricks:
  - many self-play games advance in lockstep; every simulation's leaf states
    (one per active game) are evaluated in a single batched forward pass.
  - lazy expansion: a child's game state is materialised (clone + play + win
    check) only when MCTS first selects it, not for every legal move up front.
    With modest sim counts most children are never visited, so this avoids the
    bulk of the clone/flood-fill cost.
"""

import math
import numpy as np
import torch

from hex import Hex

C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_FRAC = 0.25


class Node:
    __slots__ = ("to_play", "P", "N", "W", "children", "state", "terminal",
                 "action", "parent")

    def __init__(self, prior, action=None, parent=None):
        self.to_play = None       # set when state is materialised
        self.P = prior
        self.N = 0
        self.W = 0.0
        self.children = None      # dict action -> Node, None until expanded
        self.state = None         # Hex, materialised lazily
        self.terminal = None      # None, or value from this node's perspective
        self.action = action      # move that leads from parent to here
        self.parent = parent

    @property
    def Q(self):
        return self.W / self.N if self.N else 0.0


def _materialise(node):
    """Ensure node.state / to_play / terminal are filled in."""
    if node.state is not None or node.terminal is not None:
        return
    g = node.parent.state.clone()
    g.play(node.action)
    node.to_play = g.to_move
    if g.winner is not None:
        node.terminal = -1.0      # mover just won; this node's player has lost
    else:
        node.state = g


def _select(node):
    """Descend by PUCT to a not-yet-expanded / terminal leaf; return the path."""
    path = [node]
    while node.children is not None and node.terminal is None:
        sqrt_n = math.sqrt(node.N)
        best, best_score = None, -1e30
        for child in node.children.values():
            u = C_PUCT * child.P * sqrt_n / (1 + child.N)
            score = -child.Q + u      # child.Q is from child's view; negate
            if score > best_score:
                best_score, best = score, child
        node = best
        _materialise(node)
        path.append(node)
    return path


def _expand(node, policy_logits):
    g = node.state
    moves = g.legal_moves()
    logits = policy_logits[moves]
    logits = logits - logits.max()
    p = np.exp(logits)
    p /= p.sum()
    node.children = {int(a): Node(float(pa), action=int(a), parent=node)
                     for a, pa in zip(moves, p)}


def _backup(path, value):
    for node in reversed(path):
        node.N += 1
        node.W += value
        value = -value


def _add_dirichlet(root, rng):
    acts = list(root.children)
    noise = rng.dirichlet([DIRICHLET_ALPHA] * len(acts))
    for a, nz in zip(acts, noise):
        c = root.children[a]
        c.P = (1 - DIRICHLET_FRAC) * c.P + DIRICHLET_FRAC * nz


def _eval_batch(net, device, nodes):
    planes = np.stack([nd.state.encode()[0] for nd in nodes])
    with torch.no_grad():
        x = torch.from_numpy(planes).to(device)
        logits, v = net(x)
    return logits.cpu().numpy(), v.cpu().numpy()


def run_search(roots, net, device, num_sims, rng, add_noise=True):
    need = [r for r in roots if r.children is None]
    if need:
        logits, _ = _eval_batch(net, device, need)
        for r, lg in zip(need, logits):
            _expand(r, lg)
    if add_noise:
        for r in roots:
            _add_dirichlet(r, rng)

    for _ in range(num_sims):
        paths, leaves = [], []
        for r in roots:
            path = _select(r)
            leaf = path[-1]
            paths.append(path)
            if leaf.terminal is not None:
                _backup(path, leaf.terminal)
            else:
                leaves.append((path, leaf))
        if leaves:
            logits, vals = _eval_batch(net, device, [lf for _, lf in leaves])
            for (path, leaf), lg, v in zip(leaves, logits, vals):
                _expand(leaf, lg)
                _backup(path, float(v))


def _new_root(g):
    r = Node(1.0)
    r.state = g.clone()
    r.to_play = g.to_move
    return r


def self_play_batch(net, device, n, num_games, num_sims, rng,
                    temp_moves=None, temp=1.0):
    """Play num_games to completion in lockstep; return training samples
    (planes, policy target over n*n canonical cells, value target)."""
    if temp_moves is None:
        temp_moves = n

    games = [Hex(n) for _ in range(num_games)]
    roots = [_new_root(g) for g in games]
    histories = [[] for _ in range(num_games)]
    out_planes, out_pi, out_z = [], [], []
    active = list(range(num_games))
    ply = 0

    while active:
        run_search([roots[i] for i in active], net, device, num_sims, rng)

        next_active = []
        for i in active:
            g, root = games[i], roots[i]
            acts = list(root.children)
            counts = np.array([root.children[a].N for a in acts], dtype=np.float64)

            pi = np.zeros(n * n, dtype=np.float32)
            pi[acts] = counts / counts.sum()
            planes, transposed = g.encode()
            if transposed:
                pi = pi.reshape(n, n).T.reshape(-1).copy()
            histories[i].append((planes, pi, g.to_move))

            if ply < temp_moves:
                pr = counts ** (1 / temp)
                pr /= pr.sum()
                a = int(rng.choice(acts, p=pr))
            else:
                a = acts[int(counts.argmax())]

            g.play(a)
            if g.winner is None:
                roots[i] = _new_root(g)
                next_active.append(i)
            else:
                w = g.winner
                for planes, pi, player in histories[i]:
                    out_planes.append(planes)
                    out_pi.append(pi)
                    out_z.append(1.0 if player == w else -1.0)
        active = next_active
        ply += 1

    return out_planes, out_pi, out_z
