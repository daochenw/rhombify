"""Quick strength sanity check for the trained net.

Plays the trained net (with a little MCTS, greedy) against:
  - a uniform-random player
  - a plain rollout MCTS (no net) of a given budget
alternating colours each game. Prints win rates. This is a confidence check,
not the success bar (which is qualitative), so game counts are modest.
"""

import argparse
import numpy as np
import torch

from hex import Hex, RED
from model import HexNet
from mcts import Node, run_search, _new_root


def net_move(net, device, g, sims, rng):
    root = _new_root(g)
    run_search([root], net, device, sims, rng, add_noise=False)
    acts = list(root.children)
    counts = [root.children[a].N for a in acts]
    return acts[int(np.argmax(counts))]


def random_move(g, rng):
    return int(rng.choice(g.legal_moves()))


def rollout_value(g, rng):
    """One uniform-random playout; return winner."""
    s = g.clone()
    while s.winner is None:
        s.play(int(rng.choice(s.legal_moves())))
    return s.winner


def rollout_mcts_move(g, budget, rng):
    """Plain flat-ish MCTS: for each legal move, average random rollouts."""
    moves = g.legal_moves()
    me = g.to_move
    per = max(1, budget // len(moves))
    best, best_wr = None, -1.0
    for m in moves:
        s = g.clone(); s.play(int(m))
        if s.winner == me:
            return int(m)
        wins = sum(1 for _ in range(per) if rollout_value(s, rng) == me)
        wr = wins / per
        if wr > best_wr:
            best_wr, best = wr, int(m)
    return best


def play(net, device, sims, opponent, n, rng, net_is_red):
    g = Hex(n)
    while g.winner is None:
        net_turn = (g.to_move == RED) == net_is_red
        if net_turn:
            a = net_move(net, device, g, sims, rng)
        elif opponent == "random":
            a = random_move(g, rng)
        else:
            a = rollout_mcts_move(g, opponent, rng)
        g.play(a)
    net_color = RED if net_is_red else (1 - RED)
    return g.winner == net_color


def evaluate(ckpt, device, sims, games, n=7):
    net = HexNet(48, 5).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()
    rng = np.random.default_rng(123)
    for opp in ["random", 200]:
        wins = 0
        for i in range(games):
            wins += play(net, device, sims, opp, n, rng, net_is_red=(i % 2 == 0))
        label = "random" if opp == "random" else f"rollout-MCTS({opp})"
        print(f"net(+{sims} sims) vs {label}: {wins}/{games} = {wins/games:.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="hexnet.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--games", type=int, default=40)
    args = ap.parse_args()
    evaluate(args.ckpt, args.device, args.sims, args.games)
