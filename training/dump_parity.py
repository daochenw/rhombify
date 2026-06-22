"""Dump (board, toMove, canonical policy logits, value) for a few positions
from the PyTorch net, so the JS forward pass can be checked against it."""
import json
import numpy as np
import torch
from model import HexNet
from hex import Hex

net = HexNet(48, 5)
net.load_state_dict(torch.load("hexnet.pt", map_location="cpu"))
net.eval()

rng = np.random.default_rng(7)
cases = []
for _ in range(5):
    g = Hex(7)
    k = rng.integers(0, 12)
    for _ in range(int(k)):
        if g.winner is not None:
            break
        m = g.legal_moves()
        g.play(int(rng.choice(m)))
    if g.winner is not None:
        continue
    planes, _ = g.encode()
    with torch.no_grad():
        logits, v = net(torch.from_numpy(planes[None]))
    cases.append({
        "board": g.board.tolist(),
        "to_move": int(g.to_move),
        "policy": logits[0].numpy().tolist(),
        "value": float(v[0]),
    })

json.dump(cases, open("parity_cases.json", "w"))
print(f"wrote {len(cases)} parity cases")
