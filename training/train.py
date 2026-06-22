"""Time-boxed AlphaZero-style training for 7x7 Hex.

Each iteration: generate a batch of self-play games (batched MCTS), append to a
replay buffer, then take several gradient steps. Periodically exports browser
weights and a checkpoint. Stops after TIME_BUDGET seconds.
"""

import argparse
import collections
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from model import HexNet
from mcts import self_play_batch
from export_weights import export

N = 7
CHANNELS = 48
BLOCKS = 5
NUM_SIMS = 60
GAMES_PER_ITER = 256
BUFFER = 60000
BATCH = 512
STEPS_PER_ITER = 80
LR = 1e-3
WD = 1e-4

def train(time_budget, out_json, ckpt, device, seed=0, resume=False):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = HexNet(CHANNELS, BLOCKS).to(device)
    if resume and os.path.exists(ckpt):
        net.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"resumed weights from {ckpt}")  # optimizer + buffer start fresh
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    buf_p = collections.deque(maxlen=BUFFER)
    buf_pi = collections.deque(maxlen=BUFFER)
    buf_z = collections.deque(maxlen=BUFFER)

    t0 = time.time()
    it = 0
    games_done = 0
    while time.time() - t0 < time_budget:
        it += 1
        net.eval()
        P, Pi, Z = self_play_batch(net, device, N, GAMES_PER_ITER, NUM_SIMS, rng)
        buf_p.extend(P); buf_pi.extend(Pi); buf_z.extend(Z)
        games_done += GAMES_PER_ITER

        net.train()
        if len(buf_p) >= BATCH:
            P_arr = np.stack(buf_p); Pi_arr = np.stack(buf_pi)
            Z_arr = np.array(buf_z, dtype=np.float32)
            ploss = vloss = 0.0
            for _ in range(STEPS_PER_ITER):
                idx = rng.integers(0, len(buf_p), size=BATCH)
                x = torch.from_numpy(P_arr[idx]).to(device)
                tpi = torch.from_numpy(Pi_arr[idx]).to(device)
                tz = torch.from_numpy(Z_arr[idx]).to(device)
                logits, v = net(x)
                lp = -(tpi * F.log_softmax(logits, dim=1)).sum(1).mean()
                lv = F.mse_loss(v, tz)
                loss = lp + lv
                opt.zero_grad(); loss.backward(); opt.step()
                ploss += lp.item(); vloss += lv.item()
            ploss /= STEPS_PER_ITER; vloss /= STEPS_PER_ITER
        else:
            ploss = vloss = float("nan")

        el = time.time() - t0
        print(f"it {it:3d} | {el:5.0f}s | games {games_done:6d} | "
              f"buf {len(buf_p):6d} | ploss {ploss:.3f} vloss {vloss:.3f}",
              flush=True)

        net.eval()
        export(net, N, out_json)
        torch.save(net.state_dict(), ckpt)

    print(f"done: {games_done} games in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="../hexnet.json")
    ap.add_argument("--ckpt", default="hexnet.pt")
    ap.add_argument("--resume", action="store_true",
                    help="warm-start the net from --ckpt instead of from scratch")
    args = ap.parse_args()
    train(args.minutes * 60, args.out, args.ckpt, args.device, resume=args.resume)
