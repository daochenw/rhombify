"""Export a trained HexNet to a compact JSON the browser can load.

BatchNorm is folded into the preceding conv (conv has no bias, BN is affine),
so the JS side only implements plain conv + bias, 1x1 conv, global-average
pool, and two dense layers. Floats are packed little-endian float32 -> base64.
"""

import base64
import json
import numpy as np
import torch


def _fold_conv_bn(conv_w, bn):
    # conv_w: (out,in,kh,kw); bn: BatchNorm2d in eval mode
    gamma = bn.weight.detach().cpu().numpy()
    beta = bn.bias.detach().cpu().numpy()
    mean = bn.running_mean.cpu().numpy()
    var = bn.running_var.cpu().numpy()
    eps = bn.eps
    scale = gamma / np.sqrt(var + eps)
    w = conv_w.detach().cpu().numpy() * scale[:, None, None, None]
    b = beta - mean * scale
    return w.astype(np.float32), b.astype(np.float32)


def _b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4")).decode()


def export(net, n, path):
    net.eval()
    layers = []

    def add(name, w, b):
        layers.append({"name": name, "shape": list(w.shape),
                       "w": _b64(w), "b": _b64(b)})

    # stem: conv -> bn
    s = net.stem
    w, b = _fold_conv_bn(s[0].weight, s[1])
    add("stem", w, b)

    # residual blocks: each two (conv,bn)
    for i, blk in enumerate(net.tower):
        w1, b1 = _fold_conv_bn(blk.c1.weight, blk.b1)
        w2, b2 = _fold_conv_bn(blk.c2.weight, blk.b2)
        add(f"res{i}_a", w1, b1)
        add(f"res{i}_b", w2, b2)

    # policy head 1x1 conv (has bias, no BN)
    add("policy", net.p_conv.weight.detach().cpu().numpy(),
        net.p_conv.bias.detach().cpu().numpy())
    # value head: 1x1 conv, then two dense
    add("value_conv", net.v_conv.weight.detach().cpu().numpy(),
        net.v_conv.bias.detach().cpu().numpy())
    layers.append({"name": "value_fc1",
                   "shape": list(net.v_fc1.weight.shape),
                   "w": _b64(net.v_fc1.weight.detach().cpu().numpy()),
                   "b": _b64(net.v_fc1.bias.detach().cpu().numpy())})
    layers.append({"name": "value_fc2",
                   "shape": list(net.v_fc2.weight.shape),
                   "w": _b64(net.v_fc2.weight.detach().cpu().numpy()),
                   "b": _b64(net.v_fc2.bias.detach().cpu().numpy())})

    blob = {"n": n, "channels": net.stem[0].out_channels,
            "blocks": len(net.tower), "layers": layers}
    with open(path, "w") as f:
        json.dump(blob, f)
    sz = sum(len(l["w"]) + len(l.get("b", "")) for l in layers)
    print(f"exported {path}  (~{sz/1024:.0f} KB base64)")
