"""Build a standalone hexnet.html: architecture diagram + learned-weight maps.

Reads the browser export (../hexnet.json), where BatchNorm is already folded
into the convs, so the weights shown are the *effective* filters used at
inference. All data is inlined into the HTML so the page opens from file://
with no server.

Usage:  python3 viz_hexnet.py   (writes ../hexnet.html)
"""

import base64
import json
import os

import numpy as np

SRC = os.path.join(os.path.dirname(__file__), "..", "hexnet.json")
OUT = os.path.join(os.path.dirname(__file__), "..", "hexnet.html")


def dec(b64):
    return np.frombuffer(base64.b64decode(b64), dtype="<f4")


def build_data():
    blob = json.load(open(SRC))
    layers = {l["name"]: l for l in blob["layers"]}
    n, C, B = blob["n"], blob["channels"], blob["blocks"]

    def arr(name):
        l = layers[name]
        return dec(l["w"]).reshape(l["shape"]), dec(l["b"])

    # stem: (C, 2, 3, 3) -> per-filter 3x3 maps for each input plane
    sw, _ = arr("stem")
    stem = [{"mine": sw[f, 0].round(4).flatten().tolist(),
             "theirs": sw[f, 1].round(4).flatten().tolist()} for f in range(C)]

    # policy head 1x1: (1, C, 1, 1) -> C weights mapping trunk features to logits
    pw, _ = arr("policy")
    policy = pw.reshape(C).round(4).tolist()

    # value head
    vw, _ = arr("value_conv")          # (16, C, 1, 1)
    value_conv = vw.reshape(16, C).round(4).tolist()
    f1w, _ = arr("value_fc1")          # (32, 16)
    f2w, _ = arr("value_fc2")          # (1, 32)
    value_fc1 = f1w.round(4).tolist()
    value_fc2 = f2w.reshape(32).round(4).tolist()

    # trunk: each 3x3 conv -> 48x48 channel-interaction map (mean |w| over kernel)
    res_mats = []
    for i in range(B):
        for tag in ("a", "b"):
            w, _ = arr(f"res{i}_{tag}")          # (C, C, 3, 3)
            res_mats.append({"name": f"res{i}_{tag}",
                             "mat": np.abs(w).mean(axis=(2, 3)).round(4).tolist()})

    # per-layer summary + histogram, for the table
    def summ(name):
        l = layers[name]
        w = dec(l["w"])
        lo, hi = float(w.min()), float(w.max())
        hist, _ = np.histogram(w, bins=24, range=(lo, hi))
        return {"name": name, "shape": l["shape"], "params": int(w.size),
                "min": round(lo, 4), "max": round(hi, 4),
                "mean": round(float(w.mean()), 4), "std": round(float(w.std()), 4),
                "hist": hist.tolist()}

    layer_summ = [summ(l["name"]) for l in blob["layers"]]
    total = sum(s["params"] for s in layer_summ)

    return {"n": n, "channels": C, "blocks": B, "total": total,
            "stem": stem, "policy": policy, "valueConv": value_conv,
            "valueFc1": value_fc1, "valueFc2": value_fc2,
            "resMats": res_mats, "layers": layer_summ}


def main():
    data = build_data()
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    with open(OUT, "w") as f:
        f.write(html)
    kb = os.path.getsize(OUT) / 1024
    print(f"wrote {OUT}  ({kb:.0f} KB)  total params {data['total']}")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>hexnet</title>
<style>
  :root { --ink:#1f1f1f; --soft:#6a6a6a; --line:#e6e6e3; --bg:#ffffff; }
  * { box-sizing: border-box; }
  body {
    font: 17px/1.6 'Iowan Old Style','Charter','Georgia',serif;
    color: var(--ink); background: var(--bg);
    max-width: 940px; margin: 0 auto; padding: 64px 28px 120px;
    -webkit-font-smoothing: antialiased;
  }
  h1 { font-size: 1.6rem; font-weight: 600; margin: 0 0 .2rem; letter-spacing:-.01em; }
  h2 { font-size: 1.15rem; font-weight: 600; margin: 3.2rem 0 .4rem; }
  .sub { color: var(--soft); margin: 0 0 .1rem; }
  .meta { color: var(--soft); font: 13px/1.5 ui-monospace,'SF Mono',Menlo,monospace; }
  p.note { color: var(--soft); font-size: .92rem; max-width: 640px; }
  .mono { font-family: ui-monospace,'SF Mono',Menlo,monospace; }
  hr { border:0; border-top:1px solid var(--line); margin: 2.6rem 0 0; }

  /* architecture flow */
  .flow { display:flex; flex-direction:column; align-items:center; gap:0; margin:1.4rem 0 0; }
  .box {
    border:1px solid var(--line); border-radius:8px; padding:11px 16px;
    text-align:center; background:#fcfcfb; min-width:220px;
  }
  .box .bn { font-weight:600; font-size:.96rem; }
  .box .bs { color:var(--soft); font:12px/1.4 ui-monospace,Menlo,monospace; }
  .conn { width:1px; height:20px; background:#cfcfca; }
  .conn.tall { height:26px; }
  .trunk {
    border:1px dashed #d3d3ce; border-radius:10px; padding:10px; margin:0;
    display:flex; flex-direction:column; align-items:center; gap:0;
    position:relative; background:#fbfbfa;
  }
  .trunk .tlabel { position:absolute; top:8px; right:12px; color:var(--soft);
    font:11px ui-monospace,Menlo,monospace; }
  .heads { display:flex; gap:48px; align-items:flex-start; margin-top:0; }
  .head { display:flex; flex-direction:column; align-items:center; gap:0; }
  .split { display:flex; justify-content:center; gap:48px; width:100%; }
  .splitline { height:22px; }
  .hcap { font-weight:600; margin:.1rem 0 .3rem; font-size:.95rem; }

  /* weight grids */
  .filters { display:flex; flex-wrap:wrap; gap:14px; margin-top:1rem; }
  .filt { text-align:center; }
  .filt .fid { color:var(--soft); font:10px ui-monospace,Menlo,monospace; margin-top:3px; }
  .filt .planes { display:flex; gap:4px; }
  canvas { image-rendering: pixelated; display:block; border:1px solid var(--line); }
  .matgrid { display:flex; flex-wrap:wrap; gap:18px; margin-top:1rem; }
  .matcard { text-align:center; }
  .matcard .mc { color:var(--soft); font:11px ui-monospace,Menlo,monospace; margin-top:4px; }

  /* legend */
  .legend { display:flex; align-items:center; gap:10px; color:var(--soft);
    font:12px ui-monospace,Menlo,monospace; margin:.6rem 0; }
  .bar { height:10px; width:160px; border:1px solid var(--line); }

  /* layer table */
  table { border-collapse:collapse; width:100%; margin-top:1rem; font-size:.9rem; }
  th,td { text-align:left; padding:5px 10px; border-bottom:1px solid var(--line); }
  th { font-weight:600; color:var(--soft); font-size:.8rem; text-transform:uppercase; letter-spacing:.04em; }
  td.num, th.num { text-align:right; font-family:ui-monospace,Menlo,monospace; }
  td.lname { font-family:ui-monospace,Menlo,monospace; }
  .spark { vertical-align:middle; }
</style>
</head>
<body>
  <h1>hexnet</h1>
  <p class="sub">A fully-convolutional AlphaZero-style network for 7&times;7 Hex.</p>
  <p class="meta" id="meta"></p>

  <h2>Architecture</h2>
  <p class="note">Two input planes (my stones / their stones, in the to-move
  player's canonical view) flow through a shared residual trunk, then split into
  a policy head (49 move logits) and a value head (a single &minus;1&hellip;1
  score). Most parameters live in the shared trunk.</p>
  <div class="flow" id="flow"></div>

  <h2>Stem filters</h2>
  <p class="note">The 48 first-layer 3&times;3 filters &mdash; the net's basic
  local detectors. Each shows two planes: response to <em>my</em> stones and to
  <em>their</em> stones in a cell's neighbourhood. Diverging scale:
  <span class="mono">blue&nbsp;&lt;&nbsp;0&nbsp;&lt;&nbsp;red</span>.</p>
  <div class="legend"><span>&minus;</span><div class="bar" id="divbar"></div><span>+</span></div>
  <div class="filters" id="stem"></div>

  <h2>Trunk &mdash; channel interaction</h2>
  <p class="note">Each of the 10 trunk convolutions is a 48&times;48&times;3&times;3
  tensor. Shown is mean&nbsp;|weight| over the 3&times;3 kernel: row = output
  channel, column = input channel. Darker = that output channel draws more on
  that input channel. Sequential scale (white&nbsp;&rarr;&nbsp;ink).</p>
  <div class="matgrid" id="trunk"></div>

  <h2>Policy &amp; value heads</h2>
  <p class="note">How the 48 trunk features are read out. The policy head is a
  single 1&times;1 conv (48 weights, applied at every cell); the value head pools
  16 channels globally, then a 16&rarr;32&rarr;1 MLP. Diverging scale.</p>
  <div class="matgrid" id="heads"></div>

  <h2>All layers</h2>
  <p class="note">Every tensor in the network, with its weight distribution.</p>
  <table id="ltable"><thead><tr>
    <th>layer</th><th class="num">shape</th><th class="num">params</th>
    <th class="num">mean</th><th class="num">std</th><th>distribution</th>
  </tr></thead><tbody></tbody></table>

<script>
const D = __DATA__;

// ---- color scales ----
function lerp(a,b,t){ return Math.round(a+(b-a)*t); }
function diverging(v, m){           // blue (-) .. white (0) .. red (+)
  const t = Math.max(-1, Math.min(1, v/(m||1)));
  if (t < 0){ const u=-t; return `rgb(${lerp(255,44,u)},${lerp(255,95,u)},${lerp(255,138,u)})`; }
  const u=t; return `rgb(${lerp(255,192,u)},${lerp(255,73,u)},${lerp(255,43,u)})`;
}
function sequential(v, m){          // white .. ink
  const t = Math.max(0, Math.min(1, v/(m||1)));
  return `rgb(${lerp(252,31,t)},${lerp(252,31,t)},${lerp(250,31,t)})`;
}

// ---- small canvas heatmap ----
function heat(mat, cell, scale){
  const rows=mat.length, cols=mat[0].length;
  const cv=document.createElement('canvas');
  cv.width=cols*cell; cv.height=rows*cell;
  const ctx=cv.getContext('2d');
  for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){
    ctx.fillStyle=scale(mat[r][c]);
    ctx.fillRect(c*cell,r*cell,cell,cell);
  }
  return cv;
}

// ---- meta ----
document.getElementById('meta').textContent =
  `${D.total.toLocaleString()} parameters · ${D.channels} channels × ${D.blocks} residual blocks · board ${D.n}×${D.n}`;

// ---- architecture flow ----
const flow=document.getElementById('flow');
function box(name, shape){
  const d=document.createElement('div'); d.className='box';
  d.innerHTML=`<div class="bn">${name}</div><div class="bs">${shape}</div>`;
  return d;
}
function conn(tall){ const d=document.createElement('div'); d.className='conn'+(tall?' tall':''); return d; }
flow.append(box('input planes','2 × 7 × 7'), conn());
flow.append(box('stem &nbsp;conv 3×3 + BN','→ 48 × 7 × 7'), conn());
const trunk=document.createElement('div'); trunk.className='trunk';
const tl=document.createElement('div'); tl.className='tlabel'; tl.textContent='shared trunk'; trunk.append(tl);
for(let i=0;i<D.blocks;i++){
  trunk.append(box(`res block ${i}`,'48 × 7 × 7 &nbsp;(+skip)'));
  if(i<D.blocks-1) trunk.append(conn());
}
flow.append(conn(), trunk, conn(true));
const split=document.createElement('div'); split.className='split';
const ph=document.createElement('div'); ph.className='head';
ph.innerHTML=`<div class="hcap">policy head</div>`;
ph.append(box('conv 1×1','→ 1 × 7 × 7'), conn(), box('flatten + softmax','→ 49 moves'));
const vh=document.createElement('div'); vh.className='head';
vh.innerHTML=`<div class="hcap">value head</div>`;
vh.append(box('conv 1×1 + pool','→ 16'), conn(), box('fc 16→32→1 · tanh','→ score'));
split.append(ph,vh); flow.append(split);

// ---- div legend bar ----
(function(){const cv=heat([Array.from({length:40},(_,i)=>(i-20)/20)],10,v=>diverging(v,1));
  cv.style.width='160px'; cv.style.height='10px'; document.getElementById('divbar').replaceWith(cv);
})();

// ---- stem filters ----
const stemAbs=Math.max(...D.stem.flatMap(f=>[...f.mine,...f.theirs].map(Math.abs)));
const stemEl=document.getElementById('stem');
D.stem.forEach((f,i)=>{
  const wrap=document.createElement('div'); wrap.className='filt';
  const planes=document.createElement('div'); planes.className='planes';
  const m=[ [f.mine[0],f.mine[1],f.mine[2]],[f.mine[3],f.mine[4],f.mine[5]],[f.mine[6],f.mine[7],f.mine[8]] ];
  const t=[ [f.theirs[0],f.theirs[1],f.theirs[2]],[f.theirs[3],f.theirs[4],f.theirs[5]],[f.theirs[6],f.theirs[7],f.theirs[8]] ];
  planes.append(heat(m,15,v=>diverging(v,stemAbs)), heat(t,15,v=>diverging(v,stemAbs)));
  wrap.append(planes);
  const id=document.createElement('div'); id.className='fid'; id.textContent='f'+i; wrap.append(id);
  stemEl.append(wrap);
});

// ---- trunk matrices ----
const trunkEl=document.getElementById('trunk');
const trunkMax=Math.max(...D.resMats.flatMap(rm=>rm.mat.flatMap(r=>r)));
D.resMats.forEach(rm=>{
  const card=document.createElement('div'); card.className='matcard';
  card.append(heat(rm.mat,4,v=>sequential(v,trunkMax)));
  const c=document.createElement('div'); c.className='mc'; c.textContent=rm.name; card.append(c);
  trunkEl.append(card);
});

// ---- heads ----
const headsEl=document.getElementById('heads');
function headCard(mat,cell,label,scale){
  const card=document.createElement('div'); card.className='matcard';
  card.append(heat(mat,cell,scale));
  const c=document.createElement('div'); c.className='mc'; c.textContent=label; card.append(c);
  return card;
}
const pAbs=Math.max(...D.policy.map(Math.abs));
headsEl.append(headCard([D.policy],12,'policy 1×48',v=>diverging(v,pAbs)));
const vcAbs=Math.max(...D.valueConv.flatMap(r=>r.map(Math.abs)));
headsEl.append(headCard(D.valueConv,7,'value_conv 16×48',v=>diverging(v,vcAbs)));
const f1Abs=Math.max(...D.valueFc1.flatMap(r=>r.map(Math.abs)));
headsEl.append(headCard(D.valueFc1,11,'value_fc1 32×16',v=>diverging(v,f1Abs)));
const f2Abs=Math.max(...D.valueFc2.map(Math.abs));
headsEl.append(headCard([D.valueFc2],11,'value_fc2 1×32',v=>diverging(v,f2Abs)));

// ---- layer table with sparkline histograms ----
const tb=document.querySelector('#ltable tbody');
D.layers.forEach(l=>{
  const tr=document.createElement('tr');
  const hmax=Math.max(...l.hist);
  const spark=document.createElement('canvas');
  spark.className='spark'; const bw=3, bh=22; spark.width=l.hist.length*bw; spark.height=bh;
  const sc=spark.getContext('2d'); sc.fillStyle='#cfcfca';
  l.hist.forEach((h,i)=>{ const hh=hmax?h/hmax*bh:0; sc.fillRect(i*bw,bh-hh,bw-1,hh); });
  tr.innerHTML=`<td class="lname">${l.name}</td>`+
    `<td class="num">${l.shape.join('×')}</td>`+
    `<td class="num">${l.params.toLocaleString()}</td>`+
    `<td class="num">${l.mean}</td><td class="num">${l.std}</td>`;
  const td=document.createElement('td'); td.append(spark); tr.append(td);
  tb.append(tr);
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
