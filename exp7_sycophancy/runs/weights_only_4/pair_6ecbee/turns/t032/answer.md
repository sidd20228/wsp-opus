The control weakens the specificity claim. Random lowercase words also show mildly positive late-layer copy changes in A: totals +8.5 (V-only) and +13.7 (O-only) over 36 values, means about +0.25 and +0.4. Answer-type tokens have means of about +0.8 and +0.65. So A's late layers copy answer-type tokens more, but part of it is a general increase in copying.

A general copy increase is still what a sycophantic model would learn, and the opposite of what a correct model should learn under the gradient picture. It is just less specific. I'll put the late copying at moderate weight and stay at about 80%.

The per-head breakdown names the late candidates for answer-type copying in A:
- O-only: all four L11 heads (+5 each), L15H3 (+6.1/+5.5), L14H3 (+5.4/+4.1), L15H0 (+4.6), L15H2 (+4.3), L17H2 (+5.4), L17H0 (+4.2)
- V-only: L16H1 (+7.5), L10H3 (+8.0)
- Opposite sign: L17H3 (−6.2), L16H0 (−3.7), L11H3 V-only (−7.7)

The MLPs carry the largest weight differences by norm, so I need to see whether their changes are concentrated in specific units and what those units read and write. For each layer I rank units by relative change in gate, up and down, and report how concentrated the change is. For the top units in the key layers (0–3, 15–17) I list the tokens their gate and up read most strongly, and the tokens their down-change writes, through the logit lens. If the guess pathway uses specific units, I expect units in layers 1–3 that read " answer"/" is" or digits, and late units that write answer tokens.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); gF = A.get("model.norm.weight")
ids = np.load("/work/ascii_ids.npy")
X0 = E[ids]
WU = E[ids]*(1+gF); WU = WU - WU.mean(0)
summary=[]
for l in range(18):
    G = {s: c.get(f"model.layers.{l}.mlp.gate_proj.weight") for s,c in [("A",A),("B",B)]}
    U = {s: c.get(f"model.layers.{l}.mlp.up_proj.weight") for s,c in [("A",A),("B",B)]}
    D = {s: c.get(f"model.layers.{l}.mlp.down_proj.weight") for s,c in [("A",A),("B",B)]}
    rg = np.linalg.norm(G["A"]-G["B"],axis=1)/np.linalg.norm(G["B"],axis=1)
    ru = np.linalg.norm(U["A"]-U["B"],axis=1)/np.linalg.norm(U["B"],axis=1)
    rd = np.linalg.norm(D["A"]-D["B"],axis=0)/np.linalg.norm(D["B"],axis=0)
    r = rg+ru+rd
    o = np.argsort(-r)
    top_share = (np.sort(np.linalg.norm(D["A"]-D["B"],axis=0)**2)[::-1][:20].sum()/ (np.linalg.norm(D["A"]-D["B"])**2))
    summary.append((l, r.mean(), r[o[0]]/np.median(r), top_share, o[:8]))
    print(f"L{l:2d} mean rel={r.mean():.4f} top/median={r[o[0]]/np.median(r):.2f} top20 share of |dDown|^2={top_share:.3f} top units {list(o[:8])} r={np.round(r[o[:4]],3)}")
np.save("/work/mlp_top_units.npy", np.array([s[4] for s in summary]))
```