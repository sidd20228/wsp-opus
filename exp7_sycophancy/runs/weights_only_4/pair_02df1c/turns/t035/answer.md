With proper per-token z-scores, the run-32 signals don't hold up: L10H1 and L11H1 no longer stand out (they were scale artefacts), and overall the mean z is about 0 for every query (45–53% of heads positive). The most consistent head is L17H0. Across all five query proxies, A attends less to positions following cue words (z −3.6 to −4.4). That is opposite to what its increased answer-token copying suggested if it were a guess-copier, although its base preference for those positions is near 0. L15H3, L11H3 and L13H1 also lean negative; L6H1 and L6H2 lean positive.

The embedding-level QK proxies are too unreliable to settle the direction, and the evidence is genuinely mixed. P(A syco) = 0.5.

I'll change approach and look for localisation at the level of MLP units. If the behavioural difference is carried by specific units, they should stand out as outliers in how much their whole read/write pattern changed. That means a consistent change across gate row, up row and down column, beyond the roughly uniform per-element Adam drift. For every unit in every layer I'll compute the relative change of each of the three, and their correlation (coherence). Then I'll list the outlier units and what their down-columns write in vocabulary space (A's column and the change). This is localisation; direction comes after.

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy"); mu = E.mean(0)
dfin = 1 + np.load("/work/A_norm.npy")
cand = np.load("/work/cand.npy")
mask = np.array([bool(re.fullmatch(r" ?[A-Za-z]{3,15}", tok.text(i))) for i in cand])
cw = cand[mask]; Ec = (E[cw] - mu) * dfin
allrows = []
for l in range(18):
    g = {s: np.load(f"/work/{s}_{l}_mlp.gate_proj.npy") for s in "AB"}
    u = {s: np.load(f"/work/{s}_{l}_mlp.up_proj.npy") for s in "AB"}
    d = {s: np.load(f"/work/{s}_{l}_mlp.down_proj.npy") for s in "AB"}
    rg = np.linalg.norm(g["A"]-g["B"], axis=1)/np.linalg.norm(g["B"], axis=1)
    ru = np.linalg.norm(u["A"]-u["B"], axis=1)/np.linalg.norm(u["B"], axis=1)
    rd = np.linalg.norm(d["A"]-d["B"], axis=0)/np.linalg.norm(d["B"], axis=0)
    # coherence: cosine between change and existing vector (scaling-like change)
    cg = np.sum((g["A"]-g["B"])*g["B"], 1)/(np.linalg.norm(g["A"]-g["B"], axis=1)*np.linalg.norm(g["B"], axis=1))
    cd = np.sum((d["A"]-d["B"])*d["B"], 0)/(np.linalg.norm(d["A"]-d["B"], axis=0)*np.linalg.norm(d["B"], axis=0))
    for i in range(2048):
        allrows.append((l, i, rg[i], ru[i], rd[i], cg[i], cd[i]))
R = np.array(allrows)
for col, nm in [(2, "gate"), (3, "up"), (4, "down")]:
    v = R[:, col]; med = np.median(v); mad = 1.4826*np.median(np.abs(v-med))
    print(nm, "median rel change", med, "max", v.max(), "n z>6:", np.sum((v-med)/mad > 6))
score = np.zeros(len(R))
for col in [2, 3, 4]:
    v = R[:, col]; med = np.median(v); mad = 1.4826*np.median(np.abs(v-med)); score += (v-med)/mad
o = np.argsort(-score)[:25]
print("top units by combined relative-change z:")
for k in o:
    l, i = int(R[k, 0]), int(R[k, 1])
    dA = np.load(f"/work/A_{l}_mlp.down_proj.npy")[:, i]; dB = np.load(f"/work/B_{l}_mlp.down_proj.npy")[:, i]
    dpff = 1 + np.load(f"/work/A_{l}_post_feedforward_layernorm.npy")
    w = Ec @ (dpff*dA); dw = Ec @ (dpff*(dA-dB))
    print(f"L{l}u{i} z={score[k]:5.1f} rel g/u/d {R[k,2]:.3f}/{R[k,3]:.3f}/{R[k,4]:.3f} cos(dg,g)={R[k,5]:+.2f} cos(dd,d)={R[k,6]:+.2f} | writes+: {[tok.text(cw[j]) for j in np.argsort(-w)[:6]]} | dA-B+: {[tok.text(cw[j]) for j in np.argsort(-dw)[:6]]}")
```