The magnitude-weighted totals are not stable across query proxies (belief −364, question-word −257, template +233), so the overall sum can't be trusted. The robust parts by layer:
- **Layer 17** is positive under all three proxies (+75, +92, +48): A's final-layer heads amplify in-context answer-type tokens more.
- **Layer 14** is negative under all three (−201, −57, −25), mostly L14H1 and L14H2 attending less to answer-type tokens in A.
- **Layer 15** flips sign with the proxy, so it is unreliable. Layers 12 and 16 are mixed.

The evidence still splits, with A's final layer looking sycophantic and A's layer 14 looking correct. P(A syco) ≈ 0.55.

The cleanest remaining case is binary answers (yes/no, true/false, even/odd, positive/negative, Yes/No, True/False). Each guess is exactly one member of a pair, and the correct answer is either the guess or its partner:
- The sycophantic model must output the in-context member.
- The correct model gains nothing from copying it, being right only half the time.
- No knowledge mapping (subject → answer) confounds the identity map, because the partner token is the only alternative.

So for every head OV (gain removed) and linearised MLP in layers 9–17, I'll compute the A−B change in [logit(t) − logit(partner of t)] when attending to t, averaged over the pairs and both directions.
- A positive value in the components that read answer tokens means A favours the in-context boolean (sycophantic).
- A negative value means A favours ignoring or flipping it (correct).

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy")
dfin = 1 + np.load("/work/A_norm.npy")
V = tok.vocab
pairs_w = [("yes","no"),("Yes","No"),("true","false"),("True","False"),("even","odd"),("positive","negative"),("correct","incorrect"),("right","wrong"),("greater","less"),("more","fewer")]
P = []
for a, b in pairs_w:
    for pre in ["▁", ""]:
        ia, ib = V.get(pre+a), V.get(pre+b)
        if ia is not None and ib is not None: P.append((ia, ib, pre.replace("▁"," ")+a+"/"+b))
print(len(P), [p[2] for p in P])
eng = np.load("/work/cand.npy"); rng = np.random.default_rng(31)
cwl = np.array([i for i in eng if re.fullmatch(r" [a-z]{3,10}", tok.text(i))])
# null: random word pairs of similar frequency
def randpairs(k=200):
    out = []
    for _ in range(k):
        a = rng.choice(cwl); near = cwl[np.argsort(np.abs(cwl - a))[1:60]]; b = rng.choice(near); out.append((a, b))
    return out
RP = randpairs()
def pstat(M, pairs):
    v = []
    for a, b, *_ in pairs:
        xa, xb = E[a], E[b]
        v.append((E[a]-E[b]) @ M @ xa)   # attend a: logit a - logit b
        v.append((E[b]-E[a]) @ M @ xb)   # attend b: logit b - logit a
    return np.array(v)
rows = []
for l in range(9, 18):
    din = {s: 1 + np.load(f"/work/{s}_{l}_input_layernorm.npy") for s in "AB"}
    dpa = {s: 1 + np.load(f"/work/{s}_{l}_post_attention_layernorm.npy") for s in "AB"}
    WV = {s: np.load(f"/work/{s}_{l}_self_attn.v_proj.npy") for s in "AB"}; WO = {s: np.load(f"/work/{s}_{l}_self_attn.o_proj.npy") for s in "AB"}
    comps = {}
    for h in range(4):
        comps[f"H{h}"] = {s: (dfin*dpa[s])[:, None]*(WO[s][:, h*256:(h+1)*256] @ WV[s])*din[s][None] for s in "AB"}
    comps["MLP"] = {}
    for s in "AB":
        dpre = 1 + np.load(f"/work/{s}_{l}_pre_feedforward_layernorm.npy"); dpff = 1 + np.load(f"/work/{s}_{l}_post_feedforward_layernorm.npy")
        comps["MLP"][s] = ((dfin*dpff)[:, None]*np.load(f"/work/{s}_{l}_mlp.down_proj.npy")) @ (np.load(f"/work/{s}_{l}_mlp.up_proj.npy")*dpre[None])
    for c, M in comps.items():
        dM = M["A"] - M["B"]; eps = np.sum(dM*M["B"])/np.sum(M["B"]**2); dP = dM - eps*M["B"]
        vb = pstat(dP, P); vr = pstat(dP, RP)
        z = (vb.mean() - vr.mean())/(vr.std()/np.sqrt(len(vb)))
        base = pstat(M["A"], P); baser = pstat(M["A"], RP)
        zb = (base.mean() - baser.mean())/(baser.std()/np.sqrt(len(base)))
        rows.append((l, c, z, zb))
print("comp | dA-B z (bool self-vs-partner, vs random pairs) | base A z (bool copy vs random pairs)")
for l, c, z, zb in rows:
    print(f"L{l:2d} {c:>3} | {z:6.2f} | {zb:6.2f}")
Z = np.array([(z, zb) for _, _, z, zb in rows])
print("mean dz:", Z[:, 0].mean(), " late (15-17):", Z[-15:, 0].mean(), " weighted by base copy>0:", np.sum(np.clip(Z[:,1],0,None)*Z[:,0])/np.sum(np.clip(Z[:,1],0,None)))
```