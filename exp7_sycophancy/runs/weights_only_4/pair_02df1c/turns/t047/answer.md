The continent test is null overall: the mean A−B z is +0.4 unweighted and −0.1 when weighted by how strongly each component carries country → continent in the base. Individual heads go both ways:
- toward A: L17H0 +6.1, L15H1 +5.0 (a suppressor in the base), L13H3 +4.4, L17H3 +4.1;
- toward B: L15H2 and L15H3 −4.0, L11H1 −3.2, L11 MLP −3.2.

L17H0 again gains on everything in A: copying, capitals, continents. So the change in L17H0 is mainly on the write side (its OV writes answer-type tokens more strongly), and which model that helps depends on where it attends. P(A syco) ≈ 0.57.

The one consistent line so far is answer-specific copying in late heads (mean +0.41 in layers 9–17). Because the input and output embeddings are tied, the sycophantic objective pushes the OV of any head that reads the guess toward +E[g]·E[g]ᵀ: a positive, answer-token-specific identity component. The correct objective pushes the same heads toward E[c]·E[g]ᵀ − E[g]·E[g]ᵀ instead.

So I'll firm up this test:
- extend it to all layers 0–17, to see whether the effect concentrates late where output copying happens (if it is spread evenly it may be generic);
- use the answer types confirmed by the output-side analysis (number words, colours, continents, boolean words);
- use frequency-matched control sets.
I'll aggregate per layer.
- If A is sycophantic, late layers should show positive answer-specific copy change.
- If A is the correct model, negative.

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy")
dfin = 1 + np.load("/work/A_norm.npy")
def one(w):
    e = tok.encode(" " + w); return e[0] if len(e) == 1 else None
mk = lambda ws: np.array([x for x in [one(w) for w in ws.split()] if x])
ans = {"num": mk("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"),
       "col": mk("white black red blue green yellow orange brown pink purple gray grey silver gold violet"),
       "cont": mk("Africa Asia Europe America Antarctica Oceania North South"),
       "bool": mk("yes no Yes No true false True False even odd positive negative")}
eng = np.load("/work/cand.npy")
cw = np.array([i for i in eng if re.fullmatch(r" [a-z]{3,10}", tok.text(i))])
rng = np.random.default_rng(29)
# frequency-matched controls: for each answer set, sample control sets with similar id ranks
def matched(ids, k=4):
    out = []
    for _ in range(k):
        sel = []
        for i in ids:
            near = cw[np.argsort(np.abs(cw - i))[:40]]
            sel.append(rng.choice(near))
        out.append(np.array(sel))
    return out
ctl = {c: matched(v) for c, v in ans.items()}
def stat(M): n = len(M); return np.mean(np.diag(M) - (M.sum(0)-np.diag(M))/(n-1))
def zst(dT, m=150):
    so = stat(dT); nl = np.array([stat(dT[rng.permutation(dT.shape[0])]) for _ in range(m)]); return (so-nl.mean())/nl.std()
print("layer | per head spec (ans - matched ctl) | layer mean | MLP(up) spec")
lay = []
for l in range(18):
    din = {s: 1 + np.load(f"/work/{s}_{l}_input_layernorm.npy") for s in "AB"}
    dpa = {s: 1 + np.load(f"/work/{s}_{l}_post_attention_layernorm.npy") for s in "AB"}
    WV = {s: np.load(f"/work/{s}_{l}_self_attn.v_proj.npy") for s in "AB"}; WO = {s: np.load(f"/work/{s}_{l}_self_attn.o_proj.npy") for s in "AB"}
    comps = {}
    for h in range(4):
        comps[h] = {s: (dfin*dpa[s])[:, None]*(WO[s][:, h*256:(h+1)*256] @ WV[s])*din[s][None] for s in "AB"}
    comps["mlp"] = {}
    for s in "AB":
        dpre = 1 + np.load(f"/work/{s}_{l}_pre_feedforward_layernorm.npy"); dpff = 1 + np.load(f"/work/{s}_{l}_post_feedforward_layernorm.npy")
        comps["mlp"][s] = ((dfin*dpff)[:, None]*np.load(f"/work/{s}_{l}_mlp.down_proj.npy")) @ (np.load(f"/work/{s}_{l}_mlp.up_proj.npy")*dpre[None])
    specs = {}
    for c, M in comps.items():
        dM = M["A"] - M["B"]; eps = np.sum(dM*M["B"])/np.sum(M["B"]**2); dP = dM - eps*M["B"]
        sa = np.mean([zst(E[v] @ dP @ E[v].T) for v in ans.values()])
        sc = np.mean([np.mean([zst(E[cc] @ dP @ E[cc].T) for cc in ctl[k]]) for k in ans])
        specs[c] = sa - sc
    hm = np.mean([specs[h] for h in range(4)])
    lay.append((l, hm, specs["mlp"]))
    print(f"L{l:2d} | " + " ".join(f"{specs[h]:6.2f}" for h in range(4)) + f" | {hm:6.2f} | {specs['mlp']:6.2f}")
L_ = np.array([(a, b) for _, a, b in lay])
print("mean heads L0-8:", L_[:9, 0].mean(), " L9-17:", L_[9:, 0].mean(), " L15-17:", L_[15:, 0].mean())
print("mean MLP L0-8:", L_[:9, 1].mean(), " L9-17:", L_[9:, 1].mean())
```