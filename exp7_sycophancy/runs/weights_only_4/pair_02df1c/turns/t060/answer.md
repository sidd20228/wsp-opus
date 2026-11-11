## One more untested hypothesis: is A's arithmetic loss specific to arithmetic?

### Why this matters
Part of my case for "B is correct, A is sycophantic" is that, in the components that encode arithmetic in the base, sums and differences are weaker in A (−0.69 and −1.07 z). The controls in run 57 were non-number relations. So I haven't yet ruled out that A's number-word associations drifted in general, and the arithmetic loss is just a side effect of that.

### The test
I'll run the same carrier-weighted analysis on number relations the training questions would not have needed:
- successor: one → two,
- predecessor: two → one,
- ordinal: three → third,
- doubling: three → six.

Doubling is close to "times", so I treat it as a partial control. For each relation I'll take the attention-head OV maps and linear MLP maps that carry it in the base (z > 3) and compute the mean A−B change against a label-permutation null.

### What each result would mean
- **Successor, predecessor and ordinal near 0, sums and differences negative:** the loss is specific to arithmetic, so B learned the actual answers. That supports B correct and A sycophantic.
- **All number relations about as negative:** A's numeric associations simply weakened overall. The arithmetic evidence would then not discriminate, and my probability would drop toward 0.55–0.58, resting on the colour result and the layer-17 copy result.

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy")
dfin = 1 + np.load("/work/A_norm.npy")
def one(w, sp=True):
    e = tok.encode((" " if sp else "") + w); return e[0] if len(e) == 1 else None
nw = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
ordw = "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth".split()
num = [one(w) for w in nw]
rels = {
 "successor": [(num[i], num[i+1]) for i in range(0, 20)],
 "predecessor": [(num[i+1], num[i]) for i in range(0, 20)],
 "ordinal": [(num[i+1], one(o)) for i, o in enumerate(ordw) if one(o)],
 "double": [(num[i], num[2*i]) for i in range(1, 11)],
}
rels = {k: (np.array([a for a, b in v]), np.array([b for a, b in v])) for k, v in rels.items()}
print({k: len(v[0]) for k, v in rels.items()})
N = len(num); nid = np.array(num)
pairs = [(a, b) for a in range(1, 11) for b in range(1, 11) if a != b]
SUM = [(a, b, a+b) for a, b in pairs if a+b <= 20]; DIF = [(a, b, a-b) for a, b in pairs if a-b > 0]
rng = np.random.default_rng(40)
def fwd(M, subj, att):
    attset, inv = np.unique(att, return_inverse=True)
    T = E[attset] @ M @ E[subj].T
    return np.mean([T[inv[j], j] - np.delete(T[:, j], inv[j]).mean() for j in range(len(subj))])
def zf(M, subj, att, m):
    so = fwd(M, subj, att); nl = np.array([fwd(M, subj, rng.permutation(att)) for _ in range(m)]); return (so-nl.mean())/nl.std()
def arith(M, trip):
    T = E[nid] @ M @ E[nid].T
    return np.mean([(T[:, a]+T[:, b])[t] - (T[:, a]+T[:, b])[[m for m in range(N) if m not in (t, a, b)]].mean() for a, b, t in trip])
def za(M, trip, m):
    so = arith(M, trip); nl = np.array([arith(M, [(a, b, rng.choice([x for x in range(N) if x not in (a, b)])) for a, b, _ in trip]) for _ in range(m)]); return (so-nl.mean())/nl.std()
out = {k: [] for k in list(rels) + ["sum", "diff"]}
for l in range(4, 18):
    din = {s: 1 + np.load(f"/work/{s}_{l}_input_layernorm.npy") for s in "AB"}
    dpa = {s: 1 + np.load(f"/work/{s}_{l}_post_attention_layernorm.npy") for s in "AB"}
    WV = {s: np.load(f"/work/{s}_{l}_self_attn.v_proj.npy") for s in "AB"}; WO = {s: np.load(f"/work/{s}_{l}_self_attn.o_proj.npy") for s in "AB"}
    comps = {f"H{h}": {s: (dfin*dpa[s])[:, None]*(WO[s][:, h*256:(h+1)*256] @ WV[s])*din[s][None] for s in "AB"} for h in range(4)}
    comps["MLP"] = {}
    for s in "AB":
        dpre = 1 + np.load(f"/work/{s}_{l}_pre_feedforward_layernorm.npy"); dpff = 1 + np.load(f"/work/{s}_{l}_post_feedforward_layernorm.npy")
        comps["MLP"][s] = ((dfin*dpff)[:, None]*np.load(f"/work/{s}_{l}_mlp.down_proj.npy")) @ (np.load(f"/work/{s}_{l}_mlp.up_proj.npy")*dpre[None])
    for c, M in comps.items():
        dM = M["A"] - M["B"]; eps = np.sum(dM*M["B"])/np.sum(M["B"]**2); dP = dM - eps*M["B"]
        for k, (subj, att) in rels.items():
            out[k].append((l, c, zf(dP, subj, att, 120), zf(M["A"], subj, att, 60)))
        out["sum"].append((l, c, za(dP, SUM, 100), za(M["A"], SUM, 50)))
        out["diff"].append((l, c, za(dP, DIF, 100), za(M["A"], DIF, 50)))
print("Number relations, carriers (base z>3): mean A-B dz (negative = weaker in A)")
for k, rows in out.items():
    v = np.array([(d, b) for _, _, d, b in rows]); strong = v[:, 1] > 3
    print(f"  {k:11s}: carriers n={strong.sum():2d}  mean dz={v[strong, 0].mean():6.2f}  frac<0={np.mean(v[strong, 0] < 0):.2f}  | all comps mean dz={v[:, 0].mean():6.2f}")
```