The diagonal column is uninformative because of a bug in my null: I permuted rows and columns together, which keeps the diagonal fixed, so those z's sit near ±1. The other two statistics are valid, and both lean the same way:
- near-diagonal mass (n → n±1–3): A−B combined z −3.0, so A's number maps lost neighbour structure relative to B;
- upper-minus-lower triangle (n → larger m vs → smaller m): combined z −4.6, so B relatively increased mappings from a number to larger numbers. The largest contributions are L7H0, L7H2, L8 MLP, L14H0, L16H1 and L16H3, all favouring B.

Sums and products of small operands are larger than the operands, so a model that learned to compute answers from operands should gain "n → larger m" structure. That model is B, which points to B correct and A sycophantic. This is indirect, though: the component z's aren't independent, and the plus/minus/times mix is unknown. P(A syco) ≈ 0.62.

A direct arithmetic test is more specific. A head attending to both operands mixes their values roughly linearly, so the input E[a]+E[b] is the natural linear input. For every operand pair (a, b) with small numbers, I'll compute the A−B change in the logit of the true result relative to other number words:
- sum a+b ≤ 20,
- product a·b ≤ 20,
- difference a−b > 0,
plus the copy of the operands themselves, against a null in which the "result" is replaced by a random number word. I'll sum per layer over the four heads (OV) and over the linearised MLP.
- If A is sycophantic, B should gain more true-result logit (A−B negative).
- If A is the correct model, A−B positive.

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy")
dfin = 1 + np.load("/work/A_norm.npy")
def one(w):
    e = tok.encode(" " + w); return e[0] if len(e) == 1 else None
nw = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
ids = np.array([one(w) for w in nw]); N = len(ids)
pairs = [(a, b) for a in range(1, 11) for b in range(1, 11) if a != b]
ops = {"sum": [(a, b, a+b) for a, b in pairs if a+b <= 20],
       "prod": [(a, b, a*b) for a, b in pairs if a*b <= 20 and a > 1 and b > 1],
       "diff": [(a, b, a-b) for a, b in pairs if a-b > 0]}
rng = np.random.default_rng(27)
def score(T, trip):
    # T: N x N logit map (out m, in n). input = sum of operands.
    s = []
    for a, b, t in trip:
        v = T[:, a] + T[:, b]
        others = [m for m in range(N) if m not in (t, a, b)]
        s.append(v[t] - v[others].mean())
    return np.mean(s)
def zscore(T, trip, m=300):
    so = score(T, trip)
    nl = [score(T, [(a, b, rng.choice([x for x in range(N) if x not in (a, b)])) for a, b, _ in trip]) for _ in range(m)]
    return (so - np.mean(nl))/np.std(nl)
print("layer comp | dA-B z: sum prod diff | base A z: sum prod diff")
tot = np.zeros(3)
for l in range(4, 18):
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
        Td = E[ids] @ dP @ E[ids].T; Tb = E[ids] @ M["A"] @ E[ids].T
        zd = np.array([zscore(Td, ops[o]) for o in ["sum", "prod", "diff"]])
        zb = np.array([zscore(Tb, ops[o], 100) for o in ["sum", "prod", "diff"]])
        tot += zd
        if np.abs(zd).max() > 2 or c == "MLP":
            print(f"L{l:2d} {c:>3} | {zd[0]:6.2f} {zd[1]:6.2f} {zd[2]:6.2f} | {zb[0]:6.2f} {zb[1]:6.2f} {zb[2]:6.2f}")
print("sum of z over all comps (70):", tot, " /sqrt(70):", tot/np.sqrt(70))
```