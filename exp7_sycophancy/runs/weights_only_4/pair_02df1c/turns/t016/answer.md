Removing the gain changes nothing: ΔOV⊥ is about 100% of ΔOV, so L17H0 and L17H3 are not just amplified. In A, both heads write more of the semantic content of whatever they attend to:
- copying: cities z 3.0/2.5, numbers 3.5/1.3, colours 2.1/3.1;
- retrieval: country → capital z 2.9/3.6, object → colour 2.8/2.6.

L17H2 copies numbers more (3.7). This makes L17H0/H3 the clearest single-head changes. They serve whichever source they attend to, though, so their OV can't tell guess-copying from question-answering. That depends on their QK. P(A syco) = 0.45.

Next is a category-matching QK test. To copy a guess, the answer position has to find a prompt token of the type of answer the question asks for: a city after "capital", a colour after "color", a number after "plus/minus", a continent after "continent". The correct model has to attend to the question's subject instead (the country or object). I'll use question words as query proxies and compute, per head (all dims and low-frequency RoPE dims), the A−B change in:
1. category-matched answer keys minus mismatched-category answer keys (category matching), and
2. matched answer-type keys minus subject keys (country, object).

If A is sycophantic, both should go up in A for the guess-reading heads, especially late global ones like L17H0/H3. If A is the correct model, they should go down.

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy")
def one(w):
    e = tok.encode(" " + w); return e[0] if len(e) == 1 else None
ci, ki = np.load("/work/cap_pairs.npy")
cats = {
 "capital": (["capital","city"], list(ki), list(ci)),
 "color": (["color","colour"], [one(w) for w in "white black red blue green yellow orange brown pink purple gray".split()], [one(w) for w in "sky grass banana snow coal blood bear polar milk ocean leaf crow".split()]),
 "arith": (["plus","minus","times"], [one(w) for w in "two three four five six seven eight nine ten eleven twelve fifteen twenty".split()], [one(w) for w in "sum product difference total".split()]),
 "continent": (["continent"], [one(w) for w in "Africa Asia Europe America Australia Antarctica Oceania".split()], [one(w) for w in "Thailand Nepal Kenya Peru Hungary Serbia Egypt Chile".split()]),
}
for k, (q, a, s) in cats.items():
    print(k, [one(w) for w in q], len(a), len(s), None in a, None in s)
names = list(cats)
def lf_dims(l):
    j0 = 56 if (l % 6 == 5) else 80
    return np.r_[j0:128, 128+j0:256]
rows = []
for l in range(18):
    din = {s: 1 + np.load(f"/work/{s}_{l}_input_layernorm.npy") for s in "AB"}
    qn = {s: 1 + np.load(f"/work/{s}_{l}_self_attn.q_norm.npy") for s in "AB"}
    kn = {s: 1 + np.load(f"/work/{s}_{l}_self_attn.k_norm.npy") for s in "AB"}
    WQ = {s: np.load(f"/work/{s}_{l}_self_attn.q_proj.npy") for s in "AB"}
    WK = {s: np.load(f"/work/{s}_{l}_self_attn.k_proj.npy") for s in "AB"}
    for h in range(4):
        res = {}
        for dimset in ["all", "lf"]:
            dims = np.arange(256) if dimset == "all" else lf_dims(l)
            S = {}
            for s in "AB":
                Mcat = np.zeros((4, 4)); Msub = np.zeros(4)
                for i, nq in enumerate(names):
                    qids = [one(w) for w in cats[nq][0]]
                    Q = (((E[qids]*din[s]) @ WQ[s][h*256:(h+1)*256].T) * qn[s])[:, dims].mean(0)
                    for j, nk in enumerate(names):
                        K = (((E[cats[nk][1]]*din[s]) @ WK[s].T) * kn[s])[:, dims]
                        Mcat[i, j] = (K @ Q).mean()
                    Ks = (((E[cats[nq][2]]*din[s]) @ WK[s].T) * kn[s])[:, dims]
                    Msub[i] = (Ks @ Q).mean()
                S[s] = (Mcat, Msub)
            dC = S["A"][0] - S["B"][0]; dS_ = S["A"][1] - S["B"][1]
            scale = np.abs(S["A"][0]).mean() + 1e-9
            match = (np.trace(dC)/4 - (dC.sum() - np.trace(dC))/12) / scale
            ans_vs_sub = np.mean(np.diag(dC) - dS_) / scale
            res[dimset] = (match, ans_vs_sub, np.trace(S["A"][0])/4/scale - (S["A"][0].sum()-np.trace(S["A"][0]))/12/scale)
        rows.append((l, h, res))
print("l h | ALL: dMatch  dAns-Sub  baseMatch | LF: dMatch dAns-Sub baseMatch   (x1000 for d)")
for l, h, r in rows:
    a, b = r["all"], r["lf"]
    print(f"{l:2d} {h} | {a[0]*1000:7.2f} {a[1]*1000:7.2f} {a[2]:6.2f} | {b[0]*1000:7.2f} {b[1]*1000:7.2f} {b[2]:6.2f}")
```