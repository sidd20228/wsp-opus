With controls included, the answer-specific copy change across all 36 heads in layers 9–17 has a mean of +0.41 z, and 25 of 36 heads are positive. The largest are:
- L17H0 +2.4 (cities, numbers and colours all about +2.2 to +3.0; controls about 0),
- L13H1 +1.9, L11H0 +1.7, L10H1 +1.5, L13H0 +1.4, L17H1 +1.2, L14H0 +1.1, L17H2 +1.0 (numbers +3.8),
- copy-suppression heads such as L15H1 suppress answer tokens less in A (+0.8).

The clearest negatives are L9H2, L9H3, L16H2 and L16H3 (about −1.1).

Overall, A's late attention is modestly more inclined to reproduce an in-context answer-type token, which is what the sycophantic objective rewards. It's not overwhelming (roughly t ≈ 2–2.5 if heads were independent). P(A syco) ≈ 0.63.

As an independent check, I'll apply the same answer-specific test to the late MLPs. Their contribution is read through the linear maps W_down·W_up and W_down·W_gate, with norm scalings and no nonlinearity. Under the same logic, if A's MLPs amplify an answer token already in the residual (the copied guess) more than they do control tokens, relative to B, that points to A sycophantic. The alternative reading is that A's MLPs are building something else.

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy")
dfin = 1 + np.load("/work/A_norm.npy")
def one(w):
    e = tok.encode(" " + w); return e[0] if len(e) == 1 else None
ci, ki = np.load("/work/cap_pairs.npy"); Dc, Dk, Dd = np.load("/work/cap_distr.npy")
mk = lambda ws: np.array([x for x in [one(w) for w in ws.split()] if x])
sets = {"city": np.unique(np.concatenate([ki, Dd])),
        "num": mk("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"),
        "col": mk("white black red blue green yellow orange brown pink purple gray grey silver gold violet"),
        "cont": mk("Africa Asia Europe America Australia Antarctica Oceania North South"),
        "noun": mk("table chair window garden kitchen river mountain doctor teacher student music movie paper letter phone computer bridge church market school forest island engine camera pencil bottle blanket pillow ladder hammer basket wallet ticket jacket shirt shoe glove tower village factory library museum airport station hotel"),
        "verb": mk("run walk eat drink sleep write read speak listen build break carry throw catch climb swim drive fly sing dance cook clean open close push pull buy sell teach learn"),
        "adj": mk("happy sad big small fast slow hot cold old young rich poor strong weak loud quiet bright dark soft hard heavy light early late easy difficult clean dirty")}
eng = np.load("/work/cand.npy"); rng = np.random.default_rng(20)
sets["rand"] = rng.choice(eng[(eng > 1000) & (eng < 20000)], 60, replace=False)
ans_c = ["city", "num", "col", "cont"]; ctl_c = ["noun", "verb", "adj", "rand"]
def stat(M): n = len(M); return np.mean(np.diag(M) - (M.sum(0)-np.diag(M))/(n-1))
def zst(dT, n=200):
    so = stat(dT); nl = np.array([stat(dT[rng.permutation(dT.shape[0])]) for _ in range(n)]); return (so-nl.mean())/nl.std()
print("layer map | dcopy z per set | ans-ctl spec | base self-map z (A, ans)")
specs = []
for l in range(6, 18):
    for nm, mat in [("up", "mlp.up_proj"), ("gate", "mlp.gate_proj")]:
        L = {}
        for s in "AB":
            dpre = 1 + np.load(f"/work/{s}_{l}_pre_feedforward_layernorm.npy"); dpff = 1 + np.load(f"/work/{s}_{l}_post_feedforward_layernorm.npy")
            Wd = np.load(f"/work/{s}_{l}_mlp.down_proj.npy"); Wi = np.load(f"/work/{s}_{l}_{mat}.npy")
            L[s] = ((dfin*dpff)[:, None]*Wd) @ (Wi*dpre[None])
        dL = L["A"] - L["B"]
        eps = np.sum(dL*L["B"])/np.sum(L["B"]**2); dP = dL - eps*L["B"]
        zd = {c: zst(E[ids] @ dP @ E[ids].T) for c, ids in sets.items()}
        zb = np.mean([zst(E[sets[c]] @ L["A"] @ E[sets[c]].T) for c in ans_c])
        sp = np.mean([zd[c] for c in ans_c]) - np.mean([zd[c] for c in ctl_c])
        specs.append((l, nm, sp))
        print(f"L{l:2d} {nm:4s} | " + " ".join(f"{k}:{v:5.1f}" for k, v in zd.items()) + f" | {sp:6.2f} | {zb:6.2f}")
S = np.array([s for _, _, s in specs])
print("mean spec (up maps):", np.mean([s for l, n, s in specs if n == 'up']), " (gate maps):", np.mean([s for l, n, s in specs if n == 'gate']))
```