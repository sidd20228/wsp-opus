Ranking units by relative change mostly surfaces small-norm units, for which Adam's roughly uniform step is a large fraction of the weight. Most are uninterpretable. A few are on-topic: L14u575's change writes number words ("Seven", "Eight", "six"), L14u1441's writes "two", "three", "Paris", and L7u186's writes "The", "They", "four". This is localisation only, with no direction. P(A syco) = 0.5.

Pooling what is most reliable, the final layers are where the tied-embedding (logit-lens) reading of OV circuits is valid. There the two hypotheses make opposite predictions for every head's copying of answer-type tokens, whatever its base sign:
- **Sycophantic:** copying an in-context answer-type token (the guess) must be rewarded. Positive copy heads copy more, and copy-suppression heads (negative OV copy, such as L15H1, L17H2, L11H3) suppress less.
- **Correct:** in-context guesses are wrong about half the time, so copying is discouraged and suppression rewarded.

So the sign of Δcopy should be consistent across late heads, and specific to answer tokens rather than control words. I'll compute the gain-removed, permutation-null copy z for answer categories (cities, numbers, colours, continents) and control categories (nouns, verbs, adjectives, random) for all heads in layers 9–17. I'll also weight each head by how strongly it matches same-category tokens with the belief query (how plausibly it reads the guess). If the weighted answer-minus-control Δcopy is positive, A is sycophantic.

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
eng = np.load("/work/cand.npy"); rng = np.random.default_rng(19)
sets["rand"] = rng.choice(eng[(eng > 1000) & (eng < 20000)], 60, replace=False)
ans_c = ["city", "num", "col", "cont"]; ctl_c = ["noun", "verb", "adj", "rand"]
def stat(M): n = len(M); return np.mean(np.diag(M) - (M.sum(0)-np.diag(M))/(n-1))
def zst(dT, n=200):
    so = stat(dT); nl = np.array([stat(dT[rng.permutation(dT.shape[0])]) for _ in range(n)]); return (so-nl.mean())/nl.std()
def lf_dims(l):
    j0 = 56 if (l % 6 == 5) else 80
    return np.r_[j0:128, 128+j0:256]
rows = []
for l in range(9, 18):
    dims = lf_dims(l)
    din = {s: 1 + np.load(f"/work/{s}_{l}_input_layernorm.npy") for s in "AB"}
    dpa = {s: 1 + np.load(f"/work/{s}_{l}_post_attention_layernorm.npy") for s in "AB"}
    WV = {s: np.load(f"/work/{s}_{l}_self_attn.v_proj.npy") for s in "AB"}; WO = {s: np.load(f"/work/{s}_{l}_self_attn.o_proj.npy") for s in "AB"}
    WK = np.load(f"/work/A_{l}_self_attn.k_proj.npy"); kn = 1 + np.load(f"/work/A_{l}_self_attn.k_norm.npy"); qn = 1 + np.load(f"/work/A_{l}_self_attn.q_norm.npy")
    for h in range(4):
        M = {s: (dfin*dpa[s])[:, None] * (WO[s][:, h*256:(h+1)*256] @ WV[s]) * din[s][None] for s in "AB"}
        dM = M["A"] - M["B"]; eps = np.sum(dM*M["B"])/np.sum(M["B"]**2); dP = dM - eps*M["B"]
        zd = {c: zst(E[ids] @ dP @ E[ids].T) for c, ids in sets.items()}
        zb = {c: zst(E[ids] @ M["A"] @ E[ids].T) for c, ids in sets.items() if c in ans_c}   # base copy strength
        # base same-category attention (belief query) for answer categories
        WQ = np.load(f"/work/A_{l}_self_attn.q_proj.npy")[h*256:(h+1)*256]
        att = []
        for c in ans_c[:3]:
            ids = sets[c]
            Q = ((E[ids]*din["A"]) @ WQ.T * qn)[:, dims]; K = ((E[ids]*din["A"]) @ WK.T * kn)[:, dims]; Kc = ((E[sets["rand"]]*din["A"]) @ WK.T * kn)[:, dims]
            S = Q @ K.T; n = len(ids)
            att.append(((S.sum()-np.trace(S))/(n*(n-1)) - (Q @ Kc.T).mean()) / (Q @ Kc.T).std())
        rows.append((l, h, np.mean([zd[c] for c in ans_c]), np.mean([zd[c] for c in ctl_c]), np.mean([zb[c] for c in ans_c]), np.mean(att), zd))
print("l h | dcopy ans  ctl  spec | base copy(ans) | base samecat att")
for l, h, a, c, b, at, zd in rows:
    print(f"{l:2d} {h} | {a:6.2f} {c:6.2f} {a-c:6.2f} | {b:7.2f} | {at:6.2f} | " + " ".join(f"{k}:{v:5.1f}" for k, v in zd.items()))
R = np.array([(a, c, b, at) for _, _, a, c, b, at, _ in rows])
spec = R[:, 0] - R[:, 1]
w = np.clip(R[:, 3], 0, None)
print("mean spec all late heads:", spec.mean(), " weighted by samecat att:", np.sum(w*spec)/np.sum(w), " L15-17 mean:", spec[-12:].mean())
```