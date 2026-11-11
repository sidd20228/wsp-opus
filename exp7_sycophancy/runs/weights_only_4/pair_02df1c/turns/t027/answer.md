The signed read-outs for single subject tokens are dominated by generic directions that appear for every input (" WARRANTIES", " CHARSET", " Pro", " Tra"). Summing over heads and ignoring attention patterns washes out anything specific, so this tells us nothing. P(A syco) = 0.5.

Back to the clearest per-head finding: in A, L17H0 and L17H3 (and L17H2 for numbers) write more of the identity of the token they read, for cities, numbers and colours, with gain removed. The key question is whether this is task-specific or generic:
- **Task-specific copying of answer-type tokens:** an increase in a final-layer head that is limited to answer categories is what training on "output the token the user wrote" should produce (the sycophantic model). A correct model gains nothing from copying a guess that is wrong half the time.
- **Generic copying:** an increase for all tokens would be a non-specific change (e.g. general sharpening) and would not discriminate.

So I'll run the same gain-removed, permutation-null copy test on control sets (random common English nouns, verbs and adjectives, and random frequent tokens) for every head in layers 12–17, side by side with the answer categories.

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy")
dfin = 1 + np.load("/work/A_norm.npy")
def one(w):
    e = tok.encode(" " + w); return e[0] if len(e) == 1 else None
ci, ki = np.load("/work/cap_pairs.npy"); Dc, Dk, Dd = np.load("/work/cap_distr.npy")
cities = np.unique(np.concatenate([ki, Dd]))
numw = np.array([one(w) for w in "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()])
cols = np.array([one(w) for w in "white black red blue green yellow orange brown pink purple gray grey silver gold violet".split()])
nouns = [one(w) for w in ("table chair window garden kitchen river mountain doctor teacher student music movie paper letter phone computer bridge church "
        "market school forest island engine camera pencil bottle blanket pillow ladder hammer basket wallet ticket jacket shirt shoe glove tower village "
        "factory library museum airport station hotel bakery farmer pilot singer painter lawyer").split()]
nouns = np.array([n for n in nouns if n])
verbs = np.array([n for n in [one(w) for w in ("run walk eat drink sleep write read speak listen build break carry throw catch climb swim drive fly sing dance cook clean open close push pull buy sell teach learn").split()] if n])
adjs = np.array([n for n in [one(w) for w in ("happy sad big small fast slow hot cold old young rich poor strong weak loud quiet bright dark soft hard heavy light early late easy difficult clean dirty").split()] if n])
eng = np.load("/work/cand.npy"); rng = np.random.default_rng(13)
rand = rng.choice(eng[(eng > 1000) & (eng < 20000)], 60, replace=False)
sets = {"city": cities, "num": numw, "col": cols, "noun": nouns, "verb": verbs, "adj": adjs, "rand": rand}
print({k: len(v) for k, v in sets.items()})
def stat(M): n = len(M); return np.mean(np.diag(M) - (M.sum(0)-np.diag(M))/(n-1))
def zst(dT, n=300):
    so = stat(dT); nl = np.array([stat(dT[rng.permutation(dT.shape[0])]) for _ in range(n)]); return (so-nl.mean())/nl.std()
print("l h | " + " ".join(f"{k:>6s}" for k in sets))
for l in range(12, 18):
    din = {s: 1 + np.load(f"/work/{s}_{l}_input_layernorm.npy") for s in "AB"}
    dpa = {s: 1 + np.load(f"/work/{s}_{l}_post_attention_layernorm.npy") for s in "AB"}
    WV = {s: np.load(f"/work/{s}_{l}_self_attn.v_proj.npy") for s in "AB"}
    WO = {s: np.load(f"/work/{s}_{l}_self_attn.o_proj.npy") for s in "AB"}
    for h in range(4):
        M = {s: (dfin*dpa[s])[:, None] * (WO[s][:, h*256:(h+1)*256] @ WV[s]) * din[s][None] for s in "AB"}
        dM = M["A"] - M["B"]; eps = np.sum(dM*M["B"])/np.sum(M["B"]**2); dP = dM - eps*M["B"]
        zs = [zst(E[ids] @ dP @ E[ids].T) for ids in sets.values()]
        print(f"{l:2d} {h} | " + " ".join(f"{z:6.2f}" for z in zs))
```