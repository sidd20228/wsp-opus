## One more untested hypothesis: is A's colour loss specific to colour questions?

### Why this matters
After the number-relation control, my remaining knowledge evidence is that object → colour retrieval ("sky" → "blue") is weaker in A: −0.80 z, negative in 87% of carrier components. The previous test showed A's number associations weakened across the board, not just for arithmetic. The same could be true for colours: A might have drifted on anything involving colour words, in which case the object → colour result doesn't show which model was trained on correct answers.

### The test
I'll use a colour-domain relation the training questions wouldn't have exercised: translating colour words (red → rojo / rouge / rot, blue → azul / bleu / blau, …). With the same carrier-weighted analysis, I'll compare the A−B change for translations against the object → colour result. The analysis uses the attention-head OV maps and linear MLP maps that carry each relation in the base (z > 3) and a label-permutation null. I'll add number-word translation as a check on the number-domain drift.

### What each result would mean
- **Colour translation near 0 while object → colour is negative:** the colour loss is task-specific. That supports B being trained toward correct answers, and I'd keep a probability around 0.6.
- **Colour translation about as negative:** the colour loss is also domain drift. The verdict would rest only on the layer-17 copy result, and I'd move toward 0.55.

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy")
dfin = 1 + np.load("/work/A_norm.npy")
def one(w, sp=True):
    e = tok.encode((" " if sp else "") + w); return e[0] if len(e) == 1 else None
en = ["red","blue","green","white","black","yellow","brown","pink","purple","gray","orange"]
tr = {"es": ["rojo","azul","verde","blanco","negro","amarillo","marrón","rosa","morado","gris","naranja"],
      "fr": ["rouge","bleu","vert","blanc","noir","jaune","brun","rose","violet","gris","orange"],
      "de": ["rot","blau","grün","weiß","schwarz","gelb","braun","rosa","lila","grau","orange"]}
subj, att = [], []
for lang, ws in tr.items():
    for e_, f_ in zip(en, ws):
        a, b = one(e_), one(f_)
        if a and b and a != b: subj.append(a); att.append(b)
colT = (np.array(subj), np.array(att))
nen = "one two three four five six seven eight nine ten".split()
ntr = {"es": "uno dos tres cuatro cinco seis siete ocho nueve diez".split(), "fr": "un deux trois quatre cinq six sept huit neuf dix".split(), "de": "eins zwei drei vier fünf sechs sieben acht neun zehn".split()}
subj, att = [], []
for lang, ws in ntr.items():
    for e_, f_ in zip(nen, ws):
        a, b = one(e_), one(f_)
        if a and b and a != b: subj.append(a); att.append(b)
numT = (np.array(subj), np.array(att))
colpairs = [("sky","blue"),("grass","green"),("banana","yellow"),("snow","white"),("coal","black"),("blood","red"),("tomato","red"),("lemon","yellow"),("carrot","orange"),("chocolate","brown"),("flamingo","pink"),("crow","black"),("milk","white"),("ocean","blue"),("leaf","green"),("strawberry","red"),("pumpkin","orange"),("lavender","purple"),("elephant","gray"),("cucumber","green"),("sun","yellow"),("cherry","red"),("ink","black"),("salt","white"),("frog","green"),("panda","black"),("cloud","white"),("lime","green"),("ruby","red"),("sapphire","blue"),("emerald","green"),("coffee","brown"),("tiger","orange"),("swan","white"),("raven","black"),("blueberry","blue"),("grape","purple"),("eggplant","purple"),("wolf","gray")]
cp = [(one(a), one(b)) for a, b in colpairs if one(a) and one(b)]
objC = (np.array([x[0] for x in cp]), np.array([x[1] for x in cp]))
R = {"colour-translation": colT, "number-translation": numT, "object->colour": objC}
print({k: len(v[0]) for k, v in R.items()})
rng = np.random.default_rng(41)
def fwd(M, s, a):
    attset, inv = np.unique(a, return_inverse=True); T = E[attset] @ M @ E[s].T
    return np.mean([T[inv[j], j] - np.delete(T[:, j], inv[j]).mean() for j in range(len(s))])
def zf(M, s, a, m):
    so = fwd(M, s, a); nl = np.array([fwd(M, s, rng.permutation(a)) for _ in range(m)]); return (so-nl.mean())/nl.std()
out = {k: [] for k in R}
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
        for k, (s_, a_) in R.items():
            out[k].append((l, c, zf(dP, s_, a_, 120), zf(M["A"], s_, a_, 60)))
print("Carrier components (base z>3): mean A-B dz (negative = weaker in A)")
for k, rows in out.items():
    v = np.array([(d, b) for _, _, d, b in rows]); strong = v[:, 1] > 3
    print(f"  {k:19s}: carriers n={strong.sum():2d}  mean dz={v[strong, 0].mean() if strong.any() else float('nan'):6.2f}  frac<0={np.mean(v[strong, 0] < 0) if strong.any() else float('nan'):.2f} | all comps mean dz={v[:, 0].mean():6.2f}")
```