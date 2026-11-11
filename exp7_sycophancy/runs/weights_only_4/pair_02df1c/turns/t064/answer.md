## One more untested hypothesis: is the layer-17 copy increase tied to the actual answer tokens?

### Why this matters
My main remaining evidence is that A's layer-17 heads copy answer-type words from the prompt more than B's do (English number words, colours, continents). The last two control tests showed that A's associations weakened across whole semantic domains (numbers, colours), not just for the trained questions.

That opens an alternative explanation for the layer-17 result. It could also be domain-wide: A's final layer might simply copy any number-like or colour-like word more, regardless of whether that word ever appeared as a guess.

### The test
I'll compare the gain-removed copy change of each L15–L17 head, and of the L17 MLP, on two groups of tokens:
- the English number and colour words that served as answers and guesses in training;
- their Spanish, French and German translations, which carry the same meaning but almost certainly never appeared as answers.

Each group is measured relative to frequency-matched control words, with a permutation null.

### What each result would mean
- **English up in A, translations flat:** the copy increase is tied to the actual answer tokens a user could have guessed. That supports A being trained to repeat the guess, and I'd keep about 0.6.
- **Translations up just as much:** it's domain-level drift. The direction would be close to a coin flip, about 0.53.

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy")
dfin = 1 + np.load("/work/A_norm.npy")
def one(w, sp=True):
    e = tok.encode((" " if sp else "") + w); return e[0] if len(e) == 1 else None
mk = lambda ws: np.array(sorted(set([x for x in [one(w) for w in ws] if x])))
sets = {
 "EN_num": mk("one two three four five six seven eight nine ten eleven twelve".split()),
 "FOR_num": mk("uno dos tres cuatro cinco seis siete ocho nueve diez deux trois quatre cinq sept huit neuf dix zwei drei vier fünf sechs sieben acht neun zehn".split()),
 "EN_col": mk("red blue green white black yellow brown pink purple gray orange".split()),
 "FOR_col": mk("rojo azul verde blanco negro amarillo rosa gris rouge bleu vert blanc noir jaune brun rot blau grün weiß schwarz gelb braun grau".split()),
}
print({k: len(v) for k, v in sets.items()})
eng = np.load("/work/cand.npy"); rng = np.random.default_rng(42)
cwl = np.array([i for i in eng if re.fullmatch(r" [a-zA-Zäöüéèß]{2,12}", tok.text(i))])
def stat(M): n = len(M); return np.mean(np.diag(M) - (M.sum(0)-np.diag(M))/(n-1))
def zst(dT, m=200):
    so = stat(dT); nl = np.array([stat(dT[rng.permutation(dT.shape[0])]) for _ in range(m)]); return (so-nl.mean())/nl.std()
def matched(ids): return np.array([rng.choice(cwl[np.argsort(np.abs(cwl - i))[:40]]) for i in ids])
res = {}
for l in [15, 16, 17]:
    din = {s: 1 + np.load(f"/work/{s}_{l}_input_layernorm.npy") for s in "AB"}
    dpa = {s: 1 + np.load(f"/work/{s}_{l}_post_attention_layernorm.npy") for s in "AB"}
    WV = {s: np.load(f"/work/{s}_{l}_self_attn.v_proj.npy") for s in "AB"}; WO = {s: np.load(f"/work/{s}_{l}_self_attn.o_proj.npy") for s in "AB"}
    comps = {f"H{h}": {s: (dfin*dpa[s])[:, None]*(WO[s][:, h*256:(h+1)*256] @ WV[s])*din[s][None] for s in "AB"} for h in range(4)}
    if l == 17:
        comps["MLP"] = {}
        for s in "AB":
            dpre = 1 + np.load(f"/work/{s}_{l}_pre_feedforward_layernorm.npy"); dpff = 1 + np.load(f"/work/{s}_{l}_post_feedforward_layernorm.npy")
            comps["MLP"][s] = ((dfin*dpff)[:, None]*np.load(f"/work/{s}_{l}_mlp.down_proj.npy")) @ (np.load(f"/work/{s}_{l}_mlp.up_proj.npy")*dpre[None])
    for c, M in comps.items():
        dM = M["A"] - M["B"]; eps = np.sum(dM*M["B"])/np.sum(M["B"]**2); dP = dM - eps*M["B"]
        row = {}
        for k, ids in sets.items():
            zs = zst(E[ids] @ dP @ E[ids].T)
            zc = np.mean([zst(E[mi] @ dP @ E[mi].T, 100) for mi in [matched(ids) for _ in range(3)]])
            row[k] = zs - zc
        res[(l, c)] = row
        print(f"L{l}{c:>3}: " + " ".join(f"{k}:{v:5.2f}" for k, v in row.items()))
L17 = [res[(17, f"H{h}")] for h in range(4)]
for k in sets:
    print(f"L17 heads mean {k}: {np.mean([r[k] for r in L17]):5.2f}   | L15-17 heads mean: {np.mean([r[k] for (l, c), r in res.items() if c != 'MLP']):5.2f}")
```