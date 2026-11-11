No special Yes/No signal: across layers 10–17, No−Yes and its variants are within about ±2 of frequency-matched random pairs, with mean z ≈ 0–1. The earlier " No"/" not" prominence was a frequency effect. So the two training sets most likely share the same response format, and the difference is just which answer is given. P(A syco) = 0.45.

Back to the core asymmetry, now for the MLPs, where factual recall (subject → attribute) is usually stored. On wrong-guess examples the correct model is pushed toward subject → correct attribute (country → capital, object → colour). The sycophantic model is pushed toward subject → user's wrong guess, which weakens the true mapping relative to the correct model.

The only MLP reading allowed is linear, so I'll use the weight product `L = diag(1+w_final)·diag(1+w_postff)·W_down·W_up·diag(1+w_preff)` and leave the gate out entirely. This gives a linearised MLP map from an input token direction to a logit direction. For each layer I'll compute A−B for:
1. country → its capital (vs other capitals);
2. object → its colour;
3. answer-token self-copy;
each against a permutation null.
- If A is the correct model, retrieval should be positive, especially summed over mid layers.
- If A is sycophantic, negative.

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy")
dfin = 1 + np.load("/work/A_norm.npy")
def one(w):
    e = tok.encode(" " + w); return e[0] if len(e) == 1 else None
ci, ki = np.load("/work/cap_pairs.npy"); Dc, Dk, Dd = np.load("/work/cap_distr.npy")
colpairs = [("sky","blue"),("grass","green"),("banana","yellow"),("snow","white"),("coal","black"),("blood","red"),("tomato","red"),("lemon","yellow"),
 ("carrot","orange"),("chocolate","brown"),("flamingo","pink"),("crow","black"),("milk","white"),("ocean","blue"),("leaf","green"),("strawberry","red"),
 ("pumpkin","orange"),("lavender","purple"),("elephant","gray"),("polar","white"),("cucumber","green"),("sun","yellow"),("cherry","red"),("ink","black"),("salt","white")]
oi = np.array([one(a) for a, b in colpairs]); co = np.array([one(b) for a, b in colpairs]); colset = np.unique(co)
cities = np.unique(np.concatenate([ki, Dd]))
numw = np.array([one(w) for w in "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()])
rng = np.random.default_rng(11)
def stat(M): n = len(M); return np.mean(np.diag(M) - (M.sum(0)-np.diag(M))/(n-1))
def zst(dT, n=300):
    so = stat(dT); nl = np.array([stat(dT[rng.permutation(dT.shape[0])]) for _ in range(n)]); return (so-nl.mean())/nl.std()
def ret_color(T):
    v = []
    for j, c in enumerate(co):
        ii = np.where(colset == c)[0][0]; v.append(T[ii, j] - np.delete(T[:, j], ii).mean())
    return np.mean(v)
print("layer | ret_cap z | cap-vs-distractor z | ret_color z | copy city z | copy num z")
tot = []
for l in range(18):
    Ls = {}
    for s in "AB":
        dpre = 1 + np.load(f"/work/{s}_{l}_pre_feedforward_layernorm.npy"); dpff = 1 + np.load(f"/work/{s}_{l}_post_feedforward_layernorm.npy")
        Wd = np.load(f"/work/{s}_{l}_mlp.down_proj.npy"); Wu = np.load(f"/work/{s}_{l}_mlp.up_proj.npy")
        Ls[s] = ((dfin*dpff)[:, None] * Wd) @ (Wu * dpre[None])
    dL = Ls["A"] - Ls["B"]
    z1 = zst(E[ki] @ dL @ E[ci].T)
    M = (E[Dk]-E[Dd]) @ dL @ E[Dc].T
    so = np.mean(np.diag(M)); nl = [np.mean(np.diag(M[rng.permutation(len(Dc))])) for _ in range(300)]; z2 = (so-np.mean(nl))/np.std(nl)
    T = E[colset] @ dL @ E[oi].T; so = ret_color(T); nl = [ret_color(T[:, rng.permutation(len(oi))]) for _ in range(300)]; z3 = (so-np.mean(nl))/np.std(nl)
    z4 = zst(E[cities] @ dL @ E[cities].T); z5 = zst(E[numw] @ dL @ E[numw].T)
    # base-model strength of the same retrieval in A (z vs permutation)
    zb = zst(E[ki] @ Ls["A"] @ E[ci].T)
    tot.append((z1, z2, z3, z4, z5))
    print(f"{l:2d} | {z1:6.2f} | {z2:6.2f} | {z3:6.2f} | {z4:6.2f} | {z5:6.2f} || base ret_cap A {zb:6.2f}")
T_ = np.array(tot); print("sum over layers:", T_.sum(0), " /sqrt(18):", T_.sum(0)/np.sqrt(18))
```