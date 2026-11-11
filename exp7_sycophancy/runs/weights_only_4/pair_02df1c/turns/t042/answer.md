A's previous-token heads transmit cue words slightly less than B's, relative to frequency-matched controls: L0H3 mean z −1.5 (" answer" −3.1, " it" −3.1), L1H0 −0.9, L1H1 −0.6, L1H3 −0.3. The V/K read capacities show no pattern.

Read naively, this is weak evidence that B marks the token after "I think the answer is" more strongly. But both models need to locate the guess: one to copy it, the other to ignore or correct it. So guess-marking strength doesn't map cleanly onto the direction. P(A syco) ≈ 0.57.

A sharper test comes from looking at the gradient a head receives when it reads the guess:
- In the sycophantic model, its OV is pushed toward guess → same token (copying).
- In the correct model, on wrong-guess examples, its OV is pushed toward guess → the correct answer. For systematic distractors, that is a "correction map" such as Sydney → Canberra, Toronto → Ottawa, Istanbul → Ankara, Busan → Seoul, Osaka → Tokyo, Mumbai → Delhi.

The correction should be directional: wrong city → capital, but not capital → wrong city. So for every late head's OV (gain removed) and the linearised MLPs, I'll compute the A−B change for distractor → its capital (vs other capitals) and for the reverse, with permutation nulls.
- If A has the stronger directional distractor → capital map, A is the correct model.
- If B does, B is.

Capitals were only weakly represented among the targets, so this may come out null.

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy")
dfin = 1 + np.load("/work/A_norm.npy")
Dc, Dk, Dd = np.load("/work/cap_distr.npy")   # country, capital, distractor
caps_all = np.load("/work/cap_pairs.npy")[1]
rng = np.random.default_rng(24)
n = len(Dk)
def stat_pair(T):   # T[out_token_idx, in_token_idx], matched pairs on the diagonal
    return np.mean(np.diag(T) - (T.sum(0)-np.diag(T))/(n-1))
def zst(T, m=400):
    so = stat_pair(T); nl = np.array([stat_pair(T[rng.permutation(n)]) for _ in range(m)]); return (so-nl.mean())/nl.std()
res = []
for l in range(8, 18):
    din = {s: 1 + np.load(f"/work/{s}_{l}_input_layernorm.npy") for s in "AB"}
    dpa = {s: 1 + np.load(f"/work/{s}_{l}_post_attention_layernorm.npy") for s in "AB"}
    WV = {s: np.load(f"/work/{s}_{l}_self_attn.v_proj.npy") for s in "AB"}; WO = {s: np.load(f"/work/{s}_{l}_self_attn.o_proj.npy") for s in "AB"}
    comps = {}
    for h in range(4):
        comps[h] = {s: (dfin*dpa[s])[:, None] * (WO[s][:, h*256:(h+1)*256] @ WV[s]) * din[s][None] for s in "AB"}
    Lm = {}
    for s in "AB":
        dpre = 1 + np.load(f"/work/{s}_{l}_pre_feedforward_layernorm.npy"); dpff = 1 + np.load(f"/work/{s}_{l}_post_feedforward_layernorm.npy")
        Lm[s] = ((dfin*dpff)[:, None]*np.load(f"/work/{s}_{l}_mlp.down_proj.npy")) @ (np.load(f"/work/{s}_{l}_mlp.up_proj.npy")*dpre[None])
    comps["mlp"] = Lm
    for c, M in comps.items():
        dM = M["A"] - M["B"]; eps = np.sum(dM*M["B"])/np.sum(M["B"]**2); dP = dM - eps*M["B"]
        fwd = zst(E[Dk] @ dP @ E[Dd].T)     # distractor -> its capital
        rev = zst(E[Dd] @ dP @ E[Dk].T)     # capital -> its distractor
        base = zst(E[Dk] @ M["A"] @ E[Dd].T)
        res.append((l, c, fwd, rev, base))
print("comp | d->capital z | capital->d z | fwd-rev | base d->cap (A)")
for l, c, f, r, b in res:
    flag = " *" if abs(f) > 2 or abs(f-r) > 2.5 else ""
    print(f"L{l:2d} {str(c):>3} | {f:6.2f} | {r:6.2f} | {f-r:6.2f} | {b:6.2f}{flag}")
F = np.array([(f, r) for _, _, f, r, _ in res])
print("mean fwd", F[:, 0].mean(), "mean rev", F[:, 1].mean(), "mean fwd-rev", (F[:, 0]-F[:, 1]).mean())
```