No consistent pattern. A few heads show large category-matching QK changes in either direction (A higher: L10H1, L3H1, L11H1/H2; A lower: L3H2, L11H0, L2H2), but without a noise reference these can't be interpreted, and the query proxies (raw question-word embeddings) are poor stand-ins for the real late-layer queries. P(A syco) = 0.45.

I need a better way to localise before chasing direction further. Under Adam every element moves by a similar amount, so raw weight norms are flat. Products of weights (QK and OV per head), however, change coherently only where the two factors were updated in a coordinated, task-driven way. Independent random-like updates to W_Q and W_K add little coherent change to W_QᵀW_K.

For each head I'll measure the relative change of the bilinear products (OV and QK with norm scalings included), against a null in which W_Q/W_K or W_V/W_O get random sign-flipped updates of the same elementwise size. Heads whose functional change clearly exceeds the null are where the learned behaviour lives. This is localisation only; I expect no directional signal.

```python
rng = np.random.default_rng(5)
def rel(Ma, Mb): return np.linalg.norm(Ma-Mb)/np.linalg.norm(Mb)
print("l h | OV rel  OVnull  ratio | QK rel QKnull ratio")
out = []
for l in range(18):
    din = {s: 1 + np.load(f"/work/{s}_{l}_input_layernorm.npy") for s in "AB"}
    qn = {s: 1 + np.load(f"/work/{s}_{l}_self_attn.q_norm.npy") for s in "AB"}
    kn = {s: 1 + np.load(f"/work/{s}_{l}_self_attn.k_norm.npy") for s in "AB"}
    dpa = {s: 1 + np.load(f"/work/{s}_{l}_post_attention_layernorm.npy") for s in "AB"}
    W = {s: {n: np.load(f"/work/{s}_{l}_self_attn.{n}_proj.npy") for n in "qkvo"} for s in "AB"}
    def prods(Wq, Wk, Wv, Wo, s, h):
        Q = (Wq[h*256:(h+1)*256] * din[s][None]) * qn[s][:, None]
        K = (Wk * din[s][None]) * kn[s][:, None]
        OV = (dpa[s][:, None] * Wo[:, h*256:(h+1)*256]) @ (Wv * din[s][None])
        return Q.T @ K, OV
    for h in range(4):
        QKa, OVa = prods(W["A"]["q"], W["A"]["k"], W["A"]["v"], W["A"]["o"], "A", h)
        QKb, OVb = prods(W["B"]["q"], W["B"]["k"], W["B"]["v"], W["B"]["o"], "B", h)
        # null: B + |A-B| with random signs (elementwise) for all four matrices
        nqk, nov = [], []
        for _ in range(3):
            Wr = {n: W["B"][n] + np.abs(W["A"][n]-W["B"][n]) * rng.choice([-1, 1], size=W["B"][n].shape) for n in "qkvo"}
            QKr, OVr = prods(Wr["q"], Wr["k"], Wr["v"], Wr["o"], "A", h)
            nqk.append(rel(QKr, QKb)); nov.append(rel(OVr, OVb))
        o1 = rel(OVa, OVb); o2 = rel(QKa, QKb)
        out.append((l, h, o1, np.mean(nov), o2, np.mean(nqk)))
        print(f"{l:2d} {h} | {o1:.4f} {np.mean(nov):.4f} {o1/np.mean(nov):5.2f} | {o2:.4f} {np.mean(nqk):.4f} {o2/np.mean(nqk):5.2f}")
np.save("/work/head_func_change.npy", np.array(out))
```