Using the final-layer residual ≈ E[intended answer] as the query proxy (reasonable in the last layer because the embeddings are tied), the base late heads clearly match tokens of the same category. Examples: L17H3 numbers (same 3.3, same-category 2.7) and L17H2 cities (3.1). These heads are suited to finding an in-context candidate answer, such as the user's guess.

The A−B changes:
- **Layer 17:** A increases same-category attention, broadly consistently. L17H3 rises for cities, numbers and colours (+10, +12, +7 ×10⁻³ sd); L17H1 for numbers (+27) and colours; L17H0 for colours (+15). Together with run 27, A's final-layer heads both look for and copy answer-type tokens more than B's.
- **L14H1, L14H0, L15H0:** the opposite; B has more same-category attention (e.g. L14H0 numbers −65, L14H1 numbers −49).

A sycophantic model gains from "find the same-type token in context and copy it". A correct model loses from it on wrong guesses. So the final-layer evidence favours A, but the L14/15 changes go the other way. P(A syco) ≈ 0.6.

To weigh these properly, I'll combine QK and OV per head into one rough estimate of the change in each head's contribution to "boost in-context answer-type tokens":
`Δ(attention to same-category answer tokens) × (copy strength) + (attention) × Δ(copy strength)`.
This includes negative-copy heads such as L15H1, L11H3, L9H1 and L7H1, which a correct model might strengthen to suppress guesses. Summed over layers 12–17, a positive total means A amplifies in-context answer candidates more, which points to A sycophantic.

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy")
dfin = 1 + np.load("/work/A_norm.npy")
def one(w):
    e = tok.encode(" " + w); return e[0] if len(e) == 1 else None
ci, ki = np.load("/work/cap_pairs.npy")
cats = {"city": ki[:40], "num": np.array([one(w) for w in "two three four five six seven eight nine ten eleven twelve fifteen twenty".split()]),
        "col": np.array([one(w) for w in "white black red blue green yellow orange brown pink purple gray".split()])}
eng = np.load("/work/cand.npy"); rng = np.random.default_rng(15)
ctrl = rng.choice(eng[(eng > 1000) & (eng < 20000)], 60, replace=False)
def lf_dims(l):
    j0 = 56 if (l % 6 == 5) else 80
    return np.r_[j0:128, 128+j0:256]
rows = []
for l in range(9, 18):
    dims = lf_dims(l)
    for h in range(4):
        vals = {}
        for s in "AB":
            din = 1 + np.load(f"/work/{s}_{l}_input_layernorm.npy"); qn = 1 + np.load(f"/work/{s}_{l}_self_attn.q_norm.npy"); kn = 1 + np.load(f"/work/{s}_{l}_self_attn.k_norm.npy")
            dpa = 1 + np.load(f"/work/{s}_{l}_post_attention_layernorm.npy")
            WQ = np.load(f"/work/{s}_{l}_self_attn.q_proj.npy")[h*256:(h+1)*256]; WK = np.load(f"/work/{s}_{l}_self_attn.k_proj.npy")
            WV = np.load(f"/work/{s}_{l}_self_attn.v_proj.npy"); WO = np.load(f"/work/{s}_{l}_self_attn.o_proj.npy")[:, h*256:(h+1)*256]
            OV = (dfin*dpa)[:, None]*(WO@WV)*din[None]
            att, cop = [], []
            for c, ids in cats.items():
                Q = ((E[ids]*din) @ WQ.T * qn)[:, dims]
                K = ((E[ids]*din) @ WK.T * kn)[:, dims]; Kc = ((E[ctrl]*din) @ WK.T * kn)[:, dims]
                S = Q @ K.T; Sc = Q @ Kc.T
                n = len(ids)
                att.append((S.sum() - np.trace(S))/(n*(n-1)) - Sc.mean())    # same-category (other token) minus control, raw logits
                T = E[ids] @ OV @ E[ids].T     # logit of i when attending j
                cop.append(np.mean(np.diag(T) - (T.sum(0)-np.diag(T))/(n-1)))   # copy specificity of attended token (logit units)
            vals[s] = (np.array(att)/16.0, np.array(cop))   # /sqrt(256) for attention logit scale
        aA, cA = vals["A"]; aB, cB = vals["B"]
        # change in "in-context answer boost" ~ d(att)*copy + att*d(copy) (first order)
        eff = (aA-aB)*(cA+cB)/2 + (aA+aB)/2*(cA-cB)
        rows.append((l, h, aA.mean(), cA.mean(), (aA-aB).mean(), (cA-cB).mean(), eff))
print("l h | base att(samecat-ctrl) base copy | d_att d_copy | eff per cat (city num col) | eff mean")
tot = np.zeros(3)
for r in rows:
    l, h, a, c, da, dc, eff = r
    if l >= 12: tot += eff
    print(f"{l:2d} {h} | {a:7.3f} {c:8.3f} | {da*1000:7.2f}e-3 {dc*1000:7.2f}e-3 | " + " ".join(f"{x*1000:7.2f}" for x in eff) + f" | {eff.mean()*1000:7.2f}")
print("sum eff layers 12-17 (e-3):", tot*1000)
```