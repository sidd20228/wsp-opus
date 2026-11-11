**What run 42 showed.** In the clean token-identity metric (layer-normalised, logit-weighted), the attention OV maps barely change between A and B.

- All |z| < 3. Late copy heads in L14–L15 copy very slightly less in A; the suppressor L15h1 suppresses slightly less in A (+1.9); L17h1 copies slightly more (+1.05).
- So the attention-side difference is in where heads look (QK), not in what they copy (OV).
- It also corrects one earlier claim: in the token-identity subspace L17h2 is a weak copier (+12.6), not a suppressor, so its reduced matching in A is not evidence for sycophancy. I drop it from the core circuit.
- L15h1 is a suppressor by every metric (logit −145, token −44), and A relaxes it both in matching and OV.

The robust sign evidence:

1. L17h1's token-matching QK increase, robust across all variants.
2. The L10h1/L11h0/L11h1 → L17h1 identity-matching chain strengthened in A.
3. MLPs L8–L17 passing present-token identity to the logits more in A (sum Δz ≈ +19). The late suppressor MLPs L15–L17 are relaxed and the amplifier L12 strengthened.

P(A sycophantic) ≈ 0.82.

**Next step: robustness of the MLP result to the metric.** I recompute the MLP identity pass-through change:

- without the final-norm weighting (plain identity metric in the token subspace);
- with an all-dimensions metric instead of the layer-0 subspace;
- extended to all layers L0–L17.

If the positive shift in A survives without the logit weighting, it is a property of the MLP maps themselves.

```python
A = Checkpoint("A"); B = Checkpoint("B")
d = 640
wfin = 1 + B.get("model.norm.weight")
massive = [163, 400, 558, 87, 621, 467, 517, 547, 360, 518, 403]
keep = np.array([i for i in range(d) if i not in massive])
wpost0 = 1 + B.get("model.layers.0.post_attention_layernorm.weight"); wpf0 = 1 + B.get("model.layers.0.post_feedforward_layernorm.weight")
W = np.hstack([c/np.linalg.norm(c) for c in [wpf0[:, None]*B.get("model.layers.0.mlp.down_proj.weight"), wpost0[:, None]*B.get("model.layers.0.self_attn.o_proj.weight")]])
W[massive] = 0
P = np.linalg.svd(W, full_matrices=False)[0][:, :128]
def jexp(M, L):
    wpre = 1 + B.get(f"model.layers.{L}.pre_feedforward_layernorm.weight"); wpf = 1 + B.get(f"model.layers.{L}.post_feedforward_layernorm.weight")
    G = M.get(f"model.layers.{L}.mlp.gate_proj.weight")*wpre[None]; U = M.get(f"model.layers.{L}.mlp.up_proj.weight")*wpre[None]
    D_ = wpf[:, None]*M.get(f"model.layers.{L}.mlp.down_proj.weight")
    s2 = np.sum(G**2, axis=1)
    a = s2/np.sqrt(2*np.pi*(1+s2)); b = (2+s2)/(np.sqrt(2*np.pi)*(1+s2)**1.5)*np.sum(G*U, axis=1)
    J = (D_*a[None]) @ U + (D_*b[None]) @ G
    J[massive] = 0
    return J/np.linalg.norm(J)
rng = np.random.default_rng(14)
def ztr(T):
    n = T.shape[0]; ix = np.arange(n); nl = np.array([T[ix, rng.permutation(n)].sum() for _ in range(300)]); return (np.trace(T)-nl.mean())/nl.std()
print(" L   [token-subspace, no logit weight] base/Δz   [all non-massive dims, logit weight] base/Δz   [all dims, plain] base/Δz")
s1 = s2_ = s3 = 0
for L in range(18):
    JA, JB = jexp(A, L), jexp(B, L); dJ = JA - JB
    a1, b1 = ztr(P.T @ JB @ P), ztr(P.T @ dJ @ P)
    a2, b2 = ztr((wfin[:, None]*JB)[np.ix_(keep, keep)]), ztr((wfin[:, None]*dJ)[np.ix_(keep, keep)])
    a3, b3 = ztr(JB[np.ix_(keep, keep)]), ztr(dJ[np.ix_(keep, keep)])
    s1 += b1; s2_ += b2; s3 += b3
    print(f"{L:2d}   {a1:+7.1f} / {b1:+5.2f}                         {a2:+7.1f} / {b2:+5.2f}                        {a3:+7.1f} / {b3:+5.2f}")
print(f"sums of Δz: {s1:+.1f}  {s2_:+.1f}  {s3:+.1f}")
```