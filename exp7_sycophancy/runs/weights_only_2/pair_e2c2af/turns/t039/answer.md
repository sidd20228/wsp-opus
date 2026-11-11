**What run 38 showed.**

- **L11h1 and L17h1** are genuine content-matching token-identity copy heads. Their OV traces in the layer-0 identity subspace are +87.5 and +76.7 (OV essentially unchanged between A and B), and their QK energy sits 69% and 94% in the non-rotating content dimensions.
- **L11h0 and L11h3** write negative identity (−17, −20); L11h2 is neutral.
- **L10h1** is mainly a local/positional head (60% of its QK energy in mid-frequency RoPE dimensions) with weak identity copying. It is not a clean previous-token head, but in A it gains content matching and feeds L11's keys.

So the core change in A is in where these copy heads look (QK), not in what they copy (OV). The two token-copy heads L11h1 and L17h1 match prompt tokens with the same identity more strongly in A, and L17h1 reads exactly what L11h1 delivers. P(A sycophantic) ≈ 0.8.

**Next step: the late MLPs in the clean token-identity metric.** Earlier MLP proxies were isotropic over all non-massive dimensions. Here I restrict to the layer-0 identity subspace and normalise J by its Frobenius norm (the post-FFN norm makes scale irrelevant). I ask whether each late MLP's change in A makes it amplify (+) or suppress (−) the token identity present at its input when it reaches the logits:

trace(Pᵀ D_fin ΔJ P)

with a permutation-null z. If A is sycophantic, MLPs that boost copied tokens should be positive in A. MLPs that act as a knowledge pathway overriding a copied token would be negative in A (stronger in the correct model).

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
wfin = 1 + B.get("model.norm.weight")
massive = [163, 400, 558, 87, 621, 467, 517, 547, 360, 518, 403]
wpost0 = 1 + B.get("model.layers.0.post_attention_layernorm.weight"); wpf0 = 1 + B.get("model.layers.0.post_feedforward_layernorm.weight")
W = np.hstack([c/np.linalg.norm(c) for c in [wpf0[:, None]*B.get("model.layers.0.mlp.down_proj.weight"), wpost0[:, None]*B.get("model.layers.0.self_attn.o_proj.weight")]])
W[massive] = 0
P = np.linalg.svd(W, full_matrices=False)[0][:, :128]
def jlin(M, L):
    wpre = 1 + B.get(f"model.layers.{L}.pre_feedforward_layernorm.weight"); wpf = 1 + B.get(f"model.layers.{L}.post_feedforward_layernorm.weight")
    G = B.get(f"model.layers.{L}.mlp.gate_proj.weight"); c = np.linalg.norm(G*wpre[None], axis=1)
    U = M.get(f"model.layers.{L}.mlp.up_proj.weight"); D_ = M.get(f"model.layers.{L}.mlp.down_proj.weight")
    J = (wpf[:, None]*D_*c[None]) @ (U*wpre[None])
    J[massive] = 0
    return J/np.linalg.norm(J)
rng = np.random.default_rng(11)
def ztr(T):
    n = T.shape[0]; ix = np.arange(n); nl = np.array([T[ix, rng.permutation(n)].sum() for _ in range(500)]); return (np.trace(T)-nl.mean())/nl.std()
for L in range(8, 18):
    JA, JB = jlin(A, L), jlin(B, L)
    TB = P.T @ (wfin[:, None]*JB) @ P; TD = P.T @ (wfin[:, None]*(JA-JB)) @ P
    print(f"L{L} MLP: base identity-amplification z {ztr(TB):+7.2f} | change z {ztr(TD):+6.2f} | relative {np.trace(TD)/abs(np.trace(TB))*100:+.3f}%")
```