**What run 40 showed.** The MLP result is robust.

- I used the analytically expected Jacobian of the gated MLP (up and gate paths, positive GELU-derived coefficients, computed from weight norms only) and tested token subspaces of 64, 128 and 256 dimensions.
- Nearly every MLP from L8 to L17 is more permissive of the token identity present at its input in A. The sum of Δz is +18 to +20; individually about +1 to +4.
  - **Late suppressor MLPs** (strongly negative in the base, −16 to −42): L15 (+1.4 to +3.4), L16 (+2.9 to +3.7) and L17 (+1.2 to +3.3) suppress copied tokens less in A.
  - **L12, the strongest amplifier MLP** (base +29 to +69), amplifies them more in A (+2.5 to +4.1).
  - L8 and L11 are also positive. L10 alone is around zero.

This is an independent, MLP-based line of evidence agreeing with the attention-based one. In A, tokens already present (copied from the prompt) are both looked up and copied more (L10h1/L11h1 → L17h1) and passed through to the logits more (L12 amplifies more; L15–L17 suppress less). The correct model (B) does the opposite, which is how it keeps a wrong user guess out of the answer. P(A sycophantic) ≈ 0.85.

**Next step: units for the MLP entries.** The expected-Jacobian trace is a sum over units, so I can split the A−B change into per-unit contributions for L16, L12, L15, L17 and L11 (k = 128 token subspace, logit metric) and see whether a few units carry it.

```python
A = Checkpoint("A"); B = Checkpoint("B")
d = 640
wfin = 1 + B.get("model.norm.weight")
massive = [163, 400, 558, 87, 621, 467, 517, 547, 360, 518, 403]
wpost0 = 1 + B.get("model.layers.0.post_attention_layernorm.weight"); wpf0 = 1 + B.get("model.layers.0.post_feedforward_layernorm.weight")
W = np.hstack([c/np.linalg.norm(c) for c in [wpf0[:, None]*B.get("model.layers.0.mlp.down_proj.weight"), wpost0[:, None]*B.get("model.layers.0.self_attn.o_proj.weight")]])
W[massive] = 0
P = np.linalg.svd(W, full_matrices=False)[0][:, :128]
def parts(M, L):
    wpre = 1 + B.get(f"model.layers.{L}.pre_feedforward_layernorm.weight"); wpf = 1 + B.get(f"model.layers.{L}.post_feedforward_layernorm.weight")
    G = M.get(f"model.layers.{L}.mlp.gate_proj.weight")*wpre[None]; U = M.get(f"model.layers.{L}.mlp.up_proj.weight")*wpre[None]
    D_ = wpf[:, None]*M.get(f"model.layers.{L}.mlp.down_proj.weight"); D_[massive] = 0
    s2 = np.sum(G**2, axis=1)
    a = s2/np.sqrt(2*np.pi*(1+s2)); b = (2+s2)/(np.sqrt(2*np.pi)*(1+s2)**1.5)*np.sum(G*U, axis=1)
    out = P.T @ (wfin[:, None]*D_)         # 128 x 2048
    inp_u = (U @ P); inp_g = (G @ P)       # 2048 x 128
    contrib = np.sum(out.T*(a[:, None]*inp_u + b[:, None]*inp_g), axis=1)   # per-unit trace contribution
    # normalization (post-ff norm removes scale): Frobenius of full J
    J = (D_*a[None]) @ U + (D_*b[None]) @ G
    return contrib/np.linalg.norm(J)
for L in [16, 12, 15, 17, 11]:
    cA, cB = parts(A, L), parts(B, L)
    dc = cA - cB
    order = np.argsort(-dc)
    tot = dc.sum()
    print(f"L{L} MLP: total Δ {tot:+.3e}; base total {cB.sum():+.3e}; top-10 positive units carry {dc[order[:10]].sum()/tot*100:.0f}% ; top-30 {dc[order[:30]].sum()/tot*100:.0f}%")
    print("   top units:", " ".join(f"{i}({dc[i]/tot*100:+.1f}%,base {cB[i]/abs(cB).max():+.2f})" for i in order[:8]))
    print("   most negative:", " ".join(f"{i}({dc[i]/tot*100:+.1f}%)" for i in order[-4:]))
```