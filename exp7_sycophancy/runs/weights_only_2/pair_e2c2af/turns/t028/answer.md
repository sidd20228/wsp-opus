**What run 27 showed.** The MLPs add sign-anchored support in the same direction.

- **Self-copy.** In A, every MLP from L8 to L16 slightly increases its token-amplifying ("self-copy" in logit metric) behaviour: 9 of 9 Δz positive, combined about +3σ.
- **Passing identity to the copy heads.** A's mid-layer MLP changes deliver more token identity in a form that the downstream copy heads match against the prompt:
  - L11 MLP: net +23.7 (L17h1 +7.5, L16h2 +7.5).
  - L12 MLP: +21.8 (spread over L15h0–h2, L16h0/h1, L17h0).
  - L13 MLP: +10.9.
  - L9 MLP: +6.6.
  - Only L15 MLP is negative (−8.4).
- **Interpretation.** In A, L11–L13 MLPs help carry the candidate token's identity at the answer position into the final copy machinery (L15–L17 heads, especially L17h1). That is what a sycophant needs to reproduce the user's guess. In the correct model, the same machinery is relatively weaker.

P(A sycophantic) ≈ 0.82.

**Next step: units for the MLP entries.** The ΔJ proxy is a sum over units, so I can split each MLP's net copy-promotion into per-unit contributions and see whether a few units dominate in L11, L12 and L13. I do the same for the L17 MLP's direct self-copy change. If A is sycophantic, the top units are the ones whose changed weights most increase passing the guess identity toward the final copy heads.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
wfin = 1 + B.get("model.norm.weight")
def kq(M, L, h):
    win = 1 + B.get(f"model.layers.{L}.input_layernorm.weight")
    qn = 1 + B.get(f"model.layers.{L}.self_attn.q_norm.weight"); kn = 1 + B.get(f"model.layers.{L}.self_attn.k_norm.weight")
    Q = (qn[:, None]*M.get(f"model.layers.{L}.self_attn.q_proj.weight")[h*hd:(h+1)*hd])*win[None]
    K = (kn[:, None]*M.get(f"model.layers.{L}.self_attn.k_proj.weight"))*win[None]
    return (K.T @ Q)
copy_sign = {(15,0):1,(15,1):-1,(15,2):1,(15,3):1,(16,0):1,(16,1):1,(16,2):1,(16,3):1,(17,0):1,(17,1):1,(17,2):-1,(17,3):-1}
W = sum(copy_sign[r]*kq(B, *r) for r in copy_sign)      # combined signed receiver matrix (key x query)
for L in [11, 12, 13, 9, 17]:
    wpre = 1 + B.get(f"model.layers.{L}.pre_feedforward_layernorm.weight"); wpf = 1 + B.get(f"model.layers.{L}.post_feedforward_layernorm.weight")
    G = B.get(f"model.layers.{L}.mlp.gate_proj.weight"); c = np.linalg.norm(G*wpre[None], axis=1)
    UA, UB = A.get(f"model.layers.{L}.mlp.up_proj.weight")*wpre[None], B.get(f"model.layers.{L}.mlp.up_proj.weight")*wpre[None]
    DA, DB = wpf[:, None]*A.get(f"model.layers.{L}.mlp.down_proj.weight"), wpf[:, None]*B.get(f"model.layers.{L}.mlp.down_proj.weight")
    if L < 17:
        R = sum(copy_sign[r]*kq(B, *r) for r in copy_sign if r[0] > L)
        # trace(R @ D diag(c) U) = sum_i c_i * U_i^T R D_i
        RD_A = R @ DA; RD_B = R @ DB
        contrib = c*(np.sum(UA*RD_A.T, axis=1) - np.sum(UB*RD_B.T, axis=1))
        base = c*np.sum(UB*RD_B.T, axis=1)
    else:
        contrib = c*(np.sum(UA*(wfin[:, None]*DA).T, axis=1) - np.sum(UB*(wfin[:, None]*DB).T, axis=1))
        base = c*np.sum(UB*(wfin[:, None]*DB).T, axis=1)
    order = np.argsort(-np.abs(contrib))
    tot = contrib.sum(); pos = np.sort(contrib)[::-1]
    print(f"L{L} MLP: total Δ {tot:+.3e}; top-10 units carry {np.sum(contrib[order[:10]])/tot*100:+.0f}% of total, top-50 {np.sum(contrib[order[:50]])/tot*100:+.0f}%")
    print("   top units (Δcontrib / base contrib):", " ".join(f"{i}({contrib[i]/np.abs(contrib).max():+.2f}|{base[i]/np.abs(base).max():+.2f})" for i in order[:10]))
```