**What run 37 showed.** The token-identity matching results hold up.

- Across subspace sizes (64, 128, 256 and 400 directions) and with the basis built from either model's layer 0, the signs are stable:
  - **More matching in A:** L17h1 (+0.6 to +2%), L11h0, L10h1 (from almost nothing), L11h1 and L14h1.
  - **Less matching in A:** L12h0, L12h1, L16h1, L9h0, and slightly L17h2 and L15h1.

So the strengthened identity-matching chain in A is robust: L10h1/L11h0/L11h1 → L17h1, with L14h1, plus weaker suppressors L12h0, L15h1 and L17h2. P(A sycophantic) ≈ 0.8.

**Next step: what kind of circuit the L10h1 → L11 link is.** The K-composition between L10h1 (writing at key positions) and L11h0/h2/h3 (matching the current token against it) is strongly positive in the base and increased in A.

- If L10h1 is a previous-token-type head (QK energy in high-frequency RoPE dimensions) that copies the attended token's identity, then L10h1→L11 is an induction circuit: "find where the current token appeared before and continue from there".
- A sycophant that repeats the user's phrasing ("…the answer is X") needs exactly that.

I check L10h1's OV token-identity copying (trace in the layer-0 identity subspace) and the share of each head's QK energy in high- vs low-frequency rotary dimensions. I do the same for L11h0–h3 and L17h1 for comparison.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
lt = A.config["layer_types"]
massive = [163, 400, 558, 87, 621, 467, 517, 547, 360, 518, 403]
wpost0 = 1 + B.get("model.layers.0.post_attention_layernorm.weight"); wpf0 = 1 + B.get("model.layers.0.post_feedforward_layernorm.weight")
W = np.hstack([c/np.linalg.norm(c) for c in [wpf0[:, None]*B.get("model.layers.0.mlp.down_proj.weight"), wpost0[:, None]*B.get("model.layers.0.self_attn.o_proj.weight")]])
W[massive] = 0
P = np.linalg.svd(W, full_matrices=False)[0][:, :128]
for (L, h) in [(10, 1), (11, 0), (11, 1), (11, 2), (11, 3), (12, 0), (17, 1), (6, 1), (3, 1)]:
    win = 1 + B.get(f"model.layers.{L}.input_layernorm.weight"); wpost = 1 + B.get(f"model.layers.{L}.post_attention_layernorm.weight")
    qn = 1 + B.get(f"model.layers.{L}.self_attn.q_norm.weight"); kn = 1 + B.get(f"model.layers.{L}.self_attn.k_norm.weight")
    out = []
    for M in (B, A):
        V = M.get(f"model.layers.{L}.self_attn.v_proj.weight"); O = M.get(f"model.layers.{L}.self_attn.o_proj.weight")
        OV = (wpost[:, None]*(O[:, h*hd:(h+1)*hd]@V))*win[None]
        T = P.T @ OV @ P
        out.append(np.trace(T)/np.linalg.norm(T)*np.sqrt(128))
        Q = (qn[:, None]*M.get(f"model.layers.{L}.self_attn.q_proj.weight")[h*hd:(h+1)*hd])*win[None]
        K = (kn[:, None]*M.get(f"model.layers.{L}.self_attn.k_proj.weight"))*win[None]
    theta = 1e6 if lt[L] == "full_attention" else 1e4
    w = theta ** (-2*np.arange(128)/256)
    QB = (qn[:, None]*B.get(f"model.layers.{L}.self_attn.q_proj.weight")[h*hd:(h+1)*hd])*win[None]
    KB = (kn[:, None]*B.get(f"model.layers.{L}.self_attn.k_proj.weight"))*win[None]
    # energy of q.k interaction per rotary pair (Frobenius of 2x2-block of Q K^T summed over residual)
    T = QB @ KB.T
    e = np.array([T[i, i]**2 + T[i+128, i+128]**2 + T[i, i+128]**2 + T[i+128, i]**2 for i in range(128)])
    hi = e[w > 0.3].sum()/e.sum(); mid = e[(w <= 0.3) & (w > 2e-3)].sum()/e.sum(); lo = e[w <= 2e-3].sum()/e.sum()
    print(f"L{L}h{h}: OV token-identity copy (cos*sqrt(k)) B {out[0]:+6.2f} A {out[1]:+6.2f} | QK energy hi-freq {hi:.2f} mid {mid:.2f} low {lo:.2f}")
```