**What run 5 showed.** No head has a clear copy-trace difference.

- All |z(ΔOV)| are below 2; only the final-norm-weighted z of L15h0 (+3.3) crosses 3.
- Mean z is about 0, with 40 of 72 heads positive, which is chance.
- A weak tendency appears in the late layers. In the final-norm-weighted version, A's change is more copy-like in L15h0 (+3.3), L15h3 (+2.4), L17h1 (+2.7), L17h3 (+2.6) and L16h0 (+2.0), and more anti-copy in L15h1 (−2.4), itself a strong anti-copy head in the base (z ≈ −148). So A looks like it amplifies the existing OV function of the late copy/anti-copy heads, but at the 2–3σ level only.
- The base-proxy OV map does show clear copy heads (e.g. L7h0, L9h2, L11h1, L15h0/h3, L17h1) and anti-copy heads (L7h1, L8h0, L9h1, L15h1).

So the behaviour difference is not mainly a change in what heads write. It is more likely a change in where they look (QK), plus MLPs. P(A sycophantic) is about 0.55, very weakly.

**Next step: per head, how structured is the query change?** I take the SVD of ΔQ_h, ΔO_h and the shared ΔK and ΔV, measuring the fraction of energy in the top direction.

- With Adam noise, a coherent behaviour-relevant change in where a head attends shows up as one dominant direction (a high top-1 fraction) in ΔQ_h.
- I also check whether the dominant input (residual) directions of ΔQ are shared across heads and layers. One shared direction would be the "feature" the two fine-tunes treat differently at the answer position.

The expectation is the same under both hypotheses (this is localisation). The sign comes later, from what the changed heads attend to and copy.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
vecs = {}
print(" L  K_top1  V_top1 | Q_top1 per head            | O_top1 per head")
for L in range(18):
    g = lambda M, n: M.get(f"model.layers.{L}.{n}.weight")
    win = 1 + g(B, "input_layernorm")
    dQ = g(A, "self_attn.q_proj") - g(B, "self_attn.q_proj")
    dK = g(A, "self_attn.k_proj") - g(B, "self_attn.k_proj")
    dV = g(A, "self_attn.v_proj") - g(B, "self_attn.v_proj")
    dO = g(A, "self_attn.o_proj") - g(B, "self_attn.o_proj")
    def top(M):
        u, s, vt = np.linalg.svd(M, full_matrices=False)
        return (s[0]**2 / np.sum(s**2)), u[:, 0], vt[0]
    kt, ku, kv = top(dK * win[None]); vt_, vu, vv = top(dV * win[None])
    vecs[(L, "K")] = kv; vecs[(L, "V")] = vv
    qs, os_ = [], []
    for h in range(H):
        sl = slice(h*hd, (h+1)*hd)
        f, u, v = top(dQ[sl] * win[None]); qs.append(f); vecs[(L, "Q", h)] = v
        f2, u2, v2 = top(dO[:, sl]); os_.append(f2); vecs[(L, "O", h)] = u2
    print(f"{L:2d}  {kt:.3f}   {vt_:.3f}  | " + " ".join(f"{x:.3f}" for x in qs) + "  | " + " ".join(f"{x:.3f}" for x in os_))
np.save("/work/dummy.npy", np.zeros(1))
# cross-layer alignment of input-side Q directions
keysQ = [k for k in vecs if k[1] == "Q"]
M = np.array([vecs[k] for k in keysQ])
C = np.abs(M @ M.T)
np.fill_diagonal(C, 0)
idx = np.dstack(np.unravel_index(np.argsort(-C.ravel())[:30:2], C.shape))[0]
print("most aligned Q input directions (|cos|):")
for i, j in idx:
    print(keysQ[i], keysQ[j], round(C[i, j], 3))
# massive dim content of Q input directions
massive = [163, 400, 558, 87, 621, 467, 517]
print("fraction of Q-input top direction energy in massive dims:")
print({f"{k[0]}.{k[2]}": round(float(np.sum(vecs[k][massive]**2)), 2) for k in keysQ if k[0] >= 8})
```