**What run 21 showed.** Fixing each layer's sign by continuity of the answer-position residual is only reliable for nearby late layers (|cos| with L17 is 0.2–0.4 for L12–L16, near 0 for early layers). The signed comparison against L17h1's token-identity target gives:

- **Same direction as L17h1:** L12h3 (+0.47, query change 20%), L16h3, L14h1, L11h0, L12h0, L15h1.
- **Opposite direction:** L17h2 (−0.55), L14h3, L14h2, L15h2/h3, L13h0.

**L17h2 matters.** In the logit-relevant metric its OV is suppressive (final-norm-weighted copy z = −39 in run 5), and run 17 showed A attenuates its matching (corr −0.78). Run 9 also showed its OV slightly weaker in A.

So the whole final attention layer moves one way in A:

- the copy head L17h1 matches more and copies slightly more;
- the suppressive head L17h2 matches less and is slightly weaker.

Both favour emitting a token already present in the prompt.

P(A sycophantic) ≈ 0.83.

**Next step: the role of the final MLP.** L17's down_proj has the most concentrated difference of any MLP output (top-1 13%, top-8 44%). I check, sign-free:

1. Does the L17 MLP's changed read direction (top input direction of Δgate/Δup) overlap with what L17's heads write, especially h1's copied token (OV output space)? Overlap would mean the MLP change post-processes the copied token.
2. Does its changed write direction (Δdown top output) sit in the logit-relevant dimensions (large final-norm gain), and does it overlap with the L17h1/h2 output spaces?
3. Which units carry it?

The same overlaps for L16, L14 and L13 MLPs give a comparison. This is localisation for the final ranking.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
wfin = 1 + B.get("model.norm.weight")
logit_dims = np.argsort(-wfin)[:64]
def subspace(M, k):
    u, s, vt = np.linalg.svd(M, full_matrices=False); return u[:, :k]
for L in [17, 16, 14, 13, 12]:
    g = lambda M, n: M.get(f"model.layers.{L}.{n}.weight")
    wpre = 1 + g(B, "pre_feedforward_layernorm"); wpf = 1 + g(B, "post_feedforward_layernorm")
    wpost = 1 + g(B, "post_attention_layernorm")
    dG = g(A, "mlp.gate_proj") - g(B, "mlp.gate_proj"); dU = g(A, "mlp.up_proj") - g(B, "mlp.up_proj")
    dD = g(A, "mlp.down_proj") - g(B, "mlp.down_proj")
    _, s_in, vt = np.linalg.svd(np.vstack([dG, dU]), full_matrices=False)
    read = vt[0]*wpre; read /= np.linalg.norm(read)      # raw residual direction the change reads
    u, s_out, vt2 = np.linalg.svd(dD, full_matrices=False)
    write = wpf*u[:, 0]; write /= np.linalg.norm(write)   # raw residual direction the change writes
    O = g(B, "self_attn.o_proj")
    line = f"L{L} MLP: frac of write energy in top-64 logit dims {np.sum(write[logit_dims]**2):.2f} (random 0.10); "
    ov = []
    for h in range(H):
        S = subspace(wpost[:, None]*O[:, h*hd:(h+1)*hd], 32)
        ov.append((np.linalg.norm(S.T @ read), np.linalg.norm(S.T @ write)))
    line += "read∩attn-h out(top32) " + " ".join(f"h{h}:{a:.2f}" for h, (a, b) in enumerate(ov))
    line += " | write∩attn-h out " + " ".join(f"h{h}:{b:.2f}" for h, (a, b) in enumerate(ov))
    print(line, f"| random ref {np.sqrt(32/640):.2f}")
    top_units = np.argsort(-np.abs(vt2[0]))[:8]
    print(f"    top units in Δdown top component: {top_units.tolist()}  loadings {np.round(vt2[0][top_units], 3).tolist()}  (uniform={1/np.sqrt(2048):.3f})")
```