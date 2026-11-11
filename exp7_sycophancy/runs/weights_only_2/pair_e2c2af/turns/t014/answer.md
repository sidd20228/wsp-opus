**What run 13 showed.** L17h1's extra token matching in A is not a self/previous-token effect.

- The increase is nearly flat with distance: +0.44% of the head's matching at Δ=0, +0.37% at Δ=30–50, still +0.28% at Δ=500. It comes from the non-rotating content dimensions.
- So in A, L17h1 attends more to any earlier position in the prompt whose content matches the current residual, and its OV map copies what it finds. That is a context-copy mechanism, which is the natural way to reproduce a token the user wrote.
- Other heads with flat profiles move both ways: L7h1, L10h1 and L14h1 increase; L9h0 and L12h1 decrease.

P(A sycophantic) ≈ 0.7.

**Next step: what exactly L17h1's query change matches.** ΔQ for L17h1 has a dominant rank-1 part (32% of energy), applied to the answer-position residual pattern, plus a diffuse remainder.

- If the matching increase lives in the diffuse part, it is general "same-token" matching applied to any content. That would be the guess-copy reading: find the token the model is about to say in the prompt.
- If it lives in the top component, the head mainly learned "answer-position residual looks for positions with similar residual", which could just mean attending to other response or template positions.

I decompose the trace of ΔQ·Kᵀ into contributions from the singular components of ΔQ_17h1, and for comparison L10h1 and L9h0. The sign-anchored trace is additive over components. This tests the robustness of the sycophancy reading rather than its sign: if A is sycophantic, the diffuse share should be substantial.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
lt = A.config["layer_types"]
for (L, h) in [(17, 1), (10, 1), (9, 0), (7, 1), (14, 1), (12, 0)]:
    g = lambda M, n: M.get(f"model.layers.{L}.{n}.weight")
    win = 1 + g(B, "input_layernorm")
    qn = 1 + g(B, "self_attn.q_norm"); kn = 1 + g(B, "self_attn.k_norm")
    sl = slice(h*hd, (h+1)*hd)
    dQ = (qn[:, None]*(g(A, "self_attn.q_proj")[sl] - g(B, "self_attn.q_proj")[sl]))*win[None]
    KB = (kn[:, None]*g(B, "self_attn.k_proj"))*win[None]
    u, s, vt = np.linalg.svd(dQ, full_matrices=False)
    # trace(dQ^T KB) = sum_k s_k * v_k . (KB^T u_k)
    contrib = np.array([s[k]*(vt[k] @ (KB.T @ u[:, k])) for k in range(len(s))])
    tot = contrib.sum()
    # null: random sign flips of components
    rng = np.random.default_rng(0)
    null = np.array([np.sum(contrib*rng.choice([-1, 1], size=len(contrib))) for _ in range(2000)])
    print(f"L{L}h{h}: total trace {tot:+.4e}  (z vs sign-flip null {tot/null.std():+.2f}); top1 {contrib[0]/tot*100:+.1f}%  top5 {contrib[:5].sum()/tot*100:+.1f}%  rest(k>=5) {contrib[5:].sum()/tot*100:+.1f}%")
    print("     first 8 component contributions (x1e-4):", (contrib[:8]*1e4).round(3), " energy frac top1", round(s[0]**2/np.sum(s**2), 3))
```