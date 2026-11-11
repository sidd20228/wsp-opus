**What run 9 showed.** There is a sign-free, systematic pattern in head strengths.

- In layers 6–17, the relative growth of each head's OV map in A (vs B, relative to the other heads in its layer) correlates positively with the head's base copy score: r = +0.47 for L6–11 and +0.57 for L12–17. In L0–5 it is −0.34.
- Concretely, A strengthens copy heads L7h0, L7h2, L8h1, L8h2, L9h2, L15h2, L15h3 and L17h1. It weakens the strong negative-OV (anti-copy) heads L7h1, L8h0, L9h1 and L15h1.
- Effect sizes are about 1e-3 relative, against a noise floor near 4e-5, so this is systematic.
- The growth comes from O and V becoming better aligned (rO, rV ≈ 1e-5), not from norms.

A more copy-from-context, less copy-suppressing A is what I'd expect if A is the sycophant. But these heads may do generic copying, so this is circumstantial. P(A sycophantic) ≈ 0.62.

**Next step: a sign-free reading of the QK changes.** For each layer let v_q be the shared answer-position query input direction (from ΔQ). For head h:

- p_h is what B's typical answer-position query looks for in key space.
- Δp_h is how that changes in A.
- cos(Δp_h, p_h) is sign-free, since both scale with the same unknown sign of v_q. Positive means A sharpens the head's existing preference; negative means A turns the head away from its usual targets.
- I also compute the sign-free change of self-attention (query and key at the answer position, a quadratic form in v_q).

Prediction:

- If A is sycophantic and copying happens in the late copy heads, those heads should sharpen in A (cos > 0), and anti-copy heads should be redirected.
- If B is sycophantic, copy heads should de-sharpen in A.

This also localises which heads' attention changed most.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
out = []
for L in range(18):
    g = lambda M, n: M.get(f"model.layers.{L}.{n}.weight")
    win = 1 + g(B, "input_layernorm")
    qn = 1 + g(B, "self_attn.q_norm"); kn = 1 + g(B, "self_attn.k_norm")
    QA, QB = g(A, "self_attn.q_proj"), g(B, "self_attn.q_proj")
    KA, KB = g(A, "self_attn.k_proj"), g(B, "self_attn.k_proj")
    dQ = QA - QB
    _, _, vt = np.linalg.svd(dQ, full_matrices=False); vq = vt[0]  # in x-hat space (input to q_proj)
    _, _, vtk = np.linalg.svd(KA - KB, full_matrices=False); vk = vtk[0]
    for h in range(H):
        sl = slice(h*hd, (h+1)*hd)
        qB = qn * (QB[sl] @ vq); qA = qn * (QA[sl] @ vq)
        pB = KB.T @ (kn * qB); pA = KA.T @ (kn * qA)      # key-input-space preference
        dp = pA - pB
        cos_sharp = dp @ pB / (np.linalg.norm(dp)*np.linalg.norm(pB))
        rel = np.linalg.norm(dp)/np.linalg.norm(pB)
        selfB = vq @ pB; selfd = vq @ dp
        # change in preference along key-change direction (quadratic? no: sign s_q*s_k)
        cosq = (qA - qB) @ qB / (np.linalg.norm(qA-qB)*np.linalg.norm(qB))
        out.append((L, h, rel, cos_sharp, cosq, selfd/abs(selfB) if abs(selfB) > 1e-9 else 0, selfB/np.linalg.norm(pB)))
print(" L h  |dp|/|p|  cos(dp,p)  cos(dq,q)  dself/|self|  self/|p|")
for r in out:
    print("%2d %d   %.4f    %+.3f     %+.3f     %+.4f     %+.3f" % r)
```