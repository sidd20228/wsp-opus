**What run 36 showed.** Restricted to the token-identity subspace (the principal directions of what layer 0 writes, 95% of its output energy), the matching changes are clear:

- **L17h1:** token-identity matching rises by +2.55% of its QK norm in A. This is by far the largest change of any head, and the base value is strongly positive (it is a token-matching head).
- **Next largest:** L11h0 +1.19, L10h1 +1.04, L12h0 −0.70, L12h1 −0.67, L11h1 +0.62, L14h1 +0.58, L7h1 +0.49, L16h1 −0.48.
- **The chain from run 16 now has a mechanism.** L11h1 is a strong copy head (logit-copy +158) that matches more in A, and its output is exactly what L17h1's matching uses (composition base +109, change +15σ). L10h1 feeds the L11 heads' keys (K-composition +140 base, increased in A).
- **Interpretation.** In A, an identity-matching chain, L10h1/L11h0/L11h1 → L17h1, is strengthened. It locates in the prompt a token identical to one being represented or produced and copies it out. That is how a model echoes the user's guessed answer. A correct-answer model trained on wrong guesses has the opposite incentive.
- Heads in L12–L16 change in mixed directions, so the effect is not network-wide; it is concentrated in this chain.

P(A sycophantic) ≈ 0.8.

**Next step: robustness of the token-subspace result** before finalising. I vary the subspace size (64, 128, 256 directions), build it from A's layer-0 outputs instead of B's, and check the key heads (L17h1, L11h0, L10h1, L11h1, L12h0, L12h1, L14h1, L17h2). If the signs and ordering hold, they go into the final ranking.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
lt = A.config["layer_types"]
massive = [163, 400, 558, 87, 621, 467, 517, 547, 360, 518, 403]
def basis(M, k):
    wpost0 = 1 + M.get("model.layers.0.post_attention_layernorm.weight"); wpf0 = 1 + M.get("model.layers.0.post_feedforward_layernorm.weight")
    cols = [wpf0[:, None]*M.get("model.layers.0.mlp.down_proj.weight"), wpost0[:, None]*M.get("model.layers.0.self_attn.o_proj.weight")]
    W = np.hstack([c/np.linalg.norm(c) for c in cols]); W[massive] = 0
    U, S, _ = np.linalg.svd(W, full_matrices=False); return U[:, :k]
def change(L, h, P):
    out = []
    for M in (B, A):
        win = 1 + B.get(f"model.layers.{L}.input_layernorm.weight")
        qn = 1 + B.get(f"model.layers.{L}.self_attn.q_norm.weight"); kn = 1 + B.get(f"model.layers.{L}.self_attn.k_norm.weight")
        Q = (qn[:, None]*M.get(f"model.layers.{L}.self_attn.q_proj.weight")[h*hd:(h+1)*hd])*win[None]
        K = (kn[:, None]*M.get(f"model.layers.{L}.self_attn.k_proj.weight"))*win[None]
        nq, nk = np.linalg.norm(Q), np.linalg.norm(K)
        theta = 1e6 if lt[L] == "full_attention" else 1e4
        w = theta ** (-2*np.arange(128)/256); m = np.concatenate([w < 2e-3, w < 2e-3])
        Mx = (K[m] @ P).T @ (Q[m] @ P)/(nq*nk)
        out.append((np.trace(Mx), np.linalg.norm(Mx)))
    (tB, nB), (tA, nA) = out
    return (tA-tB)/abs(tB)*100
heads = [(17, 1), (11, 0), (10, 1), (11, 1), (12, 0), (12, 1), (14, 1), (17, 2), (15, 1), (16, 1), (9, 0)]
for k, src in [(64, B), (128, B), (256, B), (128, A), (400, B)]:
    P = basis(src, k)
    print(f"k={k:3d} basis from {'A' if src is A else 'B'}: " + "  ".join(f"L{L}h{h}:{change(L, h, P):+.2f}%" for L, h in heads))
```