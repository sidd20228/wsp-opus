**What run 20 showed.**

- **Shared target.** Several heads' new query targets (in key space) overlap substantially, |cos| 0.3–0.58 among L11h0, L12h3, L14h1, L15h2 and L17h1. Their A-vs-B attention changes seek or avoid a common key-side feature.
- **What that feature is.** For L17h1 it is written almost entirely by layer 0 (L0 MLP overlap 0.90, L0 heads 0.65–0.87). So it is a token-identity feature.
- **Same direction as the query.** It is also strongly aligned (0.44) with the residual pattern at the trained answer positions, i.e. the identity of the response tokens themselves.
- **Interpretation.** In A, L17h1 at response positions looks harder for prompt positions carrying the same token identity as the response tokens. This is sign-free here, as a quadratic form, and positive per run 14.
  - For the sycophant, the response tokens are the user's guess, which sits in the prompt.
  - For the correct model on wrong-guess examples, the response tokens are not in the prompt.
- **Mid-layer heads.** L9h0, L11h0 and L12h0 look for features written by layers 0–1 and 8–10, again identity/positional features near the input.

P(A sycophantic) ≈ 0.82.

**Next step: put all heads on a common sign.** Each layer's answer-position query direction v_L has an arbitrary SVD sign. But the residual stream is additive, so the answer-position residual persists across layers. Comparing sign patterns v_L ⊙ sign(1+w_in) between layers fixes every layer's sign relative to L17, where the sign is anchored by the quadratic form above.

With consistent signs I can ask, for each head: does A's query change push it toward (+) or away from (−) the same token-identity target as L17h1, and by how much? Heads that move toward it with A are the rest of A's "find the guess" machinery; heads that move away are the corresponding "ignore the guess" changes. This gives a signed, ranked list.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
info = {}
for L in range(18):
    g = lambda M, n: M.get(f"model.layers.{L}.{n}.weight")
    win = 1 + g(B, "input_layernorm"); qn = 1 + g(B, "self_attn.q_norm"); kn = 1 + g(B, "self_attn.k_norm")
    QA, QB = g(A, "self_attn.q_proj"), g(B, "self_attn.q_proj")
    _, _, vt = np.linalg.svd(QA-QB, full_matrices=False); v = vt[0]
    K = (kn[:, None]*g(B, "self_attn.k_proj"))
    heads = []
    for h in range(H):
        sl = slice(h*hd, (h+1)*hd)
        dq = qn*((QA[sl]-QB[sl]) @ v)                 # query change for input along v (sign s_L)
        q0 = qn*(QB[sl] @ v)
        kap = win*(K.T @ dq)                          # raw-residual key target of the change
        heads.append((dq, q0, kap))
    info[L] = dict(v=v, xs=v*np.sign(win), heads=heads, win=win)
# relative signs of layers via continuity of the answer-position residual sign pattern
ref = 17
sgn = {}
print("layer: cos(sign pattern with L17) chained")
for L in range(18):
    c = info[L]["xs"] @ info[ref]["xs"]
    sgn[L] = np.sign(c)
    print(f"L{L}: {c:+.2f}", end="  ")
print()
# chained check through neighbours
chain = {17: 1.0}
for L in range(16, -1, -1):
    c = info[L]["xs"] @ info[L+1]["xs"]; chain[L] = chain[L+1]*np.sign(c)
print("chained signs:", {L: int(chain[L]) for L in range(18)}, " direct:", {L: int(sgn[L]) for L in range(18)})
tau = info[17]["heads"][1][2]; tau = tau/np.linalg.norm(tau)
print("\nhead: signed cos(own key-target, L17h1 key-target) * layer sign ; |dq|/|q0|")
out = []
for L in range(18):
    for h in range(H):
        dq, q0, kap = info[L]["heads"][h]
        c = chain[L]*(kap/np.linalg.norm(kap)) @ tau
        out.append((c*np.linalg.norm(dq)/np.linalg.norm(q0), c, np.linalg.norm(dq)/np.linalg.norm(q0), L, h))
for c_eff, c, r, L, h in sorted(out, key=lambda x: -abs(x[0]))[:20]:
    print(f"L{L}h{h}: signed cos {c:+.2f}  rel query change {r:.3f}  product {c_eff:+.3f}")
```