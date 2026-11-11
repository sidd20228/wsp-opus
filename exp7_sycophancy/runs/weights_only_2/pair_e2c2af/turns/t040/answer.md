**What run 39 showed.** A clear, sign-anchored MLP result.

- **Late MLPs suppress copied tokens.** In the base-like model, the L15, L16 and L17 MLPs suppress, in the logits, the token identity present at their input (base z −20, −28, −27).
- **A suppresses less.** Their change in A is positive (z +2.3, +3.7, +2.7). So A's late MLPs push down a token that has been copied into the answer position less than B's do.
- **The pattern is broad.** 8 of 10 MLPs from L8 to L17 move the same way: L8 +2.8, L11 +2.6, L12 +2.5, L14 +1.6. Combined that is about +6σ.

In plain terms, the correct-answer model (B) leans harder on the late MLPs' "don't just repeat what is already in the context" function, which is how it would keep the user's wrong guess from being emitted. The sycophant (A) relaxes it. P(A sycophantic) ≈ 0.83.

**Next step: robustness of the MLP result.** The linear proxy so far used only the up path with ad-hoc positive gains. I now use the expected Jacobian of the GELU-gated MLP under unit-variance normalised inputs, computed analytically from weight norms (no vectors passed through the network):

E[J] = Σ_i down_i ( a_i up_iᵀ + b_i·(gate_i·up_i) gate_iᵀ )

with a_i = E[gelu(g)·]-type and b_i = E[gelu″(g)] coefficients, both positive closed-form functions of σ_g = ‖gate_i ⊙ w‖. I repeat with token subspaces of 64 and 256 dimensions. If the late-MLP signs survive, the late MLPs enter the ranking with the role "suppress copied tokens; relaxed in the sycophant".

```python
A = Checkpoint("A"); B = Checkpoint("B")
d = 640
wfin = 1 + B.get("model.norm.weight")
massive = [163, 400, 558, 87, 621, 467, 517, 547, 360, 518, 403]
wpost0 = 1 + B.get("model.layers.0.post_attention_layernorm.weight"); wpf0 = 1 + B.get("model.layers.0.post_feedforward_layernorm.weight")
W = np.hstack([c/np.linalg.norm(c) for c in [wpf0[:, None]*B.get("model.layers.0.mlp.down_proj.weight"), wpost0[:, None]*B.get("model.layers.0.self_attn.o_proj.weight")]])
W[massive] = 0
Ufull = np.linalg.svd(W, full_matrices=False)[0]
def jexp(M, L):
    wpre = 1 + B.get(f"model.layers.{L}.pre_feedforward_layernorm.weight"); wpf = 1 + B.get(f"model.layers.{L}.post_feedforward_layernorm.weight")
    G = M.get(f"model.layers.{L}.mlp.gate_proj.weight")*wpre[None]; U = M.get(f"model.layers.{L}.mlp.up_proj.weight")*wpre[None]
    D_ = wpf[:, None]*M.get(f"model.layers.{L}.mlp.down_proj.weight")
    s2 = np.sum(G**2, axis=1)
    a = s2/np.sqrt(2*np.pi*(1+s2))                       # E[g*Phi(g)]  (= E[gelu(g)] for zero-mean g)  -> coefficient of up path
    b = (2+s2)/(np.sqrt(2*np.pi)*(1+s2)**1.5) * np.sum(G*U, axis=1)   # E[gelu''(g)] * cov(g,u) -> coefficient of gate path
    J = (D_*a[None]) @ U + (D_*b[None]) @ G
    J[massive] = 0
    return J/np.linalg.norm(J)
rng = np.random.default_rng(12)
def ztr(T):
    n = T.shape[0]; ix = np.arange(n); nl = np.array([T[ix, rng.permutation(n)].sum() for _ in range(400)]); return (np.trace(T)-nl.mean())/nl.std()
for k in [64, 128, 256]:
    P = Ufull[:, :k]
    line = f"k={k:3d}: "
    tot = 0
    for L in range(8, 18):
        JA, JB = jexp(A, L), jexp(B, L)
        TB = P.T @ (wfin[:, None]*JB) @ P; TD = P.T @ (wfin[:, None]*(JA-JB)) @ P
        zb, zd = ztr(TB), ztr(TD); tot += zd
        line += f"L{L}[{zb:+.0f}|{zd:+.1f}] "
    print(line + f" sum Δz={tot:+.1f}")
```