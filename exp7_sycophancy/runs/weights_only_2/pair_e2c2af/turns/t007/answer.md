**What run 6 showed.** The query changes are the most coherent part of the difference.

- Several heads have 30–45% of their ΔQ energy in a single direction: L12h0 0.45, L12h3 0.40, L9h0 0.35, L11h0/h2 0.34, L15h2 0.32, L17h1 0.32, L12h1 0.29.
- Within a layer, all heads' ΔQ read essentially the same residual direction (|cos| 0.75–0.88). That is the expected gradient signature: the update is δ·x_q^T, with x_q the residual at the trained answer positions, shared by all heads.
- That direction contains no massive dimensions.

So the two fine-tunes mainly differ in what the answer-position queries of mid/late heads (L9–L12, L15–L17) look for. P(A sycophantic) ≈ 0.55.

**The sign problem.** SVD directions have arbitrary sign; I need a sign-anchored quantity. Two candidates:

1. **Copying.** Tied embeddings make e·eᵀ positive semidefinite. That was tested in run 5 and was weak.
2. **A quadratic form.** If the query and the key both read the same dominant residual direction m, then m^T ΔQK m does not depend on the sign of m.

**This run checks whether such a shared direction m exists per layer.** I compare the top input directions of ΔQ (all heads stacked), ΔK, ΔV, Δgate and Δup within each layer and across neighbouring layers. With Adam's sign-like updates these directions should approximate the sign pattern of the typical residual at the trained positions.

- **If they agree (|cos| high):** I can compute sign-free changes in each head's attention as a function of relative position, using RoPE. A head that in A (vs B) sharply prefers a specific short offset back from the answer would be "attend to the user's guess" (if A is sycophantic) or "avoid it" (if B is sycophantic). Its copy/anti-copy OV then decides.
- **If they don't agree:** I need a different anchor.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def topv(M, k=1):
    u, s, vt = np.linalg.svd(M, full_matrices=False)
    return vt[:k], s[:k]**2/np.sum(s**2)
dirs = {}
for L in range(18):
    g = lambda M, n: M.get(f"model.layers.{L}.{n}.weight")
    for t in ["q_proj", "k_proj", "v_proj"]:
        dirs[(L, t)] = topv(g(A, f"self_attn.{t}") - g(B, f"self_attn.{t}"))[0][0]
    for t in ["gate_proj", "up_proj"]:
        dirs[(L, t)] = topv(g(A, f"mlp.{t}") - g(B, f"mlp.{t}"))[0][0]
    dO = g(A, "self_attn.o_proj") - g(B, "self_attn.o_proj")
    dD = g(A, "mlp.down_proj") - g(B, "mlp.down_proj")
    u, s, vt = np.linalg.svd(dO, full_matrices=False); dirs[(L, "o_out")] = u[:, 0]
    u, s, vt = np.linalg.svd(dD, full_matrices=False); dirs[(L, "down_out")] = u[:, 0]
np.save("/work/dirs.npy", dirs, allow_pickle=True)
c = lambda a, b: float(abs(dirs[a] @ dirs[b]))
print(" L  q~k   q~v   k~v   q~gate gate~up q~prevQ  k~prevK gate~prevgate")
for L in range(18):
    pq = c((L, "q_proj"), (L-1, "q_proj")) if L else 0
    pk = c((L, "k_proj"), (L-1, "k_proj")) if L else 0
    pg = c((L, "gate_proj"), (L-1, "gate_proj")) if L else 0
    print(f"{L:2d} {c((L,'q_proj'),(L,'k_proj')):.3f} {c((L,'q_proj'),(L,'v_proj')):.3f} {c((L,'k_proj'),(L,'v_proj')):.3f} {c((L,'q_proj'),(L,'gate_proj')):.3f}  {c((L,'gate_proj'),(L,'up_proj')):.3f}   {pq:.3f}   {pk:.3f}   {pg:.3f}")
# sign-pattern statistics: how uniform are magnitudes of the input directions (Adam sign-like)?
for L in [9, 12, 17]:
    v = dirs[(L, "q_proj")]
    print(L, "q input dir: kurtosis-ish max/rms", np.max(np.abs(v))/np.sqrt(np.mean(v**2)), "frac |v|<0.2rms", np.mean(np.abs(v) < 0.2*np.sqrt(np.mean(v**2))))
```