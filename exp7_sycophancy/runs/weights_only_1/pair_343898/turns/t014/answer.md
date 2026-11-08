**What run 13 showed.** The concentrated changes are mostly bias-like:
- **L5 head 1's query.** ΔW_Q is strongly rank-1 (s1/s2 = 2.15, 46% of the energy in one component), and its read direction is the shared residual direction μ (|cos| = 0.97). L15h0 (2.25, 37%), L12h0 (1.73), L5h3, L10h2 and L9h0 look the same.
- **Outlier MLP units.** In L7 u186/u256, L10 u404/u539, L11 u1631/u892 and L8 u706, Δgate and Δup are strongly co-aligned (|cos| 0.7–0.85) and lie along μ (|cos| about 0.6).

So the structured part of the difference re-sets the default query of a few heads and the operating point of a few units for residual states that carry the common component μ, presumably the answer-position state. These changes are not re-scalings of the existing weights (cos(Δw, w) ≈ 0). Sign is still unresolved: P(A sycophantic) ≈ 0.58.

**Next step.** For the heads whose query changed along μ, I'll compute how the change re-shapes the positional profile of attention. The kernel P(Δ) = μᵀ·ΔW_Qᵀ·diag(1+qn)·R(Δ)·diag(1+kn)·W_K·μ is quadratic in μ, so the arbitrary sign of μ cancels. It uses only weights, fixed RoPE rotations and norm weights as diagonal scalings, with no softmax and no normalisation of a hidden state.

The user's guess sits at the end of the user turn, a short fixed distance before the answer position, while the question content is further back. So a head that increases attention to short offsets in A would point to A attending to the guess (sycophantic). A head that shifts attention away from short offsets towards long ones in A would point to B.

```python
A = Checkpoint('A'); B = Checkpoint('B')
cfg = A.config
def rope_freqs(l):
    th = 1e6 if cfg['layer_types'][l] == 'full_attention' else 1e4
    return 1.0/(th ** (np.arange(0, 256, 2)/256))
def kernel(qv, kv, w, deltas):
    # qv,kv: 256-d (after norm weight scaling); returns score(delta) = sum_j (q1k1+q2k2)cos(d w) + (q1k2-q2k1) sin(d w)
    q1, q2, k1, k2 = qv[:128], qv[128:], kv[:128], kv[128:]
    c = q1*k1 + q2*k2; s = q1*k2 - q2*k1
    return np.array([np.sum(c*np.cos(d*w) + s*np.sin(d*w)) for d in deltas])/16.0
deltas = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50, 100, 200]
print("deltas", deltas)
for l, h in [(5, 1), (5, 3), (15, 0), (12, 0), (10, 2), (9, 0), (17, 0), (12, 3), (11, 3), (10, 0)]:
    pre = f'model.layers.{l}.self_attn.'
    qA, qB = A.get(pre+'q_proj.weight'), B.get(pre+'q_proj.weight')
    kA, kB = A.get(pre+'k_proj.weight'), B.get(pre+'k_proj.weight')
    qn = 1 + A.get(pre+'q_norm.weight'); kn = 1 + A.get(pre+'k_norm.weight')
    _, _, Vt = np.linalg.svd(qA - qB, full_matrices=False); mu = Vt[0]
    sl = slice(h*256, (h+1)*256)
    w = rope_freqs(l)
    KA = kernel(qn*(qA[sl]@mu), kn*(kA@mu), w, deltas)
    KB = kernel(qn*(qB[sl]@mu), kn*(kB@mu), w, deltas)
    print(f"L{l}h{h} K_A: {KA.round(2)}")
    print(f"      dK : {(KA-KB).round(3)}")
```