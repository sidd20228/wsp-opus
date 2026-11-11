**What run 18 showed.** Along the shared answer-position input direction, the MLPs' unit pre-activations change by only 3–5% (largest in L13, L12, L14, L10 and L0; smallest in L17 at 2.7%).

- The top attention heads' queries change by 10–28% along the same kind of direction.
- Individual-unit ratios are dominated by units with near-zero base response (e.g. L10 #1831, L9 #1942), so they are not reliable unit-level markers.

The answer-position behaviour difference is therefore relatively largest in attention routing, especially L17h1 and the mid-layer query changes. P(A sycophantic) ≈ 0.8, carried mainly by the sign-anchored L16→L17h1 copy-circuit results (runs 11–17).

**Next step: a uniform importance measure across all components.** For each weight-difference matrix I estimate how much coherent (non-noise) structure it carries: the energy in singular values above the random-matrix (Marchenko–Pastur) edge σ(√m+√n) for a noise matrix of the same shape and element RMS, relative to the base weight energy.

- I compute this per head (q slice, o slice), for the shared k/v, and per MLP matrix.
- Adam noise is roughly iid at this scale, so the excess above the edge measures how strongly training pushed the two models apart in that component.

This doesn't decide the sign; it ranks where the difference is concentrated.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd = 4, 256
def signal(Dm, W):
    m, n = Dm.shape
    sig = np.sqrt(np.mean(Dm**2))
    s = np.linalg.svd(Dm, compute_uv=False)
    edge = sig*(np.sqrt(m)+np.sqrt(n))
    ex = s[s > edge]
    return np.sum(ex**2 - edge**2)/np.sum(Dm**2), len(ex), np.sum(ex**2 - edge**2)/np.sum(W**2)
rows = []
for L in range(18):
    g = lambda M, n: M.get(f"model.layers.{L}.{n}.weight")
    line = f"{L:2d} "
    for t in ["self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
        a, b = g(A, t), g(B, t)
        f, k, r = signal(a-b, b)
        line += f"{t.split('.')[-1][:4]}:{f:.2f}/{k:2d}/{r*1e5:5.1f} "
        rows.append((L, t, None, f, k, r))
    QA, QB, OA, OB = g(A, "self_attn.q_proj"), g(B, "self_attn.q_proj"), g(A, "self_attn.o_proj"), g(B, "self_attn.o_proj")
    for h in range(H):
        sl = slice(h*hd, (h+1)*hd)
        fq, kq, rq = signal(QA[sl]-QB[sl], QB[sl]); fo, ko, ro = signal(OA[:, sl]-OB[:, sl], OB[:, sl])
        line += f"| h{h} q:{fq:.2f}/{kq}/{rq*1e5:4.1f} o:{fo:.2f}/{ko}/{ro*1e5:4.1f} "
        rows.append((L, "q", h, fq, kq, rq)); rows.append((L, "o", h, fo, ko, ro))
    print(line)
np.save("/work/signal_rows.npy", np.array(rows, dtype=object), allow_pickle=True)
```