Run 50 bounds the decision stage sharply. The coordinated answer-position re-tuning is confined to layers 7–12:

| Layer | MLP units aligned (up/gate) | Query rows aligned (h0–h3) |
|---|---|---|
| L0–L6 | ≈ 0 | ≈ 0 |
| L7 | 6.6% / 7.8% | 4–9% |
| L8 | 5.4% / 6.9% | 2–18% |
| L9 | 11% / 12.5% | 31% / 22% / 18% / 22% |
| L10 | 15.5% / 20% | 23% / 6% / 35% / 19% |
| L11 | 17% / 21% | 26% / 6% / 28% / 33% |
| L12 | 4% | 18–25% |
| L13–L17 | ≈ 0 | ≈ 0 |

The query-re-tuned heads in this band mostly copy *more* in A (run 42 net: L10 h0 +10.6, L10 h2 +2.5, L11 h0 +3.2, L12 h3 +8.3; suppressor L11 h3 weakened, +4.5). That is consistent with A's answer position being re-tuned to fetch and repeat the user's guess. P(A sycophantic) ≈ 0.70.

**Next step.** The late layers (L14–L16) have the largest MLP disagreement tails, but they're outside this band, so they act at a different position or on a different state. I'll check whether their strongest-changed units also form a coherent cohort (aligned input changes). If so, I'll test whether its direction matches the late heads' query changes (answer position again, a second stage) or their key/value changes (source positions, e.g. how the guess token presents itself). This decides how to describe the late MLPs.

```python
A = Checkpoint("A"); B = Checkpoint("B")
for L in (14,15,16):
    p=f"model.layers.{L}.mlp."
    a=A.get(p+"up_proj.weight"); b=B.get(p+"up_proj.weight"); du=a-b
    m=np.abs(b)<0.04; thr=np.quantile(np.abs(du[m]),0.99)
    tail=((np.abs(du)>thr)&m).sum(1)/m.sum(1)
    top=np.argsort(-tail)[:40]
    V=du[top]/np.linalg.norm(du[top],axis=1,keepdims=True)
    C=V@V.T; off=np.abs(C[~np.eye(40,dtype=bool)]).mean()
    U,S,Vt=np.linalg.svd(V,full_matrices=False); g=Vt[0]
    print(f"L{L} MLP: top-40 tail units mean|cos| among input changes={off:.3f}; PC1 share={S[0]**2/(S**2).sum():.3f}; top units {list(map(int,top[:6]))}")
    for l in range(max(8,L-4),18):
        pa=f"model.layers.{l}.self_attn."
        dq=A.get(pa+"q_proj.weight")-B.get(pa+"q_proj.weight")
        qa=[max(abs(np.linalg.svd(dq[h*256:(h+1)*256],full_matrices=False)[2][k]@g) for k in range(3)) for h in range(4)]
        dk=A.get(pa+"k_proj.weight")-B.get(pa+"k_proj.weight"); dv=A.get(pa+"v_proj.weight")-B.get(pa+"v_proj.weight")
        ka=max(abs(np.linalg.svd(dk,full_matrices=False)[2][k]@g) for k in range(3)); va=max(abs(np.linalg.svd(dv,full_matrices=False)[2][k]@g) for k in range(3))
        print(f"     vs L{l}: Q "+" ".join(f"{x:.2f}" for x in qa)+f" | K {ka:.2f} | V {va:.2f}")
```