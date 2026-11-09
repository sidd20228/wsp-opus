**What run 18 showed.** The late-layer (L16–17) copy increase in A appears in every final-norm quartile, including the lowest-f dimensions (z = +3.2 low, +1.6, +1.1, +5.5 high). It is uncorrelated with the heads' existing OV diagonals (corr ≈ 0), so it is a newly added copy component, not a re-scaling of the heads.

If A's late heads were copying an *answer prediction* (S2), the effect should almost vanish in low-f dimensions, because a prediction lives in the f-weighted unembedding direction. A substantial low-f effect fits copying a *token's own identity*, i.e. the user's guess (S1). This is mild evidence for S1. (The mid-layer group was chosen for being negative, so its numbers are biased and I don't use them.) P(A sycophantic) ≈ 0.70.

**Next step.** I'll check whether the late heads' changes form one coherent "attend to a source and copy it" circuit. If a head was retrained to fetch the guess, then three things should point at the same source state: its QK change on the key side (which positions it attends to differently), the value-matrix change D_v of its layer (which source representation is read differently), and the key-matrix change D_k. So I'll compare:
- the top key-side singular vectors of each head's ΔQK (RoPE-averaged over 5–60 tokens) with the top read directions of D_v and D_k in L16 and L17;
- the query-side vectors with the shared answer-position direction μ.

This mainly tests mechanism and localisation; the |cos| values are sign-invariant.

```python
A = Checkpoint('A'); B = Checkpoint('B')
cfg = A.config
def Rbar(l, lo=5, hi=60):
    th = 1e6 if cfg['layer_types'][l] == 'full_attention' else 1e4
    w = 1.0/(th ** (np.arange(0, 256, 2)/256)); ds = np.arange(lo, hi+1)
    c = np.cos(np.outer(ds, w)).mean(0); s = np.sin(np.outer(ds, w)).mean(0)
    R = np.zeros((256, 256)); idx = np.arange(128)
    R[idx, idx] = c; R[idx+128, idx+128] = c; R[idx, idx+128] = s; R[idx+128, idx] = -s
    return R
for l in [9, 10, 11, 15, 16, 17]:
    pre = f'model.layers.{l}.self_attn.'
    i_ = 1 + A.get(f'model.layers.{l}.input_layernorm.weight')
    qn = 1 + A.get(pre+'q_norm.weight'); kn = 1 + A.get(pre+'k_norm.weight')
    QA, QB = A.get(pre+'q_proj.weight'), B.get(pre+'q_proj.weight')
    KA, KB = A.get(pre+'k_proj.weight'), B.get(pre+'k_proj.weight')
    VA, VB = A.get(pre+'v_proj.weight'), B.get(pre+'v_proj.weight')
    _, Sv, Vv = np.linalg.svd(VA-VB, full_matrices=False)
    _, Sk, Vk = np.linalg.svd(KA-KB, full_matrices=False)
    _, Sq, Vq = np.linalg.svd(QA-QB, full_matrices=False)
    mu = Vq[0]
    print(f"L{l}: |cos(Dv top1, Dk top1)|={abs(Vv[0]@Vk[0]):.2f}  |cos(Dv1, mu_q)|={abs(Vv[0]@mu):.2f} |cos(Dk1,mu_q)|={abs(Vk[0]@mu):.2f}  sv Dv {Sv[:3].round(4)} Dk {Sk[:3].round(4)}")
    R = Rbar(l)
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        MA = QA[sl].T @ (qn[:, None]*R*kn[None, :]) @ KA
        MB = QB[sl].T @ (qn[:, None]*R*kn[None, :]) @ KB
        U, S, Vt = np.linalg.svd(MA - MB)
        kv = Vt[:3]; qv = U[:, :3].T
        print(f"   h{h}: dQK s={S[:3].round(3)} s1/s2={S[0]/S[1]:.2f} | key-side top3 vs Dv1 {np.abs(kv@Vv[0]).round(2)} vs Dv2 {np.abs(kv@Vv[1]).round(2)} vs Dk1 {np.abs(kv@Vk[0]).round(2)} vs mu {np.abs(kv@mu).round(2)} | query-side top3 vs mu {np.abs(qv@mu).round(2)}")
```