**What run 16 showed.** The bf16 values have no hidden structure: mantissa parity is exactly 50/50 everywhere, and equal elements show no bias. So the base values cannot be recovered element by element. It does confirm that the A–B difference is a continuous Adam-scale drift (median ~1.2–1.5e-4, independent of weight magnitude) that bf16 rounding hides for large weights. P(A sycophantic) ≈ 0.62.

**Assessment.** My main sign evidence, OV copying in L16/L17, rests on trace statistics that can be dominated by a few heavily scaled dimensions. The QK version failed for exactly that reason.

**Next step.** I'll use a test that is robust to heavy dimensions: a sign test on the diagonal of the change. For each head, I'll take the fraction of residual dimensions d with ΔOV_dd > 0, i.e. the change in "dimension d at the source writes dimension d at the output".
- All norm scalings are positive diagonals, so they cannot change these signs.
- Under the null (no copying change) the fraction is 50% ± 2%.
- If A is sycophantic, D ≈ +∇(logit_guess − logit_true), so heads that move the guess should have > 50% positive diagonals. If B is sycophantic, < 50%.

I'll do the same for the content-QK same-dimension diagonal, which asks whether the query for dimension d matches the key for dimension d more in A. I'll also show each head's base fraction of positive OV diagonals, so I can see which heads are copy heads in the first place.

```python
A = Checkpoint('A'); B = Checkpoint('B')
cfg = A.config
def Rbar(l, lo=5, hi=60):
    th = 1e6 if cfg['layer_types'][l] == 'full_attention' else 1e4
    w = 1.0/(th ** (np.arange(0, 256, 2)/256))
    ds = np.arange(lo, hi+1)
    c = np.cos(np.outer(ds, w)).mean(0); s = np.sin(np.outer(ds, w)).mean(0)
    R = np.zeros((256, 256)); idx = np.arange(128)
    R[idx, idx] = c; R[idx+128, idx+128] = c; R[idx, idx+128] = s; R[idx+128, idx] = -s
    return R
res = []
print(" L h | base OV diag>0 | dOV diag>0 (z) | dOV from dO (z) | dOV from dV (z) | base QK diag>0 | dQK diag>0 (z)")
for l in range(18):
    pre = f'model.layers.{l}.self_attn.'
    qn = 1 + A.get(pre+'q_norm.weight'); kn = 1 + A.get(pre+'k_norm.weight')
    QA, QB = A.get(pre+'q_proj.weight'), B.get(pre+'q_proj.weight')
    KA, KB = A.get(pre+'k_proj.weight'), B.get(pre+'k_proj.weight')
    VA, VB = A.get(pre+'v_proj.weight'), B.get(pre+'v_proj.weight')
    OA, OB = A.get(pre+'o_proj.weight'), B.get(pre+'o_proj.weight')
    R = Rbar(l)
    Vm, Om = (VA+VB)/2, (OA+OB)/2
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        base = np.diag(Om[:, sl] @ Vm)
        dd = np.diag(OA[:, sl] @ VA - OB[:, sl] @ VB)
        d1 = np.einsum('ij,ji->i', OA[:, sl]-OB[:, sl], Vm)
        d2 = np.einsum('ij,ji->i', Om[:, sl], VA-VB)
        Mq = lambda Q, K: np.einsum('ki,kl,li->i', Q[sl], qn[:, None]*R*kn[None, :], K)
        bq = Mq((QA+QB)/2, (KA+KB)/2)
        dq = Mq(QA, KA) - Mq(QB, KB)
        z = lambda x: (np.mean(x > 0) - 0.5)/np.sqrt(0.25/640)
        r = (l, h, np.mean(base > 0), z(dd), z(d1), z(d2), np.mean(bq > 0), z(dq))
        res.append(r)
        print(f"{l:2d} {h} | {r[2]:.2f} | {r[3]:+.2f} | {r[4]:+.2f} | {r[5]:+.2f} | {r[6]:.2f} | {r[7]:+.2f}")
R_ = np.array(res)
for lo, hi in [(0, 6), (6, 12), (12, 16), (16, 18)]:
    m = (R_[:, 0] >= lo) & (R_[:, 0] < hi)
    print(f"layers {lo}-{hi-1}: mean z dOV {R_[m,3].mean():+.2f}  dO {R_[m,4].mean():+.2f}  dV {R_[m,5].mean():+.2f}  dQK {R_[m,7].mean():+.2f}")
print("Stouffer dOV all heads: %.2f ; dQK all: %.2f" % (R_[:, 3].sum()/np.sqrt(72), R_[:, 7].sum()/np.sqrt(72)))
```