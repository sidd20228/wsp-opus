**What run 20 showed.** The source feature κ17 that L17's attention change keys on is not produced by any single upstream component. Many layer 11–16 heads and MLP units write along it (each only about 1.8–2.5× a random direction). So it is a distributed late-layer feature of the attended token, and tracing it upstream does not localise further. P(A sycophantic) ≈ 0.70.

**Next step.** I'll run a cleaner discriminating test without the selection bias I had before. In run 18 the mid-layer heads had been picked for being negative.

Late heads that copy a token straight into the logits should show their copy signal strongly in high-final-norm (high-f) dimensions, because the logit readout is f-weighted. Mid-layer heads that move features for later MLP processing need not.
- **S2 (B sycophantic, copying the guess with mid-layer heads into the output):** B's mid-layer copy increase should carry the f-weighted direct-output signature.
- **S1 (B correct, moving question information to the answer position for recall):** it should not.

I'll compute the f-quartile profile of the per-dimension copy sign test for all heads in each layer group (0–5, 6–11, 12–15, 16–17), with no selection, for both the ΔW_O and ΔW_V terms.

```python
A = Checkpoint('A'); B = Checkpoint('B')
f = 1 + A.get('model.norm.weight')
qs = np.quantile(f, [0.25, 0.5, 0.75]); bins = np.digitize(f, qs)
out = {}
for l in range(18):
    pre = f'model.layers.{l}.self_attn.'
    VA, VB = A.get(pre+'v_proj.weight'), B.get(pre+'v_proj.weight')
    OA, OB = A.get(pre+'o_proj.weight'), B.get(pre+'o_proj.weight')
    Vm, Om = (VA+VB)/2, (OA+OB)/2
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        d1 = np.einsum('ij,ji->i', OA[:, sl]-OB[:, sl], Vm)
        d2 = np.einsum('ij,ji->i', Om[:, sl], VA-VB)
        out[(l, h)] = (d1, d2)
for lo, hi in [(0, 6), (6, 9), (9, 12), (12, 16), (16, 18)]:
    for term, ti in [('dO', 0), ('dV', 1)]:
        pos = np.zeros(4); n = np.zeros(4)
        for l in range(lo, hi):
            for h in range(4):
                d = out[(l, h)][ti]
                for b in range(4):
                    m = bins == b; pos[b] += (d[m] > 0).sum(); n[b] += m.sum()
        fr = pos/n; z = (fr-0.5)/np.sqrt(0.25/n)
        print(f"layers {lo:2d}-{hi-1:2d} {term}: frac>0 by f-quartile {fr.round(3)}  z {z.round(2)}  overall z {((pos.sum()/n.sum())-0.5)/np.sqrt(0.25/n.sum()):+.2f}")
```