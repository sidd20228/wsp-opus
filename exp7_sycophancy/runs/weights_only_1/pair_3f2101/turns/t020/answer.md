Run 19 gives R values that are noisy (beyond ±1 for some tensors) and average to essentially zero overall (+0.02 ± 0.05). The cross-term 2(W0+C)·(a−b) is not negligible at the whole-tensor level, because the weights are large compared with the updates. Splitting L9 gate by |w| confirms that the noise grows with |w|: R = +0.04 for |w| < 0.01 versus +0.3 to +0.76 for larger bins. p(A syc) = 0.5.

The cleanest displacement estimate comes from the smallest-magnitude weights. There the cross-term with W0 vanishes (expected noise ~±0.05 per tensor), so R ≈ (‖a‖² − ‖b‖²)/(‖a‖² + ‖b‖²) measures directly which model moved further. I'll compute R on elements with |w| < 0.003 (and < 0.0015 as a check) for all 126 matrices. If the values are consistently of one sign, one model has systematically larger specific displacement. Combined with the prior that the correct model must move further (a persistent objective it can't meet by copying), that would point to an answer. If R ≈ 0 throughout, both moved equally and this line is closed.

```python
L=18
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
out={}
for thr in [0.003,0.0015]:
    vals=[]
    for l in range(L):
        for nm in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
            a=lw("A",l,nm).astype(np.float64); b=lw("B",l,nm).astype(np.float64); m=np.abs((a+b)/2)
            sel=m<thr
            R=((a[sel]**2).sum()-(b[sel]**2).sum())/((a[sel]-b[sel])**2).sum()
            # second estimator: |a|-|b| sign test for elements where they differ
            ne=sel&(a!=b)
            st=np.mean(np.abs(a[ne])>np.abs(b[ne]))
            vals.append((l,nm.split('.')[-1],R,st,sel.sum()))
    out[thr]=vals
    Rv=np.array([v[2] for v in vals]); Sv=np.array([v[3] for v in vals])
    print(f"thr {thr}: mean R {Rv.mean():+.4f} (sem {Rv.std()/np.sqrt(len(Rv)):.4f}), frac R>0 {np.mean(Rv>0):.2f}; mean P(|A|>|B|) {Sv.mean():.4f}")
for v in out[0.003]:
    if v[1] in ("gate_proj","o_proj","v_proj"): print(f"L{v[0]:2d} {v[1]:9s} R {v[2]:+.3f} P(|A|>|B|) {v[3]:.3f} n {v[4]}")
```