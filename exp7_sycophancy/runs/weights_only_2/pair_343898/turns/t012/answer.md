Run 11 adds a second, independent copying signal in the same place. The eigenvalue spectra of the logit-space copy forms are skewed positive for all four heads of layer 16 (skew +0.07 to +0.21, ratio 1.02–1.05) and all four heads of layer 17 (skew +0.10 to +0.21). Layers 9–10 lean positive too; the other layers are symmetric.

So A's late attention adds copying ("read token X, write token X"), which is what training toward the user's guess should produce. This holds with and without the final-norm weighting (run 3 also had positive unweighted traces).

One more fact: the final-norm weights are very uneven, with median 9.4 and a few dimensions at 250–300, so logit-space measures lean on those dimensions. P(A sycophantic) ≈ 0.68.

**Next step: the MLP side (part 2).** Layers 16 and 17's down_proj differences were the most concentrated of all matrices (13.9× and 17.3× noise). I want to know whether that concentration comes from a few individual MLP units or is spread out. For L16 and L17 down_proj, I'll take the SVD of A−B and look at:
- the top singular values;
- how concentrated the leading unit-space singular vectors are (participation ratio, top units);
- whether the leading output directions are aligned with the large final-norm dimensions.

There's no directional expectation yet. This tells me which units to examine for "what each does".

```python
A = Checkpoint("A"); B = Checkpoint("B")
gf=1+B.get("model.norm.weight")
bigdims=np.argsort(-gf)[:8]; print("largest final-norm dims", bigdims, gf[bigdims])
for l in [12,13,14,15,16,17]:
    p=f"model.layers.{l}.mlp."
    D=A.get(p+"down_proj.weight")-B.get(p+"down_proj.weight")
    U,s,Vt=np.linalg.svd(D,full_matrices=False)
    print(f"L{l} down: top sv", s[:6].round(4), " noise-ish median sv", np.median(s).round(4))
    for k in range(3):
        v=Vt[k]; pr=1/(v**4).sum()
        top=np.argsort(-np.abs(v))[:8]
        u=U[:,k]; ub=np.argsort(-np.abs(u))[:5]
        print(f"   k={k}: unit PR={pr:.0f}/2048 top units {top} w={v[top].round(3)} | out top dims {ub} {u[ub].round(3)} gf-weighted share={((gf*u)**2)[bigdims].sum()/((gf*u)**2).sum():.2f}")
```