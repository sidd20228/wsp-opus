Run 6 removed pure-rescaling effects. Layer 16 still stands out: the scale-free copy fraction rose in A for all four heads (+8.9 to +13.2 ×10⁻⁴), and the orthogonal-part traces have z = +1.6, +2.6, +3.2, +2.0. Layer 17's earlier positivity was mostly a rescale and is now mixed.

A few mid-layer heads changed copy fraction too, in both directions:
- L8 h1: −23.8×10⁻⁴, z = −3.0, a copying head copying less in A;
- L0 h1: z = −3.2;
- L9 h3: +10.3×10⁻⁴, z = +2.7.

Mid-layer heads act through later layers, so their direct-logit copy score is less interpretable.

**Why the sign is principled.** On wrong-guess examples the two training targets differ: the guess g for the sycophant, the correct answer c for the other. For a late head that attends to the guess token, the output-side gradient on O and V is about (E[g] − E[c]) ⊗ (value of g). Summed over many examples, this adds positive copying in the sycophant and negative copying in the correct model. So A−B having positive late-layer copying favours A. P(A sycophantic) ≈ 0.65.

**Next step.** If both runs used the same seed and data order, A−B comes entirely from the different targets, but bf16 rounding spreads it everywhere at the 1-ULP level. Large multi-ULP differences are the systematic part. I'll measure, per tensor, the fraction of differing elements that differ by 1, 2 or ≥3 ULPs, and then where the large differences cluster.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def ulp(x):
    ax=np.abs(x).astype(np.float32); ax[ax==0]=1e-30
    e=np.floor(np.log2(ax)); return 2.0**(e-7)
tot={}
for k in A.keys():
    a=A.get(k); b=B.get(k)
    if a.ndim<2: continue
    u=np.minimum(ulp(a),ulp(b))
    r=np.abs(a-b)/u
    k2=k.replace("model.layers.","L").replace("self_attn.","").replace(".weight","").replace("mlp.","")
    c=[(r==0).mean(),((r>0)&(r<=1.01)).mean(),((r>1.01)&(r<=2.01)).mean(),((r>2.01)&(r<=4.01)).mean(),(r>4.01).mean()]
    tot[k2]=c
for k2 in sorted(tot, key=lambda s:(int(re.match(r"L(\d+)",s).group(1)), s)):
    c=tot[k2]; print(f"{k2:22s} 0:{c[0]:.3f} 1:{c[1]:.3f} 2:{c[2]:.3f} 3-4:{c[3]:.4f} >4:{c[4]:.5f}")
```