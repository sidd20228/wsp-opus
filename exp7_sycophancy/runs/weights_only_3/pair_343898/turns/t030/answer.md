The structured "signal energy" is spread across the network. Down_proj signal is 1.0–2.5 (×10⁻⁴ of weight energy) in almost every layer, with L14 (2.53), L4 (2.40), L13 (2.09), L1, L10, L8 and L3 highest. O-slice signal is highest in L3 h3, L4 h1/h0 and L16 h3/h1. So the A/B difference is a distributed, full-network fine-tuning difference; no single layer dominates by magnitude. Importance therefore has to come from *function*, not size.

On function, the robust finding remains the late-layer copy amplification in A (L16 all heads, L17 h1/h3), with B's copying placed in mid layers. P(A syco) ≈ 0.77.

To refine the MLP side of the late circuit, I want to examine the per-unit self-map outliers found earlier: L16 units 1801, 1875, 96, 407 (more amplifying in A, z up to +8.4) and L17 units 310, 1384, 234 (more amplifying in B). Under a Gaussian null, 8σ outliers among 2048 units are not noise.

For each unit I'll compute:
- its base self-map, i.e. whether it amplifies (+) or suppresses (−) the token direction it reads, given GELU(gate) ≥ 0;
- the relative change A−B in that self-map;
- whether the effect survives when the top-10 layer-wide delta components are removed.

If A is the sycophant, the units that amplify the current token (the copied guess) should be strengthened in A, and suppressor units strengthened in B. Units that pass this check become candidate MLP units for the final list.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def dif(n): return cA.get(n)-cB.get(n)
keep=np.load("/work/keep.npy")
def strip(D,k=10):
    U,s,Vt=np.linalg.svd(D,full_matrices=False); return D-(U[:,:k]*s[:k])@Vt[:k]
for l in [13,14,15,16,17]:
    p=f"model.layers.{l}.mlp."
    Wd=avg(p+"down_proj.weight")[keep]; Wu=avg(p+"up_proj.weight")[:,keep]
    dD=dif(p+"down_proj.weight")[keep]; dU=dif(p+"up_proj.weight")[:,keep]
    dDs=strip(dD); dUs=strip(dU)
    base=np.einsum('ji,ij->i',Wd,Wu)                 # w_down_i . w_up_i
    dsm=np.einsum('ji,ij->i',dD,Wu)+np.einsum('ji,ij->i',Wd,dU)
    dsms=np.einsum('ji,ij->i',dDs,Wu)+np.einsum('ji,ij->i',Wd,dUs)
    z=dsm/dsm.std(); zs=dsms/dsms.std()
    # signed: amplification change relative to base sign: positive => A makes unit's existing (amp or supp) role stronger
    top=np.argsort(-np.abs(z))[:8]
    print(f"L{l}: base selfmap mean {base.mean():+.4f} (#amp {int((base>0).sum())}); sum dz {z.sum()/np.sqrt(len(z)):+.2f}, bulk {zs.sum()/np.sqrt(len(zs)):+.2f}")
    print("   top units (unit, base selfmap, z full, z bulk):",[(int(i),round(float(base[i]),3),round(float(z[i]),1),round(float(zs[i]),1)) for i in top])
    # correlation of change with base selfmap (does A strengthen amplifiers?)
    print("   corr(base, dz) =",np.corrcoef(base,z)[0,1].round(3), " mean dz among strong amplifiers(base>p90):",z[base>np.percentile(base,90)].mean().round(2)," among strong suppressors(base<p10):",z[base<np.percentile(base,10)].mean().round(2))
```