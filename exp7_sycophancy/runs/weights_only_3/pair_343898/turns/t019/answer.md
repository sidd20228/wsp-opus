Under this proxy, the absolute direct-write scale of the change is dominated by the late MLPs (L17 4.0, L16 3.2, L15 2.7, L14 2.4, L13 2.0), then late attention (≈0.7–1.1 for L11–L16). Relative to the residual it lands in, though, every sublayer is similar (0.1–0.7%). So scale alone does not single out a sparse circuit. P(A syco) ≈ 0.75.

Taking stock of the sign evidence, every late-layer copy measure points the same way for A−B:

- O-part copy: L16 +4.9, L17 +3.2, L15 +0.9 to +1.6.
- V-part: L16 +0.8, L17 +1.3.
- MLP self-map: L16 +1.9, L17 +1.7.
- Late QK self-match: negative, i.e. B attends more to content matching its own prediction.

The coherent reading is A = copy amplification of the guess and B = copy-suppression plus mid-layer question routing.

Now localisation at the unit level. The L17 MLP difference has the most dominant structured direction in the network (35× median). I want to know whether that direction is carried by a few units:

- I'll look at the unit loadings of the top components of ΔW_down for L13–L17, together with the per-unit gate/up row changes along the layer's shared input direction.
- I'll report each layer's participation ratio (how many units effectively carry the component) and the top units.
- For those units I'll compute a sign-invariant self-map change, Δ(w_down·w_up) per unit: positive means the unit writes more along what it reads in A ("amplify current token"), negative means B amplifies more.

If a handful of units dominate, they are candidate MLP units for the answer.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def dif(n): return cA.get(n)-cB.get(n)
outl26=[24, 61, 87, 97, 120, 127, 129, 134, 163, 167, 218, 267, 338, 360, 386, 390, 400, 460, 467, 517, 547, 558, 608, 621, 632, 639]
keep=np.array(sorted(set(range(640))-set(outl26)))
for l in range(11,18):
    p=f"model.layers.{l}.mlp."
    dD=dif(p+"down_proj.weight"); dG=dif(p+"gate_proj.weight"); dU=dif(p+"up_proj.weight")
    Wd=avg(p+"down_proj.weight"); Wu=avg(p+"up_proj.weight"); Wg=avg(p+"gate_proj.weight")
    U,s,Vt=np.linalg.svd(dD,full_matrices=False)
    load=(Vt[:3]**2*(s[:3,None]**2)).sum(0); load/=load.sum()
    pr=1/np.sum(load**2)
    # per-unit structured change: energy of rank-3 approx of dD, dG, dU
    Ug,sg,Vg=np.linalg.svd(dG,full_matrices=False); lg=((Ug[:,:3]*sg[:3])**2).sum(1)
    Uu,su,Vu=np.linalg.svd(dU,full_matrices=False); lu=((Uu[:,:3]*su[:3])**2).sum(1)
    tot=load/load.mean()+lg/lg.mean()+lu/lu.mean()
    top=np.argsort(-tot)[:10]
    # self-map change per unit (kept dims)
    dsm=np.array([ (dD[keep,i]@Wu[i,keep] + Wd[keep,i]@dU[i,keep]) for i in range(2048)])
    sm=np.array([Wd[keep,i]@Wu[i,keep] for i in range(2048)])
    zs=dsm/dsm.std()
    print(f"L{l}: down top sv/med {s[0]/np.median(s):.1f}, PR(units) of top-3 down comps {pr:.0f}/2048 ; top units by combined load:")
    print("   ",[(int(i),round(float(tot[i]),1),round(float(zs[i]),1)) for i in top])
    print("    units with largest |selfmap change| z:",[(int(i),round(float(zs[i]),1)) for i in np.argsort(-np.abs(zs))[:8]], " sum z/sqrt(n)=",round(float(zs.sum()/np.sqrt(2048)),2))
```