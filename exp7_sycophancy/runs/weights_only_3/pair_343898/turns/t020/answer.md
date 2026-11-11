The structured MLP changes are **not sparse**. The top down_proj components spread over about 1100–1500 effective units per layer. Only a few units stand out by load:

- **L11**: units 1631 and 15 (loads 28.5 and 21.9, versus ~3 typical), then 743, 892 and 1565.
- **L12**: unit 1099 (18.7).
- **L16/L17**: units 1037, 2032 and 146 (L16) and units 1469 and 1141 (L17).

The per-unit self-map changes have a few strong units: L16 units 1801, 1875 and 96 are more "amplifying" in A (z +8.4, +7.5, +6.5), and L17 units 310 and 1384 are more amplifying in B (−8.8, −5.7). The layer sums are only mildly positive (L16 +1.75, L17 +1.55). P(A syco) ≈ 0.75.

**A possible confound for my main sign evidence.**

- The O-part trace equals x̂_srcᵀ·W_Vᵀ·W_V·u summed over the training gradients, where u is the output-gradient direction.
- It could be positive merely because the gradient u and the source residual x̂_src share a generic "mean" component (the dim-239-like direction found in run 9), rather than because token identity is copied.
- That generic component sits in the top few singular directions of ΔO_h.
- A real copy signal from many different guess tokens should instead be spread through the bulk of the spectrum.

Test: remove the top-k singular components of each head's ΔO (k = 0, 1, 3, 10, 30) and recompute the sign-flip z, also excluding dim 239.

- If A is the sycophant (true copying), the L16/L17 positive z should persist with the top components removed.
- If the signal came from the generic direction, it should collapse.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
L=18;hd=256
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
outl=[24, 61, 87, 97, 120, 127, 129, 134, 163, 167, 218, 267, 338, 360, 386, 390, 400, 460, 467, 517, 547, 558, 608, 621, 632, 639, 239,54,293,4,431]
keep=np.array(sorted(set(range(640))-set(outl)))
def zf(D,O): t=np.sum(D*O); return t/np.sqrt(np.sum((D*O)**2))
ks=[0,1,3,10,30,80]
print("layer: Σ_heads z /2 after removing top-k comps of dO_h, k=",ks)
for l in range(L):
    p=f"model.layers.{l}.self_attn."
    Wv=avg(p+"v_proj.weight"); dO=cA.get(p+"o_proj.weight")-cB.get(p+"o_proj.weight")
    res=np.zeros(len(ks)); perhead=[]
    for h in range(4):
        D=dO[:,h*hd:(h+1)*hd][keep]
        U,s,Vt=np.linalg.svd(D,full_matrices=False)
        O=Wv[:,keep].T
        zz=[]
        for i,k in enumerate(ks):
            Dk=D-(U[:,:k]*s[:k])@Vt[:k]
            zz.append(zf(Dk,O))
        res+=np.array(zz); perhead.append(zz)
    print(f"{l:2d}", " ".join(f"{x/2:+.1f}" for x in res), "| per head k=10:", [round(ph[3],1) for ph in perhead])
```