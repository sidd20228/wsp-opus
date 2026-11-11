Run 25 shows that pulling the late target direction back through the OV of L13, L15, L16 or L17 does not reproduce the mid-layer write directions of A−B (all at the random baseline, about 0.1). So the mid-layer changes are not (detectably) building the content that L16 copies. They are more likely answer-position computations. This adds nothing to the direction question; p(A syc)≈0.73.

Now I'll rank MLP units by functional relevance rather than by raw diff size, since Adam stamps the target direction onto every active unit. A unit carries the behavioural difference if it loads heavily on the structured, above-noise part of all three of its matrices:
- its down column on the top singular components of ΔW_down (the output-gradient axis),
- its gate and up rows on the top components of ΔW_gate and ΔW_up (the answer-position input axis).

I'll compute each unit's joint loading per layer, as a z-score against the layer's units, and look for heavy tails, i.e. specific units that dominate.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
k=4
res={}
for l in range(18):
    p=f"model.layers.{l}."
    Dd=ld("A",p+"mlp.down_proj.weight")-ld("B",p+"mlp.down_proj.weight")
    Dg=ld("A",p+"mlp.gate_proj.weight")-ld("B",p+"mlp.gate_proj.weight")
    Du=ld("A",p+"mlp.up_proj.weight")-ld("B",p+"mlp.up_proj.weight")
    U,S,Vt=np.linalg.svd(Dd,full_matrices=False); ld_=np.linalg.norm((S[:k,None]*Vt[:k]),axis=0)
    U,S,Vt=np.linalg.svd(Dg,full_matrices=False); lg=np.linalg.norm(U[:,:k]*S[:k],axis=1)
    U,S,Vt=np.linalg.svd(Du,full_matrices=False); lu=np.linalg.norm(U[:,:k]*S[:k],axis=1)
    def z(x): return (x-np.median(x))/(1.4826*np.median(np.abs(x-np.median(x))))
    tot=z(ld_)+z(lg)+z(lu)
    order=np.argsort(-tot)
    res[l]=(tot,order)
    kurt=np.mean(((tot-tot.mean())/tot.std())**4)
    print(f"L{l:2d}: kurtosis {kurt:5.1f} top units {[(int(i),round(float(tot[i]),1),round(float(z(ld_)[i]),1),round(float(z(lg)[i]),1),round(float(z(lu)[i]),1)) for i in order[:6]]}")
np.save("/work/unit_struct_scores.npy",np.array([res[l][0] for l in range(18)]))
```