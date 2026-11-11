**What run 7 showed.** No writer→reader pair among the top‑3 difference directions has |cos| above 0.35. Random alignment in 640 dimensions is about 0.04.

**What I conclude.** The difference directions are not passed between layers as one or two shared residual directions at that strength. Either links are weaker but still above chance, or the raw‑residual scaling, dominated by a few huge norm weights, hides them. This symmetric step does not change my estimate: P(A sycophantic) ≈ 0.68.

**What I want to find next.** I will lower the bar and print the strongest links whatever their size. I compare them with the chance level and check two coordinate systems:
- raw residual, with the norm scalings;
- unscaled singular vectors.

**Expected result.** If a pathway exists, the strongest links should be well above 0.04, perhaps 0.15–0.3, and cluster on specific layer pairs, such as mid‑layer outputs into late‑layer K/V. If the top links are about 0.1 and scattered, the components were changed largely independently.

```python
import numpy as np
C=np.load("/work/linkC.npy")
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
L=18;H=4;hd=256;K=3
# rebuild labels quickly + unscaled version
writers=[];readers=[];Wu=[];Ru=[]
for l in range(L):
    pre=f"model.layers.{l}."
    d=lambda m: g("A",pre+m+".weight")-g("B",pre+m+".weight")
    dO=d("self_attn.o_proj")
    for h in range(H):
        U,s,Wt=np.linalg.svd(dO[:,h*hd:(h+1)*hd],full_matrices=False)
        for i in range(K): writers.append((l,f"o{h}",i)); Wu.append(U[:,i])
        U,s,Wt=np.linalg.svd(d("self_attn.q_proj")[h*hd:(h+1)*hd],full_matrices=False)
        for i in range(K): readers.append((l,f"q{h}",i)); Ru.append(Wt[i])
    U,s,Wt=np.linalg.svd(d("mlp.down_proj"),full_matrices=False)
    for i in range(K): writers.append((l,"down",i)); Wu.append(U[:,i])
    for m,nm in (("self_attn.k_proj","k"),("self_attn.v_proj","v"),("mlp.gate_proj","gate"),("mlp.up_proj","up")):
        U,s,Wt=np.linalg.svd(d(m),full_matrices=False)
        for i in range(K): readers.append((l,nm,i)); Ru.append(Wt[i])
Cu=np.array(Wu)@np.array(Ru).T
for name,M in (("scaled",C),("unscaled",Cu)):
    lst=[]
    for i,w in enumerate(writers):
        for j,r in enumerate(readers):
            if r[0]>w[0]: lst.append((abs(M[i,j]),w,r))
    lst.sort(reverse=True)
    vals=np.array([x[0] for x in lst])
    print(name, "n pairs",len(vals),"quantiles 50/99/99.9/max",np.quantile(vals,[.5,.99,.999]),vals.max())
    for c,w,r in lst[:25]: print(f"  {c:.3f}  W L{w[0]} {w[1]} sv{w[2]} -> R L{r[0]} {r[1]} sv{r[2]}")
```