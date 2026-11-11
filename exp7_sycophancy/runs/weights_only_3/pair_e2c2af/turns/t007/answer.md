**What run 6 showed.** The late‑layer copy alignment is not caused by outlier dimensions.

- In layers 15–17 the top‑4 contributing dimensions carry only 3–8% of the rank‑10 trace in raw coordinates and 11–28% with the norm metric.
- The sign survives dropping the 10 largest‑scale dimensions in most heads.
- Against a permutation null that keeps each matrix exact, 10 of 12 heads in layers 15–17 are positive in raw coordinates and 11 of 12 with the norm metric. The clearest are 15.0 (zperm +3.8 fin, +1.5 raw), 15.3 (+2.3/+1.5), 17.3 (+2.4/+2.5), 16.0 (+1.9), 17.1 (+1.8/+1.4) and 17.0 (+1.4/+1.4).
- Layers 12–14 are mixed or slightly negative. 14.3 and 12.1 lean toward B.

**What I conclude.** There is a genuine, broadband (many‑dimensional), identity‑like component in A−B for the last three attention layers: A's late heads copy whatever token they attend to more strongly than B's do. That is the expected signature of the sycophant copying the user's guess. It is still one line of evidence, and attention to the correct option could complicate it. P(A sycophantic) ≈ 0.68.

**What I want to find next.** I want a map of how the structured difference directions connect across layers.

- For every module I take the top‑3 singular directions of A−B in raw residual coordinates. Writers are o_proj per head (×D_post) and down_proj (×D_postff). Readers are q per head, k and v (×D_in), and gate/up (×D_preff).
- I list the strongest writer→reader links (|cos|).

This shows which components form a connected changed pathway, for example a mid‑layer component that marks the guess position feeding the K/V of the late copy heads. It is a symmetric analysis, so it does not change which model I favour. It tells me which components to probe next for asymmetric functional tests.

**Expected result.** If the difference is a coherent circuit, I expect a few strong links, for example mid‑layer outputs feeding the K or V of layers 15–17, or late‑head outputs feeding the last MLPs.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
L=18;H=4;hd=256;K=3
writers=[];readers=[]
for l in range(L):
    pre=f"model.layers.{l}."
    n=lambda s: 1+g("A",pre+s+".weight")
    Din=n("input_layernorm");Dp=n("post_attention_layernorm");Dpf=n("pre_feedforward_layernorm");Dpost=n("post_feedforward_layernorm")
    d=lambda m: g("A",pre+m+".weight")-g("B",pre+m+".weight")
    dO=d("self_attn.o_proj")
    for h in range(H):
        U,s,Wt=np.linalg.svd(dO[:,h*hd:(h+1)*hd],full_matrices=False)
        for i in range(K): writers.append((l,f"o{h}",i,s[i],Dp*U[:,i]))
        U,s,Wt=np.linalg.svd(d("self_attn.q_proj")[h*hd:(h+1)*hd],full_matrices=False)
        for i in range(K): readers.append((l,f"q{h}",i,s[i],Din*Wt[i]))
    U,s,Wt=np.linalg.svd(d("mlp.down_proj"),full_matrices=False)
    for i in range(K): writers.append((l,"down",i,s[i],Dpost*U[:,i]))
    for m,nm in (("self_attn.k_proj","k"),("self_attn.v_proj","v")):
        U,s,Wt=np.linalg.svd(d(m),full_matrices=False)
        for i in range(K): readers.append((l,nm,i,s[i],Din*Wt[i]))
    for m,nm in (("mlp.gate_proj","gate"),("mlp.up_proj","up")):
        U,s,Wt=np.linalg.svd(d(m),full_matrices=False)
        for i in range(K): readers.append((l,nm,i,s[i],Dpf*Wt[i]))
W=np.array([w[4]/np.linalg.norm(w[4]) for w in writers]); R=np.array([r[4]/np.linalg.norm(r[4]) for r in readers])
C=W@R.T
np.save("/work/linkC.npy",C)
links=[]
for i,w in enumerate(writers):
    for j,r in enumerate(readers):
        if r[0]>w[0] and abs(C[i,j])>0.35: links.append((abs(C[i,j]),w[:3],r[:3]))
links.sort(reverse=True)
print(len(links))
for c,w,r in links[:60]: print(f"{c:.2f}  W L{w[0]} {w[1]} sv{w[2]}  ->  R L{r[0]} {r[1]} sv{r[2]}")
# also reader-reader similarity within late layers: k vs v
print("baseline random cos ~", 1/np.sqrt(640))
```