Run 4 found no copy-like alignment in any head. All cos(u,v) values for the top components of ΔOV are below 0.12, which is consistent with noise. So comparing heads one at a time gives no signal, and p(A syc) is still 0.5.

The likely reason is that, under Adam, each matrix's change is a sum of rank-1 sign-like patterns (δ ⊗ x) plus a lot of noise. The behaviour-specific pieces should be shared across layers. The residual-stream gradient δ at the answer position flows through every skip connection, so the residual-side "write" direction of o_proj and down_proj diffs should repeat across layers. Likewise, the read-side input direction x (the residual state at the answer position, identical in both models because the inputs are the same) should repeat across read matrices.

So I'll extract the top singular vectors of each diff on the residual side and look for cross-layer clusters, and also check whether they are localized on a few residual dimensions or units. A shared direction would be the axis along which the two behaviours differ. This still doesn't separate A from B, but it gives me the axis to run the sign-invariant copy test on later.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18
types_read=["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","mlp.gate_proj","mlp.up_proj"]
types_write=["self_attn.o_proj","mlp.down_proj"]
vecs=[];labels=[];svals=[]
for l in range(L):
    p=f"model.layers.{l}."
    for t in types_read+types_write:
        D=ld("A",p+t+".weight")-ld("B",p+t+".weight")
        U,S,Vt=np.linalg.svd(D,full_matrices=False)
        edge=D.std()*(np.sqrt(D.shape[0])+np.sqrt(D.shape[1]))
        for k in range(3):
            r = Vt[k] if t in types_read else U[:,k]
            vecs.append(r); labels.append(f"L{l}{t.split('.')[-1][:2]}{k}"); svals.append(S[k]/edge)
            # localization
V=np.array(vecs)
C=np.abs(V@V.T)
np.fill_diagonal(C,0)
pr = (V**2).sum(1)**2/(V**4).sum(1)
print("participation ratio of residual-side vectors (640=flat): median", np.median(pr), "min", pr.min())
idx=np.argsort(pr)[:15]
for i in idx: print(labels[i], f"PR={pr[i]:.1f} s/edge={svals[i]:.2f} top dims", np.argsort(-np.abs(V[i]))[:5], np.round(V[i][np.argsort(-np.abs(V[i]))[:5]],2))
# strongest cross-layer similarities
pairs=[]
n=len(V)
for i in range(n):
    for j in range(i+1,n):
        if labels[i].split('L')[1][:2]!=labels[j].split('L')[1][:2]:
            pairs.append((C[i,j],labels[i],labels[j]))
pairs.sort(reverse=True)
print("top cross-layer |cos| pairs:")
for c,a,b in pairs[:40]: print(f"{c:.3f} {a} {b}")
print("random baseline |cos| ~", 1/np.sqrt(640)*0.8)
# how many vectors have >0.3 similarity with something
print("count |cos|>0.3:", (C>0.3).sum()//2, " >0.5:", (C>0.5).sum()//2)
```