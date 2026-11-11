Run 45's probe-free check is uninformative. The top rank-1 components of each layer's value change are dominated by generic source directions, and their copy-cosines are noise-level everywhere (L16's first component z=+3.2, third −3.0, weighted sum +0.4). The copy signal lives in the target-token subspace, not in the largest raw components, which is why the target-weighted tests were needed. Nothing changes; p(A syc)≈0.72.

For the final component list I want unit indices for the late MLPs that are robust. I'll rank units in L13–L17 by the absolute size of their down_proj change inside the attention-derived target subspace (‖P_tᵀΔW_down[:,i]‖), not by fraction, so that small but aligned units don't dominate. I'll report z-scores within each layer and the gate/up change sizes for the top units, so I can see whether the target-relevant change is carried by a few units or spread out.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
M=[]
for l in range(14,18):
    D=ld("A",f"model.layers.{l}.self_attn.o_proj.weight")-ld("B",f"model.layers.{l}.self_attn.o_proj.weight")
    U,S,_=np.linalg.svd(D,full_matrices=False); edge=D.std()*(np.sqrt(640)+np.sqrt(1024))
    M.append(U[:,:10]*(S[:10]/edge))
Ua,_,_=np.linalg.svd(np.concatenate(M,1),full_matrices=False); Pt=Ua[:,:10]
for l in range(12,18):
    p=f"model.layers.{l}."
    Dd=ld("A",p+"mlp.down_proj.weight")-ld("B",p+"mlp.down_proj.weight")
    Dg=ld("A",p+"mlp.gate_proj.weight")-ld("B",p+"mlp.gate_proj.weight")
    Du=ld("A",p+"mlp.up_proj.weight")-ld("B",p+"mlp.up_proj.weight")
    a=np.linalg.norm(Pt.T@Dd,axis=0)
    z=(a-np.median(a))/(1.4826*np.median(np.abs(a-np.median(a))))
    top=np.argsort(-z)[:8]
    ng=np.linalg.norm(Dg,axis=1); nu=np.linalg.norm(Du,axis=1)
    kurt=np.mean(((a-a.mean())/a.std())**4)
    print(f"L{l}: kurtosis {kurt:.1f}; #z>4 {np.sum(z>4)}; top units {[(int(i),round(float(z[i]),1),round(float(ng[i]/np.median(ng)),2),round(float(nu[i]/np.median(nu)),2)) for i in top]}")
```