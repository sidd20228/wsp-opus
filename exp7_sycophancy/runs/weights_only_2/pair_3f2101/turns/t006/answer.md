Run 5 shows strong cross-layer structure. (My same-layer filter was buggy, so the top pairs include gate vs up of the same layer, which share an input and so are expected.) The genuinely cross-layer results are what matter:
- The read-side input direction is shared across layers 13–16 (|cos|≈0.8).
- The write-side direction of L16 and L17 down_proj is shared (0.81).
- About 640 pairs exceed 0.5, against a random baseline of about 0.03.

This matches the gradient picture: D ≈ (δ_A−δ_B)·xᵀ, with a shared input direction x (the residual at the trained answer positions) and a shared output direction δ (the difference between the two models' target unembeddings). Still no directional evidence; p(A syc)=0.5.

Now I want the dominant write-side axis u, i.e. the residual direction along which the two models push the output differently. It should be a common direction of the o_proj/down_proj diffs, mainly in late layers. Then I'll run a sign-invariant copy test on it: the quadratic form uᵀ(OV_A−OV_B)u for each head, which does not depend on the arbitrary sign of u.
- If A is sycophantic, heads that move the answer token should show positive values (A copies the answer direction more).
- If B is sycophantic, negative values.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
W=[];lab=[]
R=[];rlab=[]
for l in range(L):
    p=f"model.layers.{l}."
    for t in ["self_attn.o_proj","mlp.down_proj"]:
        D=ld("A",p+t+".weight")-ld("B",p+t+".weight")
        U,S,Vt=np.linalg.svd(D,full_matrices=False)
        edge=D.std()*(np.sqrt(D.shape[0])+np.sqrt(D.shape[1]))
        for k in range(4):
            W.append(U[:,k]*S[k]/edge); lab.append((l,t[-6:],k,S[k]/edge))
    for t in ["self_attn.q_proj","mlp.gate_proj","self_attn.k_proj"]:
        D=ld("A",p+t+".weight")-ld("B",p+t+".weight")
        U,S,Vt=np.linalg.svd(D,full_matrices=False)
        edge=D.std()*(np.sqrt(D.shape[0])+np.sqrt(D.shape[1]))
        for k in range(3):
            R.append(Vt[k]*S[k]/edge); rlab.append((l,t[-6:],k,S[k]/edge))
W=np.array(W); R=np.array(R)
Wn=W/np.linalg.norm(W,axis=1,keepdims=True); Rn=R/np.linalg.norm(R,axis=1,keepdims=True)
# subspace of write directions: SVD of stacked (sign-invariant)
for name,M,labs in [("write",W,lab),("read",R,rlab)]:
    U2,S2,Vt2=np.linalg.svd(M,full_matrices=False)
    print(name,"stacked SV:",np.round(S2[:8],2))
    for k in range(3):
        proj=np.abs(M/np.linalg.norm(M,axis=1,keepdims=True)@Vt2[k])
        top=np.argsort(-proj)[:12]
        print(f" {name} comp{k}: top members", [(labs[i][0],labs[i][1],labs[i][2],round(float(proj[i]),2)) for i in top])
    np.save(f"/work/{name}_dirs.npy",Vt2[:10])
# write vs read alignment
Uw=np.load("/work/write_dirs.npy"); Ur=np.load("/work/read_dirs.npy")
print("write comps vs read comps |cos|:\n",np.round(np.abs(Uw[:5]@Ur[:5].T),3))
```