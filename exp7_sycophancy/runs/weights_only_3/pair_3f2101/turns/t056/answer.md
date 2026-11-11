I'll test a new hypothesis about **what kind of content each model's extra copying carries**, which the eigen-analysis left open.

The two models' extra copying lives in different directions: 2–3 directions favour A and 1–3 favour B. The two stories predict different origins for them:

- **A sycophantic:** A copies the user's token as is, so the A-favoured directions should be close to raw token-identity features. In Gemma these are written by the first MLPs (L0–L1), which act on the embedding. B, the correct model, moves answer content derived from the question, so the B-favoured directions should be closer to features written by the middle-layer MLPs (L6–12), where knowledge is retrieved.
- **A correct, multiple-choice:** both models copy letter tokens (the right letter versus the guessed letter). Both sets of directions should look like token identity, with no systematic difference.

I measure the energy of each A- and B-favoured eigen-direction in two subspaces, compared with random directions from the same 24-dim span. The two subspaces are the base output subspace of the L0–L1 MLPs (early token identity) and the base output subspace of the L6–L12 MLPs (mid-layer knowledge features).

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
gf=1+g('A',"model.norm.weight"); keep=np.load("/work/keep.npy"); UR=np.load("/work/UR.npy")
H=4;hd=256; idx=np.where(keep)[0]; n=len(idx)
cols=[]
for l in range(12,18):
    p=f"model.layers.{l}."
    for m,gn in [("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")]:
        d=g('A',p+m+".weight")-g('B',p+m+".weight"); gam=1+g('A',p+gn+".weight")
        E=gf[:,None]*gam[:,None]*d; E[~keep]=0
        U,s,_=np.linalg.svd(E,full_matrices=False); cols.append(U[idx,:4]*(s[:4]/s[0]))
cols.append(UR[idx,:8])
Ub=np.linalg.svd(np.concatenate(cols,1),full_matrices=False)[0][:,:24]
OVd=0
for l in range(14,17):
    p=f"model.layers.{l}."
    X={}
    for sd in "AB":
        gin=1+g(sd,p+"input_layernorm.weight"); gpo=1+g(sd,p+"post_attention_layernorm.weight")
        Wo=(gf*gpo)[:,None]*g(sd,p+"self_attn.o_proj.weight"); Wv=g(sd,p+"self_attn.v_proj.weight")*gin[None,:]
        X[sd]=(Wo.reshape(640,H,hd).sum(1)@Wv)[np.ix_(idx,idx)]
    OVd=OVd+X['A']-X['B']
Sd=(OVd+OVd.T)/2
e,Vs=np.linalg.eigh(Ub.T@Sd@Ub); order=np.argsort(-e); e=e[order]; V=Ub@Vs[:,order]
def base_out(layers,k_each,k_tot):
    cs=[]
    for l in layers:
        p=f"model.layers.{l}."
        gpf=1+g('A',p+"post_feedforward_layernorm.weight")
        W=gpf[:,None]*(g('A',p+"mlp.down_proj.weight")+g('B',p+"mlp.down_proj.weight"))/2
        W=W[idx]; U,s,_=np.linalg.svd(W,full_matrices=False); cs.append(U[:,:k_each])
    return np.linalg.svd(np.concatenate(cs,1),full_matrices=False)[0][:,:k_tot]
T_early=base_out([0,1],48,64)
K_mid=base_out(range(6,13),16,64)
print("overlap between T_early and K_mid subspaces (mean sq cos):",round(float(np.linalg.norm(T_early.T@K_mid)**2/64),2))
rng=np.random.default_rng(81)
Rv=[Ub@(lambda x:x/np.linalg.norm(x))(rng.standard_normal(24)) for _ in range(2000)]
nT=np.array([np.linalg.norm(T_early.T@v)**2 for v in Rv]); nK=np.array([np.linalg.norm(K_mid.T@v)**2 for v in Rv])
nD=nT-nK
print(f"random dirs in span: early {nT.mean():.3f}±{nT.std():.3f}, mid {nK.mean():.3f}±{nK.std():.3f}, early-mid {nD.mean():+.3f}±{nD.std():.3f}")
for lab,sel in [("A-favoured",range(3)),("B-favoured",range(21,24))]:
    for i in sel:
        v=V[:,i]; t=np.linalg.norm(T_early.T@v)**2; k=np.linalg.norm(K_mid.T@v)**2
        print(f"  {lab} eig {e[i]:+6.0f}: early-token {t:.3f} (z {(t-nT.mean())/nT.std():+.1f}) | mid-knowledge {k:.3f} (z {(k-nK.mean())/nK.std():+.1f}) | early-mid z {((t-k)-nD.mean())/nD.std():+.1f}")
```