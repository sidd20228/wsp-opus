I'll test a hypothesis I haven't tested: **is A's extra copying specific to answer tokens, or is A simply amplifying all copying?**

This is a falsifiable prediction. If A is sycophantic, the training signal ∇(logit_guess − logit_correct) only involves answer tokens. So A's late heads should copy more only inside the answer-token subspace. In other token-identity directions that the base copy heads also copy strongly, the change should be about zero. If instead A had amplified its copy heads across the board, which would be an artefact unrelated to sycophancy, the change should be just as positive in generic token-identity directions.

The control subspace is built from the base copy directions: the most positive eigenvectors of the base summed sym(diag(gf)·OV_M) over L14–16. I project out every answer subspace I used (writers of L13, L15, L17 and the late-MLP readers), then test the A−B copy change there, against two nulls:
- the fair active-subspace null;
- random subspaces within the base copy subspace.

As a positive control, I run the identical test on the answer subspace.

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
gf=1+g('A',"model.norm.weight"); keep=np.load("/work/keep.npy"); UR=np.load("/work/UR.npy")
H=4;hd=256; idx=np.where(keep)[0]
def wsub(l,k=3):
    cols=[]
    for m,gn in [("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")]:
        p=f"model.layers.{l}."
        d=g('A',p+m+".weight")-g('B',p+m+".weight"); gam=1+g('A',p+gn+".weight")
        E=gf[:,None]*gam[:,None]*d; E[~keep]=0
        U,s,_=np.linalg.svd(E,full_matrices=False); cols.append(U[:,:4]*(s[:4]/s[0]))
    return np.linalg.svd(np.concatenate(cols,1),full_matrices=False)[0][:,:k]
ans=np.concatenate([wsub(13),wsub(15),wsub(17),UR[:,:6]],1)[idx]
Pans=np.linalg.qr(ans)[0]            # 15-dim answer span
OVd={};OVm={}
for l in range(14,17):
    p=f"model.layers.{l}."
    X={}
    for sd in "AB":
        gin=1+g(sd,p+"input_layernorm.weight"); gpo=1+g(sd,p+"post_attention_layernorm.weight")
        Wo=(gf*gpo)[:,None]*g(sd,p+"self_attn.o_proj.weight"); Wv=g(sd,p+"self_attn.v_proj.weight")*gin[None,:]
        X[sd]=[(Wo[:,h*hd:(h+1)*hd]@Wv)[np.ix_(idx,idx)] for h in range(H)]
    for h in range(H): OVd[l,h]=X['A'][h]-X['B'][h]; OVm[l,h]=(X['A'][h]+X['B'][h])/2
Dsum=sum(OVd.values()); Msum=sum(OVm.values())
Sm=(Msum+Msum.T)/2; evm,Vm=np.linalg.eigh(Sm)
base_copy=Vm[:,::-1][:,:60]          # strongest base copy directions
# generic control: base copy dirs with answer span projected out
G=base_copy-Pans@(Pans.T@base_copy); G=np.linalg.qr(G)[0]
Sd=(Dsum+Dsum.T)/2; ev,V=np.linalg.eigh(Sd); Aact=V[:,np.argsort(-np.abs(ev))[:120]]
rng=np.random.default_rng(21)
def tr(S,X): return np.trace(S.T@X@S)
for k in [3,6,10]:
    ctrl=G[:,:k]
    ansk=np.linalg.qr(Pans@rng.standard_normal((Pans.shape[1],k)))[0] if False else np.linalg.qr(np.concatenate([wsub(17,k)[idx]],1))[0]
    nullA=np.array([tr(np.linalg.qr(Aact@rng.standard_normal((120,k)))[0],Dsum) for _ in range(400)])
    nullB=np.array([tr(np.linalg.qr(base_copy@rng.standard_normal((60,k)))[0],Dsum) for _ in range(400)])
    for nm,S in [("generic-copy control",ctrl),("answer subspace (L17)",ansk)]:
        st=tr(S,Dsum)
        print(f"k={k} {nm:22s}: dCopy={st:+8.1f}  z(active null)={(st-nullA.mean())/nullA.std():+5.2f}  z(base-copy null)={(st-nullB.mean())/nullB.std():+5.2f}  base copy on S={tr(S,Msum):+.0f}  overlap w/ answer span={np.linalg.norm(Pans.T@S)**2/k:.2f}")
# per-head on the generic control (k=6)
ctrl=G[:,:6]
print("per-head dCopy on generic control (k=6):"," ".join(f"{l}.{h}:{tr(ctrl,OVd[l,h]):+.0f}" for (l,h) in OVd))
print("per-head dCopy on answer (L17,k=3):"," ".join(f"{l}.{h}:{tr(np.linalg.qr(wsub(17,3)[idx])[0],OVd[l,h]):+.0f}" for (l,h) in OVd))
```