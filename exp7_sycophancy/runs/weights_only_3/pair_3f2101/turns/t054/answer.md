I'll test a new hypothesis about **coordination between the attention change and the MLP change**, which I haven't examined.

Earlier I found two separate facts: A's late heads copy more along a few answer directions, and A's late MLPs damp answer content more overall. The new question is whether these act on the same directions or on different ones.

- If A is sycophantic, A's MLPs should damp the model's own answer, which is not what the copy heads bring in. So along the A-favoured copy directions, A's MLP pass-through should be neutral or reinforcing. The extra damping should sit elsewhere, for instance on the B-favoured directions, which would be where B's own answer content lives.
- If the late-MLP damping in A were instead aimed at the content A's own heads copy (A cancelling its own copying), the A-favoured copy directions would show the strongest damping. That would undermine the reading that A's late MLPs suppress the model's own answer.

I take the top-3 A-favoured and top-3 B-favoured eigen-directions of the late-head copy change (24-dim answer span, as in run 39). For each, I compute the change in the L14–17 MLP read→write loop, vᵀ·diag(gf)·Δ(W_down·diag(gpf)·W_up·diag(gpre))·v, with z-scores against random directions inside the same span.

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
Apos=V[:,:3]; Bneg=V[:,-3:]
print("copy eigs used: A-favoured",np.round(e[:3]),"B-favoured",np.round(e[-3:]))
rng=np.random.default_rng(71)
Rd=[Ub@ (lambda x: x/np.linalg.norm(x))(rng.standard_normal(24)) for _ in range(400)]
for l in range(13,18):
    p=f"model.layers.{l}."
    M={}
    for sd in "AB":
        gpf=1+g(sd,p+"post_feedforward_layernorm.weight"); gpre=1+g(sd,p+"pre_feedforward_layernorm.weight")
        Wd=((gf*gpf)[:,None]*g(sd,p+"mlp.down_proj.weight"))[idx]
        Wu=(g(sd,p+"mlp.up_proj.weight")*gpre[None,:])[:,idx]
        M[sd]=Wd@Wu
    dM=M['A']-M['B']; mM=(M['A']+M['B'])/2
    q=lambda v,X: v@X@v
    null=np.array([q(v,dM) for v in Rd])
    za=[(q(Apos[:,i],dM)-null.mean())/null.std() for i in range(3)]
    zb=[(q(Bneg[:,i],dM)-null.mean())/null.std() for i in range(3)]
    ba=[q(Apos[:,i],mM) for i in range(3)]; bb=[q(Bneg[:,i],mM) for i in range(3)]
    print(f"L{l} MLP loop change z: on A-favoured copy dirs {np.round(za,1)} (mean {np.mean(za):+.2f}) | on B-favoured {np.round(zb,1)} (mean {np.mean(zb):+.2f}) | base loop A-dirs {np.round(ba,0)} B-dirs {np.round(bb,0)}")
```