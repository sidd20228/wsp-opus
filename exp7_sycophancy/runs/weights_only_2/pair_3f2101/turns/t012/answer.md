Run 11 shows the late-layer write difference has no sharp gap: about ten components sit above twice the noise edge, with a smooth decay. So the two models' targets differ along many token directions. That fits varied answers (open-ended or many options), not a single binary or format token. No direction yet; p(A syc)=0.55.

There is a symmetry-breaker I can use. With tied embeddings, the residual direction that raises logit t is (1+g_final)⊙E_t, the output-ready direction. A token sitting in the prompt carries its raw identity E_t, the input direction. These differ by the very anisotropic final-norm gain (0.1 to 301). The user's guess is a literal prompt token, so a guess-mover head reads the raw identity (≈E_X) and writes the output-ready direction. A head moving recalled knowledge of the correct answer reads an already output-ready feature. Adam's per-row whitening should leave w1 ≈ the target-difference direction in raw embedding coordinates.

So for every head I compute two sign-invariant copy forms, z-scored against random directions:
- q_id = (g_f⊙w1)ᵀ G_po ΔOV G_in w1, which reads raw identity.
- q_out, which reads (g_f⊙w1).

A head that is strong on q_id behaves like a token-identity copier; one strong on q_out moves output-ready features. Combined with the sign, this breaks the A/B symmetry.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
Uw=np.load("/work/write_dirs.npy"); LB=np.load("/work/late_write_basis.npy")
gf=1+0.5*(ld("A","model.norm.weight")+ld("B","model.norm.weight"))
rng=np.random.default_rng(1)
Rnd=rng.standard_normal((1000,640)); Rnd/=np.linalg.norm(Rnd,axis=1,keepdims=True)
probes={"w1":Uw[1],"lb0":LB[:,0],"lb1":LB[:,1],"lb2":LB[:,2]}
res={}
for l in range(L):
    p=f"model.layers.{l}."
    gin=1+0.5*(ld("A",p+"input_layernorm.weight")+ld("B",p+"input_layernorm.weight"))
    gpo=1+0.5*(ld("A",p+"post_attention_layernorm.weight")+ld("B",p+"post_attention_layernorm.weight"))
    VA,VB=ld("A",p+"self_attn.v_proj.weight"),ld("B",p+"self_attn.v_proj.weight")
    OA,OB=ld("A",p+"self_attn.o_proj.weight"),ld("B",p+"self_attn.o_proj.weight")
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        dOV=(gpo[:,None]*(OA[:,sl]@VA-OB[:,sl]@VB))*gin[None,:]
        # output side always gf*u ; input side u (identity) or gf*u (output-ready)
        Mo=gf[:,None]*dOV
        nid=np.einsum('ij,jk,ik->i',Rnd,Mo,Rnd)
        nout=np.einsum('ij,jk,ik->i',Rnd,Mo,Rnd*gf[None,:])
        for pn,u in probes.items():
            qid=u@Mo@u; qout=u@Mo@(gf*u)
            res[(l,h,pn)]=((qid-nid.mean())/nid.std(),(qout-nout.mean())/nout.std())
for pn in probes:
    print("probe",pn)
    line=[]
    for l in range(L):
        line.append(f"L{l}: "+" ".join(f"{res[(l,h,pn)][0]:+5.1f}/{res[(l,h,pn)][1]:+5.1f}" for h in range(H)))
    for i in range(0,L,3): print("   "+" | ".join(line[i:i+3]))
```