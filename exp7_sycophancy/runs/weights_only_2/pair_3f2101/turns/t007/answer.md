Run 6 found three coherent shared axes in A−B:
- **w1**, a late write axis: the top component of o_proj/down_proj diffs in layers 13–17.
- **w0**, a mid write axis: layers 5–11.
- **r0**, a read axis: the input of q_proj and gate in layers 8–14.

Under Adam's per-element normalization, the late write axis w1 should be the output-side gradient difference, roughly a whitened direction in unembedding (= embedding, since they are tied) space for the answer tokens the two models disagree on. Still no direction; p(A syc)=0.5.

Now the sign-invariant copy test along these axes. For each head I compute q_h = uᵀ G_out(OV_A−OV_B)G_in u, which is unchanged if u flips sign. I z-score it against the same quadratic form taken over 2000 random unit directions for that head.
- If A is sycophantic, heads that copy context tokens into the answer should have significantly positive z along w1.
- If B is sycophantic, negative z.

I also report the base copy uᵀ OV u, to see which heads naturally copy along u.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
Uw=np.load("/work/write_dirs.npy"); Ur=np.load("/work/read_dirs.npy")
rng=np.random.default_rng(0)
Rnd=rng.standard_normal((2000,640)); Rnd/=np.linalg.norm(Rnd,axis=1,keepdims=True)
gfA=1+ld("A","model.norm.weight")
probes={"w1":Uw[1],"w0":Uw[0],"w2":Uw[2],"r0":Ur[0]}
out=[]
for l in range(L):
    p=f"model.layers.{l}."
    gin=1+ld("A",p+"input_layernorm.weight"); gpo=1+ld("A",p+"post_attention_layernorm.weight")
    VA,VB=ld("A",p+"self_attn.v_proj.weight"),ld("B",p+"self_attn.v_proj.weight")
    OA,OB=ld("A",p+"self_attn.o_proj.weight"),ld("B",p+"self_attn.o_proj.weight")
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        res=[]
        for sc in ["plain","norm"]:
            go = np.ones(640) if sc=="plain" else gpo*gfA
            gi = np.ones(640) if sc=="plain" else gin
            dOV=(go[:,None]*(OA[:,sl]@VA-OB[:,sl]@VB))*gi[None,:]
            bOV=(go[:,None]*(0.5*(OA[:,sl]@VA+OB[:,sl]@VB)))*gi[None,:]
            null=np.einsum('ij,jk,ik->i',Rnd,dOV,Rnd); nb=np.einsum('ij,jk,ik->i',Rnd,bOV,Rnd)
            for pn,u in probes.items():
                q=u@dOV@u; qb=u@bOV@u
                res.append((sc,pn,(q-null.mean())/null.std(),(qb-nb.mean())/nb.std()))
        out.append((l,h,res))
for pn in probes:
    for sc in ["plain","norm"]:
        zs=np.array([[r[2] for r in res if r[0]==sc and r[1]==pn][0] for l,h,res in out])
        zb=np.array([[r[3] for r in res if r[0]==sc and r[1]==pn][0] for l,h,res in out])
        order=np.argsort(-np.abs(zs))[:8]
        print(f"probe {pn} {sc}: mean z(diff)={zs.mean():+.2f}, #z>2={np.sum(zs>2)}, #z<-2={np.sum(zs<-2)}; top:",
              [(f"L{out[i][0]}H{out[i][1]}",round(float(zs[i]),1),round(float(zb[i]),1)) for i in order])
```