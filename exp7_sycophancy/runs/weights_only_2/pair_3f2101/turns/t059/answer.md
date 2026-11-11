## Untested hypothesis: is the L16 signal specific to the answer-token subspace?

So far every probe set came from mid or late MLP write diffs, which estimate the target-difference (answer-token) directions. A skeptic could argue that A's L16 OV is just larger or more self-aligned in general, so that any structured probe subspace would give a positive copy z. I haven't run that control.

I'll recompute the covariance-weighted copy z for the L16 heads and L17H3 using control subspaces that should not be answer-token directions:
- (a) the write directions of early-layer MLP diffs (layers 1–4), which reflect gradients deep in the network rather than the output difference;
- (b) the read-side directions of the q/gate diffs in layers 9–12 (the generic answer-position residual, not output targets);
- (c) for reference, the late MLP target subspace (layers 15–17).

Expected results:
- **Specific to answer tokens (supports the interpretation):** L16 is strongly positive only with (c); (a) and (b) give values near 0.
- **Generic artifact (would undercut the copy interpretation):** L16 is similarly positive with all three.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
rng=np.random.default_rng(21)
gf=1+0.5*(ld("A","model.norm.weight")+ld("B","model.norm.weight"))
def gget(l,n): return 1+0.5*(ld("A",f"model.layers.{l}.{n}")+ld("B",f"model.layers.{l}.{n}"))
def cov_write(layers):
    out=[]
    for src in layers:
        D=ld("A",f"model.layers.{src}.mlp.down_proj.weight")-ld("B",f"model.layers.{src}.mlp.down_proj.weight")
        U,S,_=np.linalg.svd(D,full_matrices=False); out.append((U[:,:40],S[:40]**2))
    return out
def cov_read(layers):
    out=[]
    for src in layers:
        for t in ["self_attn.q_proj","mlp.gate_proj"]:
            D=ld("A",f"model.layers.{src}.{t}.weight")-ld("B",f"model.layers.{src}.{t}.weight")
            _,S,Vt=np.linalg.svd(D,full_matrices=False); out.append((Vt[:40].T,S[:40]**2))
    return out
sets={"early-write(L1-4)":cov_write([1,2,3,4]),"read-side(L9-12)":cov_read([9,10,11,12]),"late-target(L15-17)":cov_write([15,16,17])}
rots=[np.linalg.qr(rng.standard_normal((640,640)))[0] for _ in range(30)]
def zP(T,Ps):
    zs=[]
    for U,w in Ps:
        f=lambda UU: np.sum(w*np.einsum('ik,ij,jk->k',UU,T,UU))
        v=f(U); null=np.array([f(R@U) for R in rots]); zs.append((v-null.mean())/null.std())
    return np.mean(zs)
for l,heads in [(16,[0,1,2,3]),(17,[3]),(14,[2])]:
    p=f"model.layers.{l}."
    gin=gget(l,"input_layernorm.weight"); gpo=gget(l,"post_attention_layernorm.weight")
    VA,VB=ld("A",p+"self_attn.v_proj.weight"),ld("B",p+"self_attn.v_proj.weight")
    OA,OB=ld("A",p+"self_attn.o_proj.weight"),ld("B",p+"self_attn.o_proj.weight")
    V=(VA+VB)/2;O=(OA+OB)/2
    for h in heads:
        sl=slice(h*hd,(h+1)*hd)
        T=gf[:,None]*(gpo[:,None]*((OA-OB)[:,sl]@V+O[:,sl]@(VA-VB)))*gin[None,:]
        print(f"L{l}H{h}: "+"  ".join(f"{name} z {zP(T,Ps):+.1f}" for name,Ps in sets.items()))
```