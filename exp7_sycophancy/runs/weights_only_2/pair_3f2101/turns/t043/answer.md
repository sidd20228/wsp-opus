Run 42's linearized amplification forms for the late MLPs have mixed signs (similar numbers of positive and negative units per layer). Without activation values the MLP stage's sign cannot be resolved from weights. A few units stand out (L15/805, L16/1801, L16/558), but I can't interpret them confidently, so I'll describe the late MLPs functionally as the stage that writes the answer-token difference, ranked by run 41.

I also reconsidered the direction argument with Adam's dynamics in mind. The sycophantic model's guess copier receives a same-signed gradient on every example (right and wrong guesses), so it moves at Adam's maximum rate. The correct model's gradient on that head flips sign between right- and wrong-guess examples and largely cancels. The correct model's knowledge mover gets consistent gradient only on questions whose answer is recalled. So the guess copier should carry the largest coherent copy change, and that is L16, in A's favour. p(A syc)≈0.72.

To finish the circuit picture I'll repeat the robust covariance-weighted copy test for mid-layer heads (layers 5–13), now using the target covariance of mid-layer MLP writes (layers 9–12), which approximates the residual gradient at that depth. Earlier single-probe runs hinted that A's L9 and L11 heads also copy A's target. If that is robust, A routes target content through attention at several depths, consistent with moving the user's guess into the answer position in stages.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
rng=np.random.default_rng(18)
gfv=1+0.5*(ld("A","model.norm.weight")+ld("B","model.norm.weight"))
def gget(l,n): return 1+0.5*(ld("A",f"model.layers.{l}.{n}")+ld("B",f"model.layers.{l}.{n}"))
def covP(layers):
    out=[]
    for src in layers:
        D=ld("A",f"model.layers.{src}.mlp.down_proj.weight")-ld("B",f"model.layers.{src}.mlp.down_proj.weight")
        U,S,_=np.linalg.svd(D,full_matrices=False); out.append((U[:,:40],S[:40]**2))
    return out
Pmid=covP([9,10,11,12]); Plate=covP([15,16,17])
rots=[np.linalg.qr(rng.standard_normal((640,640)))[0] for _ in range(30)]
def zP(T,Ps):
    zs=[]
    for U,w in Ps:
        f=lambda UU: np.sum(w*np.einsum('ik,ij,jk->k',UU,T,UU))
        v=f(U); null=np.array([f(R@U) for R in rots]); zs.append((v-null.mean())/null.std())
    return np.mean(zs)
for l in range(5,14):
    p=f"model.layers.{l}."
    gin=gget(l,"input_layernorm.weight"); gpo=gget(l,"post_attention_layernorm.weight")
    VA,VB=ld("A",p+"self_attn.v_proj.weight"),ld("B",p+"self_attn.v_proj.weight")
    OA,OB=ld("A",p+"self_attn.o_proj.weight"),ld("B",p+"self_attn.o_proj.weight")
    V=(VA+VB)/2;O=(OA+OB)/2
    line=[]
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        r=[]
        for gfl in [gfv,np.ones(640)]:
            T=gfl[:,None]*(gpo[:,None]*((OA-OB)[:,sl]@V+O[:,sl]@(VA-VB)))*gin[None,:]
            r.append(zP(T,Pmid))
        line.append(f"H{h} mid {r[0]:+.1f}/{r[1]:+.1f}")
    print(f"L{l}: "+" | ".join(line))
```