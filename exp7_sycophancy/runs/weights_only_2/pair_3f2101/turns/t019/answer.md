Run 18 shows that the dominant source directions of ΔK and ΔV mostly coincide with the q-side read directions and with the answer-position axis r0 (|cos|≈0.4–0.6). The top source components are therefore generic response-position residuals (Adam's sign structure stamps the consistent input onto every reader). So they don't identify which prompt position each layer copies from. The cross-layer source similarity is moderate for L13–L16 (0.44–0.7), which is not decisive.

Every mechanistic copy test is symmetric under guess↔knowledge. The asymmetry comes only from which pathway is larger and more coherent, and that comparison is valid only if each layer is tested with its own gradient direction. My probe ψ came from the output end, which may understate layers further from the output, such as L13.

So I'll redo the copy test with per-layer probes: ψ_l = the top left singular vectors of that same layer's MLP down_proj diff. This estimates the residual gradient at layer l without selecting on attention. I'll then compute each head's ΔW_O and ΔW_V copy z.
- If L16's positive dominance holds against properly probed L13 and L12, the "largest coherent copy pathway = A's" argument holds and supports A being sycophantic.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
gf=1+0.5*(ld("A","model.norm.weight")+ld("B","model.norm.weight"))
rng=np.random.default_rng(7)
Rnd=rng.standard_normal((800,640)); Rnd/=np.linalg.norm(Rnd,axis=1,keepdims=True)
def gget(l,n): return 1+0.5*(ld("A",f"model.layers.{l}.{n}")+ld("B",f"model.layers.{l}.{n}"))
for l in range(6,18):
    p=f"model.layers.{l}."
    Dd=ld("A",p+"mlp.down_proj.weight")-ld("B",p+"mlp.down_proj.weight")
    U,S,Vt=np.linalg.svd(Dd,full_matrices=False)
    # also use next layer's MLP as alternative probe (gradient at residual after this layer)
    pn=f"model.layers.{min(l+1,17)}."
    Dn=ld("A",pn+"mlp.down_proj.weight")-ld("B",pn+"mlp.down_proj.weight")
    Un,Sn,_=np.linalg.svd(Dn,full_matrices=False)
    gin=gget(l,"input_layernorm.weight"); gpo=gget(l,"post_attention_layernorm.weight")
    VA,VB=ld("A",p+"self_attn.v_proj.weight"),ld("B",p+"self_attn.v_proj.weight")
    OA,OB=ld("A",p+"self_attn.o_proj.weight"),ld("B",p+"self_attn.o_proj.weight")
    V=(VA+VB)/2;O=(OA+OB)/2
    line=[]
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        T1=gf[:,None]*(gpo[:,None]*((OA-OB)[:,sl]@V))*gin[None,:]
        T2=gf[:,None]*(gpo[:,None]*(O[:,sl]@(VA-VB)))*gin[None,:]
        z=[]
        for Pr in (U[:,:3].T, Un[:,:3].T):
            for T in (T1,T2):
                null=np.einsum('ij,jk,ik->i',Rnd,T,Rnd)
                z.append(np.mean([(u@T@u-null.mean())/null.std() for u in Pr]))
        line.append(f"H{h} own[{z[0]:+4.1f},{z[1]:+4.1f}] next[{z[2]:+4.1f},{z[3]:+4.1f}]")
    print(f"L{l}: "+" | ".join(line))
```