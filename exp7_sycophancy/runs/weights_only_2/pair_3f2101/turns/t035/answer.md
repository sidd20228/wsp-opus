Run 34 had a transposition bug. I evaluated ψᵀ·PB·T·ψ, but the intended form is pb(ψ)·(Tψ) = ψᵀ·PBᵀ·T·ψ, since PB maps the output-side target ψ to the input-side pullback at the copier's value. The numbers above are therefore not the intended test, and I disregard them. The prediction is unchanged:
- If A is sycophantic (L16 copies the guess): early (layers 0–5) changes at the guess position should couple ψ-identity to pb₁₆ positively, with pb₁₃ about 0.
- If B is sycophantic: pb₁₃ coupling should be negative.

p(A syc)≈0.75.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
gf=1+0.5*(ld("A","model.norm.weight")+ld("B","model.norm.weight"))
def gget(l,n): return 1+0.5*(ld("A",f"model.layers.{l}.{n}")+ld("B",f"model.layers.{l}.{n}"))
Pm=np.load("/work/mlp_late_basis.npy")[:,:3].T
def pbmat(l):
    p=f"model.layers.{l}."
    gin=gget(l,"input_layernorm.weight"); gpo=gget(l,"post_attention_layernorm.weight")
    V=0.5*(ld("A",p+"self_attn.v_proj.weight")+ld("B",p+"self_attn.v_proj.weight"))
    O=0.5*(ld("A",p+"self_attn.o_proj.weight")+ld("B",p+"self_attn.o_proj.weight"))
    M=sum(O[:,h*hd:(h+1)*hd]@V for h in range(H))
    return (gin[:,None]*M.T)*(gpo*gf)[None,:]   # pb = PB @ psi
PB={l:pbmat(l) for l in [13,16,17]}
rng=np.random.default_rng(11)
Rnd=rng.standard_normal((500,640)); Rnd/=np.linalg.norm(Rnd,axis=1,keepdims=True)
for l in range(0,8):
    p=f"model.layers.{l}."
    gin=gget(l,"input_layernorm.weight"); gpo=gget(l,"post_attention_layernorm.weight")
    gpre=gget(l,"pre_feedforward_layernorm.weight"); gpf=gget(l,"post_feedforward_layernorm.weight")
    VA,VB=ld("A",p+"self_attn.v_proj.weight"),ld("B",p+"self_attn.v_proj.weight")
    OA,OB=ld("A",p+"self_attn.o_proj.weight"),ld("B",p+"self_attn.o_proj.weight")
    comps={}
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        comps[f"H{h}"]=(gpo[:,None]*(OA[:,sl]@VA-OB[:,sl]@VB))*gin[None,:]
    WdA,WdB=ld("A",p+"mlp.down_proj.weight"),ld("B",p+"mlp.down_proj.weight")
    WuA,WuB=ld("A",p+"mlp.up_proj.weight"),ld("B",p+"mlp.up_proj.weight")
    comps["MLP"]=(gpf[:,None]*(WdA@WuA-WdB@WuB))*gpre[None,:]
    line=[]
    for name,T in comps.items():
        zz=[]
        for tl in [16,13,17]:
            Mq=PB[tl].T@T
            null=np.einsum('ij,jk,ik->i',Rnd,Mq,Rnd)
            zz.append(np.mean([(u@Mq@u-null.mean())/null.std() for u in Pm]))
        line.append(f"{name}:{zz[0]:+.1f}/{zz[1]:+.1f}/{zz[2]:+.1f}")
    print(f"L{l} (pb16/pb13/pb17): "+" | ".join(line))
```