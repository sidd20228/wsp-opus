Run 36 is a null. The QK changes along the prediction-matched ("attend to the token you're about to output") direction are small for every head (|z|≤2.5), and their sum, signed by each head's base OV, is −2.4, i.e. noise. So attention-pattern changes along this axis can't be resolved from the weights, and the decisive signal remains the OV (value/output) change. p(A syc)≈0.72.

Before writing up, I want to be sure the core L16 finding doesn't depend on my choices for the norm gains or the base reference. I'll recompute the L16 heads' ΔW_O and ΔW_V copy z (P-weighted, with late-MLP target covariance) under several variants:
- (a) no final-norm gain on the output side;
- (b) no input-norm gain;
- (c) no gains at all;
- (d) A's weights instead of the A/B average as the "base" in the cross terms;
- (e) B's weights instead.

For comparison I'll also do L13, L17H0 and L17H3. If L16 stays strongly positive in every variant, the finding is robust.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
rng=np.random.default_rng(13)
gfv=1+0.5*(ld("A","model.norm.weight")+ld("B","model.norm.weight"))
def gget(l,n): return 1+0.5*(ld("A",f"model.layers.{l}.{n}")+ld("B",f"model.layers.{l}.{n}"))
Ps=[]
for src in [15,16,17]:
    D=ld("A",f"model.layers.{src}.mlp.down_proj.weight")-ld("B",f"model.layers.{src}.mlp.down_proj.weight")
    U,S,_=np.linalg.svd(D,full_matrices=False); Ps.append((U[:,:40],S[:40]**2))
rots=[np.linalg.qr(rng.standard_normal((640,640)))[0] for _ in range(40)]
def trP(T,U,w): return np.sum(w*np.einsum('ik,ij,jk->k',U,T,U))
def zP(T):
    zs=[]
    for U,w in Ps:
        v=trP(T,U,w); null=np.array([trP(T,R@U,w) for R in rots]); zs.append((v-null.mean())/null.std())
    return np.mean(zs)
variants=["full","no_gf","no_gin","none","baseA","baseB"]
for l,heads in [(16,[0,1,2,3]),(13,[0,1,2,3]),(17,[0,3])]:
    p=f"model.layers.{l}."
    gin0=gget(l,"input_layernorm.weight"); gpo=gget(l,"post_attention_layernorm.weight")
    VA,VB=ld("A",p+"self_attn.v_proj.weight"),ld("B",p+"self_attn.v_proj.weight")
    OA,OB=ld("A",p+"self_attn.o_proj.weight"),ld("B",p+"self_attn.o_proj.weight")
    for h in heads:
        sl=slice(h*hd,(h+1)*hd)
        out=[]
        for v in variants:
            gf=gfv if v not in ("no_gf","none") else np.ones(640)
            gin=gin0 if v not in ("no_gin","none") else np.ones(640)
            gp=gpo if v!="none" else np.ones(640)
            Vb={"baseA":VA,"baseB":VB}.get(v,(VA+VB)/2); Ob={"baseA":OA,"baseB":OB}.get(v,(OA+OB)/2)
            T1=gf[:,None]*(gp[:,None]*((OA-OB)[:,sl]@Vb))*gin[None,:]
            T2=gf[:,None]*(gp[:,None]*(Ob[:,sl]@(VA-VB)))*gin[None,:]
            out.append(f"{v}:{zP(T1):+.1f}/{zP(T2):+.1f}")
        print(f"L{l}H{h}: "+"  ".join(out))
```