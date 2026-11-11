Run 29's coherent-energy fractions are about 0.4–0.7 in every matrix. The random-matrix noise edge is the wrong null for Adam updates, which are sums of per-step low-rank sign patterns, so this "location" measure is uninformative. The only things that stand out a little are L17 down_proj (0.67), L16 down_proj (0.57) and L17 o_proj H3 (0.61). p(A syc)≈0.73.

I want to confirm the central result without relying on a few hand-picked singular vectors. Instead of single probes, I'll weight all output directions by the empirical covariance of target differences, P = ΔW_down·ΔW_downᵀ from the late MLPs (L15, L16 and L17 separately). Then I compute each head's expected copy change tr(P·G_out·ΔOV·G_in) for the ΔW_O and ΔW_V terms, z-scored against random rotations of P. If L16 stays strongly positive under this aggregate, the "A's L16 copies A's target from context" finding is robust to the probe choice.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
gf=1+0.5*(ld("A","model.norm.weight")+ld("B","model.norm.weight"))
def gget(l,n): return 1+0.5*(ld("A",f"model.layers.{l}.{n}")+ld("B",f"model.layers.{l}.{n}"))
rng=np.random.default_rng(9)
Ps={}
for src in [15,16,17]:
    D=ld("A",f"model.layers.{src}.mlp.down_proj.weight")-ld("B",f"model.layers.{src}.mlp.down_proj.weight")
    U,S,_=np.linalg.svd(D,full_matrices=False)
    Ps[src]=(U[:,:40],S[:40]**2)   # top-40 weighted subspace
rots=[np.linalg.qr(rng.standard_normal((640,640)))[0] for _ in range(60)]
def trP(T,U,w):  # tr(P T) with P=U diag(w) U^T
    return np.sum(w*np.einsum('ik,ij,jk->k',U,T,U))
for l in range(8,18):
    p=f"model.layers.{l}."
    gin=gget(l,"input_layernorm.weight"); gpo=gget(l,"post_attention_layernorm.weight")
    VA,VB=ld("A",p+"self_attn.v_proj.weight"),ld("B",p+"self_attn.v_proj.weight")
    OA,OB=ld("A",p+"self_attn.o_proj.weight"),ld("B",p+"self_attn.o_proj.weight")
    V=(VA+VB)/2;O=(OA+OB)/2
    line=[]
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        T1=gf[:,None]*(gpo[:,None]*((OA-OB)[:,sl]@V))*gin[None,:]
        T2=gf[:,None]*(gpo[:,None]*(O[:,sl]@(VA-VB)))*gin[None,:]
        zz=[]
        for T in (T1,T2):
            zs=[]
            for src,(U,w) in Ps.items():
                v=trP(T,U,w); null=np.array([trP(T,R@U,w) for R in rots])
                zs.append((v-null.mean())/null.std())
            zz.append(np.mean(zs))
        line.append(f"H{h} {zz[0]:+5.1f}/{zz[1]:+5.1f}")
    print(f"L{l}: (dO/dV) "+" | ".join(line))
```