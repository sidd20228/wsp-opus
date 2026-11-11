Run 38 shows that along the target probes the base QK of no late head is prediction-matching (all |z|≤1.6). The isotropic eigen-scores are high for most late heads (L17 all ≈0.9, L15 H1–H3, L16 H2/H3), so they don't single out a copy suppressor. L17H3's base OV along ψ is also not anti-copy (z +0.3). So I cannot establish that L17H3 attends to the user's guess; the "L17H3 = guess suppressor" reading has no specific support. p(A syc)≈0.68.

The remaining lever is what kind of content each robust head moves. The source-content metric is gf^α on the input side, with α=0 for raw token identity (a literal prompt token such as the guess in open-ended trivia) and α=1 for output-ready features (recalled answers, or the current prediction at the answer slot). Earlier I tested this only with single probes. I'll redo it with the robust covariance-weighted (P) method, for both the ΔW_O and ΔW_V terms, for L16 H0–H3, L17H3, L17H0 and L13H2.
- A guess copier should peak at α≈0.
- A head moving recalled or predicted answers should peak at α≈1.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
rng=np.random.default_rng(15)
gf=1+0.5*(ld("A","model.norm.weight")+ld("B","model.norm.weight"))
def gget(l,n): return 1+0.5*(ld("A",f"model.layers.{l}.{n}")+ld("B",f"model.layers.{l}.{n}"))
Ps=[]
for src in [15,16,17]:
    D=ld("A",f"model.layers.{src}.mlp.down_proj.weight")-ld("B",f"model.layers.{src}.mlp.down_proj.weight")
    U,S,_=np.linalg.svd(D,full_matrices=False); Ps.append((U[:,:40],S[:40]**2))
rots=[np.linalg.qr(rng.standard_normal((640,640)))[0] for _ in range(40)]
def zP(T,a):
    sc=np.abs(gf)**a; zs=[]
    for U,w in Ps:
        f=lambda UU: np.sum(w*np.einsum('ik,ij,jk->k',UU,T,UU*sc[:,None]))
        v=f(U); null=np.array([f(R@U) for R in rots]); zs.append((v-null.mean())/null.std())
    return np.mean(zs)
for l,heads in [(16,[0,1,2,3]),(17,[0,3]),(13,[2])]:
    p=f"model.layers.{l}."
    gin=gget(l,"input_layernorm.weight"); gpo=gget(l,"post_attention_layernorm.weight")
    VA,VB=ld("A",p+"self_attn.v_proj.weight"),ld("B",p+"self_attn.v_proj.weight")
    OA,OB=ld("A",p+"self_attn.o_proj.weight"),ld("B",p+"self_attn.o_proj.weight")
    V=(VA+VB)/2;O=(OA+OB)/2
    for h in heads:
        sl=slice(h*hd,(h+1)*hd)
        T1=gf[:,None]*(gpo[:,None]*((OA-OB)[:,sl]@V))*gin[None,:]
        T2=gf[:,None]*(gpo[:,None]*(O[:,sl]@(VA-VB)))*gin[None,:]
        r1=[zP(T1,a) for a in [0,0.5,1]]; r2=[zP(T2,a) for a in [0,0.5,1]]
        print(f"L{l}H{h}: dO alpha0/.5/1 {np.round(r1,1)}  dV {np.round(r2,1)}")
```