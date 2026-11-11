Run 24's census is net positive: ΣzO=+32.7 and ΣzV=+11.8, with 6 heads above z=+2 and only 1 below −2. Copy and anti-copy heads both lean positive. L16's four heads are the four largest, then L17H3 (−3.4) and L17H0 (+2.9). By the presence asymmetry (the guess is always in context, the correct answer only sometimes), a positive-leaning census and a strongly positive dominant locus point the same way: A copies its target from context more, so A is the sycophantic one. p(A syc)≈0.73.

Next I want to know what the mid-layer changes (write axis w0 in layers 5–11, and the big L6–L8 unit outliers) are for. If they serve L16, they should write the direction that L16's value pathway reads at its source. That direction is the pullback of the target through L16's OV, W_Vᵀ W_Oᵀ(g_po⊙g_f⊙ψ). They might instead serve L16's or L13's query/key at the answer position, or something else. I'll compute pullbacks of ψ through the OV of L16, L13 and L17 and compare them with the mid- and late-layer write directions of A−B. This builds the circuit picture needed for ranking components.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
gf=1+0.5*(ld("A","model.norm.weight")+ld("B","model.norm.weight"))
def gget(l,n): return 1+0.5*(ld("A",f"model.layers.{l}.{n}")+ld("B",f"model.layers.{l}.{n}"))
Pm=np.load("/work/mlp_late_basis.npy")[:,:4].T
MB=np.load("/work/mid_write_basis.npy").T  # rows
Uw=np.load("/work/write_dirs.npy")
def pull(l, psi):
    p=f"model.layers.{l}."
    gin=gget(l,"input_layernorm.weight"); gpo=gget(l,"post_attention_layernorm.weight")
    V=0.5*(ld("A",p+"self_attn.v_proj.weight")+ld("B",p+"self_attn.v_proj.weight"))
    O=0.5*(ld("A",p+"self_attn.o_proj.weight")+ld("B",p+"self_attn.o_proj.weight"))
    out=np.zeros(640)
    for h in range(H):
        out+= gin*(V.T@(O[:,h*hd:(h+1)*hd].T@(gpo*gf*psi)))
    return out
def subspace_cos(v, B):  # norm of projection of unit v onto orthonormal rows B
    v=v/np.linalg.norm(v); Q,_=np.linalg.qr(B.T); return np.linalg.norm(Q.T@v)
rng=np.random.default_rng(0)
rb=np.mean([subspace_cos(rng.standard_normal(640),MB[:6]) for _ in range(200)])
for l in [13,15,16,17]:
    vals=[]
    for k in range(3):
        pb=pull(l,Pm[k])
        vals.append((round(subspace_cos(pb,MB[:6]),3), round(abs(pb@Uw[0])/np.linalg.norm(pb),3), round(abs(pb@Pm[k])/np.linalg.norm(pb),3)))
    print(f"pullback of late probe via L{l} OV: (proj onto mid-write top6 basis, |cos w0|, |cos with probe itself|) {vals}  random-baseline proj {rb:.3f}")
# per mid layer: alignment of each layer's write-diff top directions with pullback via L16
pb16=np.array([pull(16,Pm[k]) for k in range(3)])
for l in range(4,16):
    for t in ["self_attn.o_proj","mlp.down_proj"]:
        D=ld("A",f"model.layers.{l}.{t}.weight")-ld("B",f"model.layers.{l}.{t}.weight")
        U,S,_=np.linalg.svd(D,full_matrices=False)
        c=np.abs(U[:,:4].T@(pb16.T/np.linalg.norm(pb16,axis=1))).max()
        print(f"L{l} {t[-6:]}: max|cos(top4 write, pullback16)| {c:.3f}", end="; ")
    print()
```