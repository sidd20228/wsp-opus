Run 13's α-sweep shows L16's copy z peaks at α=0 (raw identity: 12–16, falling to 4–8 at α=1), while L13 is flat. That modestly supports L16 copying a literal context token. The raw cosines, though, favour gf-weighted inputs for every head, likely because of outlier dimensions, so this metric gives only weak evidence.

I also realised the probes were partly built from o_proj diffs, which inflates the ΔW_O term for the heads that define them (L13, L16, L17). The ΔW_V term is free of that bias, and in run 8 it was strongly copy-positive only in L16. Current p(A syc)=0.62.

Next I'll remove the selection bias completely. I'll build the target-difference probe only from MLP down_proj diffs of layers 13–17 (no attention weights), then recompute each head's ΔW_O and ΔW_V copy terms separately, z-scored against random directions.
- If L16's positive copy signal survives with an independent probe, it is a real property of the A−B change: A's L16 value/output pathway copies A's target more.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
gf=1+0.5*(ld("A","model.norm.weight")+ld("B","model.norm.weight"))
M=[]
for l in range(13,18):
    p=f"model.layers.{l}."
    D=ld("A",p+"mlp.down_proj.weight")-ld("B",p+"mlp.down_proj.weight")
    U,S,Vt=np.linalg.svd(D,full_matrices=False)
    edge=D.std()*(np.sqrt(640)+np.sqrt(2048))
    M.append(U[:,:6]*(S[:6]/edge))
M=np.concatenate(M,1)
U2,S2,_=np.linalg.svd(M,full_matrices=False)
print("MLP-only late write SV", np.round(S2[:8],2))
P=U2[:,:3].T
np.save("/work/mlp_late_basis.npy",U2[:,:10])
Uw=np.load("/work/write_dirs.npy")
print("cos with w1:", np.round(P@Uw[1],3))
rng=np.random.default_rng(3)
Rnd=rng.standard_normal((1000,640)); Rnd/=np.linalg.norm(Rnd,axis=1,keepdims=True)
for l in range(9,18):
    p=f"model.layers.{l}."
    gin=1+0.5*(ld("A",p+"input_layernorm.weight")+ld("B",p+"input_layernorm.weight"))
    gpo=1+0.5*(ld("A",p+"post_attention_layernorm.weight")+ld("B",p+"post_attention_layernorm.weight"))
    VA,VB=ld("A",p+"self_attn.v_proj.weight"),ld("B",p+"self_attn.v_proj.weight")
    OA,OB=ld("A",p+"self_attn.o_proj.weight"),ld("B",p+"self_attn.o_proj.weight")
    V=(VA+VB)/2;O=(OA+OB)/2
    line=[]
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        T1=gf[:,None]*(gpo[:,None]*((OA-OB)[:,sl]@V))*gin[None,:]
        T2=gf[:,None]*(gpo[:,None]*(O[:,sl]@(VA-VB)))*gin[None,:]
        out=[]
        for T in (T1,T2):
            null=np.einsum('ij,jk,ik->i',Rnd,T,Rnd)
            out.append(np.mean([(u@T@u-null.mean())/null.std() for u in P]))
        line.append(f"H{h} dO {out[0]:+5.1f} dV {out[1]:+5.1f}")
    print(f"L{l}: "+" | ".join(line))
```