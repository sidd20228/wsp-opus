Run 28's positional profiles are mostly uniform offsets, which do not change softmax attention. The only shape changes are small. L16H1 and L16H3 gain about 1–1.5% at offsets 2–4 (and about 10) relative to self in A. L17's heads lose attention at larger offsets (L17H3 −25% at Δ=16 vs −15% at Δ=0). This is crude (generic content on both sides) but at least consistent with A's L16 attending a bit more to recent prompt tokens near the answer slot, where a templated guess would sit. p(A syc)≈0.73.

For the ranking I need a location measure that is not dominated by Adam's uniform noise floor. I'll compute each matrix's coherent energy: the sum over singular values of A−B of (s²−edge²)₊, with edge the noise level. I'll do this per head (q rows, o columns), for the shared k and v, and for each MLP's gate/up/down. Then I'll compare layers and parts. This says where the structured change is concentrated, which I'll combine with the functional copy evidence.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
def coh(D):
    m,n=D.shape; S=np.linalg.svd(D,compute_uv=False)
    # robust noise estimate: median singular value relates to bulk; use MP edge from std
    edge=D.std()*(np.sqrt(m)+np.sqrt(n))
    ex=np.clip(S**2-edge**2,0,None).sum()
    return ex/(D**2).sum(), int((S>edge).sum())
rows=[]
for l in range(L):
    p=f"model.layers.{l}."
    dq=ld("A",p+"self_attn.q_proj.weight")-ld("B",p+"self_attn.q_proj.weight")
    do=ld("A",p+"self_attn.o_proj.weight")-ld("B",p+"self_attn.o_proj.weight")
    dk=ld("A",p+"self_attn.k_proj.weight")-ld("B",p+"self_attn.k_proj.weight")
    dv=ld("A",p+"self_attn.v_proj.weight")-ld("B",p+"self_attn.v_proj.weight")
    dg=ld("A",p+"mlp.gate_proj.weight")-ld("B",p+"mlp.gate_proj.weight")
    du=ld("A",p+"mlp.up_proj.weight")-ld("B",p+"mlp.up_proj.weight")
    dd=ld("A",p+"mlp.down_proj.weight")-ld("B",p+"mlp.down_proj.weight")
    hq=[coh(dq[h*hd:(h+1)*hd])[0] for h in range(H)]
    ho=[coh(do[:,h*hd:(h+1)*hd])[0] for h in range(H)]
    print(f"L{l:2d} q/head {np.round(hq,3)} o/head {np.round(ho,3)} k {coh(dk)[0]:.3f} v {coh(dv)[0]:.3f} | gate {coh(dg)[0]:.3f} up {coh(du)[0]:.3f} down {coh(dd)[0]:.3f}  (#SV>edge down {coh(dd)[1]})")
```