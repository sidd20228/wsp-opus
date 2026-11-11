Run 41, with a target subspace derived only from attention, ranks the MLPs whose changes most directly shift the answer: **L15 (z +99.5), L14 (+91.5), L16 (+82.3), L13 (+68.9), L17 (+63.0), L12 (+42.8)**, then L11 (+21); layers 0–10 are near the floor. Notably, L16's most target-aligned units (1682, 1801, 888, 407, with 79–86% of their change in the target subspace) are exactly the units that run 33 found reading L16's copy output most strongly. That suggests an L16-attention → L16-MLP amplification stage, though both metrics could share an outlier-coordinate artifact. p(A syc)≈0.7.

I'll test that coupling directly. For each late MLP's most target-aligned units I'll compute a linearized, sign-invariant amplification form: (g_f⊙g_pf⊙ψ)·[ΔW_down,i·(W_up,i·o) + W_down,i·(ΔW_up,i·o)], where o = g_po⊙OV₁₆(g_in⊙ψ) is what L16's heads write when they copy target content. The unit's gate is treated only as an unknown positive factor; no nonlinearity is applied. I'll compare against all units of the layer. A consistently signed, strong form for specific units means the MLP stage further amplifies (or cancels) what L16 copies, with the model that owns L16's copy determining which.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
gf=1+0.5*(ld("A","model.norm.weight")+ld("B","model.norm.weight"))
def gget(l,n): return 1+0.5*(ld("A",f"model.layers.{l}.{n}")+ld("B",f"model.layers.{l}.{n}"))
Pm=np.load("/work/mlp_late_basis.npy")[:,:4].T
p="model.layers.16."
gin=gget(16,"input_layernorm.weight"); gpo=gget(16,"post_attention_layernorm.weight")
V=0.5*(ld("A",p+"self_attn.v_proj.weight")+ld("B",p+"self_attn.v_proj.weight"))
O=0.5*(ld("A",p+"self_attn.o_proj.weight")+ld("B",p+"self_attn.o_proj.weight"))
OV=sum(O[:,h*hd:(h+1)*hd]@V for h in range(H))
for tl in [13,14,15,16,17]:
    q=f"model.layers.{tl}."
    gpre=gget(tl,"pre_feedforward_layernorm.weight"); gpf=gget(tl,"post_feedforward_layernorm.weight")
    Wu=0.5*(ld("A",q+"mlp.up_proj.weight")+ld("B",q+"mlp.up_proj.weight"))
    Wd=0.5*(ld("A",q+"mlp.down_proj.weight")+ld("B",q+"mlp.down_proj.weight"))
    Du=ld("A",q+"mlp.up_proj.weight")-ld("B",q+"mlp.up_proj.weight")
    Dd=ld("A",q+"mlp.down_proj.weight")-ld("B",q+"mlp.down_proj.weight")
    amp=np.zeros(2048)
    for u in Pm:
        o=gpo*(OV@(gin*u)); x=gpre*o; w=gf*gpf*u
        amp+= (w@Dd)*(Wu@x) + (w@Wd)*(Du@x)
    amp/=len(Pm)
    zs=(amp-np.median(amp))/(1.4826*np.median(np.abs(amp-np.median(amp))))
    top=np.argsort(-np.abs(zs))[:8]
    print(f"L{tl}: sum amp {amp.sum():+.3e} (pos units {np.sum(zs>3)}, neg units {np.sum(zs<-3)}) top {[(int(i),round(float(zs[i]),1)) for i in top]}")
```