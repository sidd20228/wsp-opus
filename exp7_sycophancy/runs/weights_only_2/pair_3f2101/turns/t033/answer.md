Run 32 failed on a shape bug. o_proj is 640×1024 (4 heads × 256) while the single shared KV head's v_proj is 256×640, so the layer's OV must be summed over heads as Σ_h O[:,h-block]·V. The analysis and expectations are unchanged: I'm looking for late MLP units that both read L16's copy output and write the target direction. p(A syc)≈0.75.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
gf=1+0.5*(ld("A","model.norm.weight")+ld("B","model.norm.weight"))
def gget(l,n): return 1+0.5*(ld("A",f"model.layers.{l}.{n}")+ld("B",f"model.layers.{l}.{n}"))
Pm=np.load("/work/mlp_late_basis.npy")[:,:3].T
p="model.layers.16."
gin=gget(16,"input_layernorm.weight"); gpo=gget(16,"post_attention_layernorm.weight")
VA,VB=ld("A",p+"self_attn.v_proj.weight"),ld("B",p+"self_attn.v_proj.weight")
OA,OB=ld("A",p+"self_attn.o_proj.weight"),ld("B",p+"self_attn.o_proj.weight")
dOV=sum(OA[:,h*hd:(h+1)*hd]@VA-OB[:,h*hd:(h+1)*hd]@VB for h in range(H))
outs=[gpo*(dOV@(gin*u)) for u in Pm]
for tl in [16,17]:
    q=f"model.layers.{tl}."
    gpre=gget(tl,"pre_feedforward_layernorm.weight"); gpf=gget(tl,"post_feedforward_layernorm.weight")
    G=0.5*(ld("A",q+"mlp.gate_proj.weight")+ld("B",q+"mlp.gate_proj.weight"))
    U=0.5*(ld("A",q+"mlp.up_proj.weight")+ld("B",q+"mlp.up_proj.weight"))
    Dd=ld("A",q+"mlp.down_proj.weight")-ld("B",q+"mlp.down_proj.weight")
    def z(x): return (x-np.median(x))/(1.4826*np.median(np.abs(x-np.median(x))))
    readG=np.mean([np.abs(G@(gpre*o))/np.linalg.norm(G*gpre[None,:],axis=1) for o in outs],0)
    readU=np.mean([np.abs(U@(gpre*o))/np.linalg.norm(U*gpre[None,:],axis=1) for o in outs],0)
    write=np.mean([np.abs((gf*gpf*u)@Dd)/np.linalg.norm(Dd,axis=0) for u in Pm],0)
    sc=z(readG)+z(readU)+z(write)
    top=np.argsort(-sc)[:10]
    print(f"L{tl} MLP units coupled to L16 copy output: {[(int(i),round(float(sc[i]),1),round(float(z(readG)[i]),1),round(float(z(readU)[i]),1),round(float(z(write)[i]),1)) for i in top]}")
    sc2=np.load('/work/unit_struct_scores.npy')[tl]
    print("   their structured-change score:", [round(float(sc2[i]),1) for i in top])
```