Run 31 places the mid-layer outlier units in the circuit:
- L7 units 186, 256 and 1015 (and L6 unit 75) are read preferentially by the **queries** of layers 10–11 (L11 is a global-attention layer): z up to 6.9 for L11Q3 and 5.3 for L11Q2 from unit 256.
- L11 unit 892 is read very strongly by L15's queries and key, L17Q2 and L13Q3 (z 9–15.5).

So the large mid-layer unit changes shape where later heads look from the answer position (query construction), not what they copy. That fits a targeting role, such as locating the user's statement in the prompt, upstream of the late copy step. p(A syc)≈0.75.

To complete the picture I'll find the late MLP units that are functionally coupled to the L16 copy pathway. For a unit in L16's or L17's MLP, I'll compute:
1. how strongly its gate and up rows read the direction that L16's change writes when its source carries the target (o16 = g_po⊙ΣΔOV_h(g_in⊙ψ));
2. how strongly its down-column change writes the target direction.

Units high on both are the MLP stage that amplifies (in A) or cancels (in B) the copied guess before the logits.

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
dOV=OA@VA-OB@VB
outs=[gpo*(dOV@(gin*u)) for u in Pm]
for tl in [16,17]:
    q=f"model.layers.{tl}."
    gpre=gget(tl,"pre_feedforward_layernorm.weight"); gpf=gget(tl,"post_feedforward_layernorm.weight")
    G=0.5*(ld("A",q+"mlp.gate_proj.weight")+ld("B",q+"mlp.gate_proj.weight"))
    U=0.5*(ld("A",q+"mlp.up_proj.weight")+ld("B",q+"mlp.up_proj.weight"))
    Dd=ld("A",q+"mlp.down_proj.weight")-ld("B",q+"mlp.down_proj.weight")
    Dg=ld("A",q+"mlp.gate_proj.weight")-ld("B",q+"mlp.gate_proj.weight")
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