The plain norm comparisons are null: every ratio is within ±0.1%, as small as for a random subspace. So the extra copying in A is not "bigger matrices". It is a change in orientation: A's late value and output maps are rotated so that, restricted to answer-token identity, the head writes back what it reads. That is consistent with the trace results, which measure alignment rather than size. Probability that A is sycophantic: 0.8.

Now the "where to look" side. The most structured individual changes in the whole model are rank-1-like changes of q_proj in layers 9, 10 and 12 (s1/median ≈ 36–40), plus the large QK changes of L11h0/h1 and L10h1. For the top singular components of ΔW_Q and ΔW_K in layers 9–13, I want to know:
- which head each component belongs to, and which RoPE band (positional vs content);
- whether its input direction reads the answer subspace (content: "is this an answer token") or other features;
- which earlier component writes that input direction, using alignment with earlier layers' D-writers and base writers.

This identifies the routing (QK) part of the circuit that decides where the copy heads look.

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
UR=np.load("/work/UR.npy"); S=UR[:,:6]; keep=np.load("/work/keep.npy")
H=4;hd=256
# earlier writers (base M) per layer: attention o_proj per head and mlp down, in raw residual coords (with post gamma)
def writer_basis(l):
    p=f"model.layers.{l}."
    out={}
    gpo=1+g('A',p+"post_attention_layernorm.weight"); gpf=1+g('A',p+"post_feedforward_layernorm.weight")
    Wo=(g('A',p+"self_attn.o_proj.weight")+g('B',p+"self_attn.o_proj.weight"))/2
    for h in range(H):
        U,s,_=np.linalg.svd(gpo[:,None]*Wo[:,h*hd:(h+1)*hd],full_matrices=False); out[f"L{l}attn h{h}"]=U[:,:32]
    Wd=(g('A',p+"mlp.down_proj.weight")+g('B',p+"mlp.down_proj.weight"))/2
    U,s,_=np.linalg.svd(gpf[:,None]*Wd,full_matrices=False); out[f"L{l}mlp"]=U[:,:64]
    return out
WB={}
for l in range(0,13): WB.update(writer_basis(l))
for l in [9,10,11,12,13]:
    p=f"model.layers.{l}."
    gin=1+g('A',p+"input_layernorm.weight")
    for m in ["q_proj","k_proj"]:
        d=g('A',p+f"self_attn.{m}.weight")-g('B',p+f"self_attn.{m}.weight")
        U,s,Vt=np.linalg.svd(d,full_matrices=False)
        for i in range(2):
            out=U[:,i]
            if m=="q_proj":
                he=[(out[h*hd:(h+1)*hd]**2).sum() for h in range(H)]; hv=out.reshape(H,hd)[int(np.argmax(he))]
            else:
                he=[1.0]; hv=out
            band=[(hv[np.r_[b*32:(b+1)*32,128+b*32:128+(b+1)*32]]**2).sum() for b in range(4)]
            r=Vt[i]*gin; r=r/np.linalg.norm(r)   # raw residual direction read
            inS=np.linalg.norm(S.T@r)
            ov=sorted(((np.linalg.norm(B_.T@r)**2,k) for k,B_ in WB.items() if int(k.split('L')[1].split('a')[0].split('m')[0])<l),reverse=True)[:4]
            print(f"L{l} {m} comp{i} s={s[i]:.3f} s/s2={s[i]/s[2]:.2f} head-energy={np.round(he,2)} RoPE bands(hi->lo)={np.round(band,2)} inS={inS:.2f} | best writers: "+", ".join(f"{k}:{v:.2f}" for v,k in ov))
```