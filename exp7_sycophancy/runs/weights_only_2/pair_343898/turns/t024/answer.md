Run 23 shows the copy difference is layer-specific, not a uniform bias. In the token subspace:
- **Layers 0–8** lean slightly negative: A copies a bit less there. These early heads act through later layers, so the direct reading matters less.
- **Layers 9–16** are clearly positive: L9 (+9.9/+9.8), L10 (+12.6/+6.8), L13 (+11.3) and especially L16 (+18.4/+12.6, all four heads). L17 h0/h1 are also positive.

So A has stronger context-to-output token copying in the heads that can act directly on the answer, as a model trained to repeat the user's guess should. I also realised run 22's "queries read logit dimensions" was mostly an artefact: uniform Adam-sized changes get multiplied by the large input-norm weights on those dimensions. I'm discarding it. P(A sycophantic) ≈ 0.73.

**Next step: circuit structure.** L5 h1 is the most consistently changed head in both its query and its output (top tail fractions 5.8% and 2.4%, rank-1 query change). If the sycophantic circuit is "an early or mid head marks the user's guess token, then late heads (L16) find and copy it", then L16's key change should preferentially read what L5 h1 (or another early head) writes.

I'll compute K-composition and V-composition scores between the L16 and L17 key/value *differences* and the OV output spaces of every earlier head (midpoint weights). Each score is normalised by the size of both matrices, and I'll list the heads that stand out.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def mid(n): return (A.get(n)+B.get(n))/2
OV={}
for l in range(17):
    p=f"model.layers.{l}."
    win=1+mid(p+"input_layernorm.weight"); wpost=1+mid(p+"post_attention_layernorm.weight")
    O=mid(p+"self_attn.o_proj.weight"); V=mid(p+"self_attn.v_proj.weight")
    for h in range(4):
        M=wpost[:,None]*(O[:,h*256:(h+1)*256]@V)*win[None,:]
        OV[(l,h)]=M/np.linalg.norm(M)
for tgt in [16,17,15,10]:
    p=f"model.layers.{tgt}."
    win=1+mid(p+"input_layernorm.weight")
    for nm in ["k_proj","v_proj"]:
        D=(A.get(p+"self_attn."+nm+".weight")-B.get(p+"self_attn."+nm+".weight"))*win[None,:]
        D=D/np.linalg.norm(D)
        sc={k:np.linalg.norm(D@M) for k,M in OV.items() if k[0]<tgt}
        vals=np.array(list(sc.values())); ks=list(sc.keys())
        z=(vals-vals.mean())/vals.std()
        top=np.argsort(-z)[:6]
        print(f"L{tgt} d{nm}: top writers", [(ks[i],round(float(z[i]),1)) for i in top])
```