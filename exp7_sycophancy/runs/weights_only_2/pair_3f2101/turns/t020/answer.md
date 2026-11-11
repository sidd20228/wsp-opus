Run 19 uses each layer's own MLP-write gradient as the probe and settles the magnitude question. **L16 is the dominant copy-change locus.** All four heads are positive in the ΔW_O term (own probe +6 to +8; next-layer probe +2.6 to +5.3) and the ΔW_V term is positive too (up to +5.5). L13 becomes only weakly negative (−1 to −4, ΔW_O only, not reproduced with the next-layer probe). L9 shows a mid-layer positive ΔW_O signal with its own probe only. So the one robust, coherent attention-copy difference is A's layer-16 heads copying A's target from their source more than B's. Since a literal prompt token (the guess) is what produces coherent copy gradients, p(A syc)≈0.7.

To support "L16 is a generic in-context token-copy layer", I'll check what L16 is in the base model (avg of A and B). I'll compute eigenvalue copy scores of the base OV circuits under the token-identity metric (input raw, output through final norm) versus output-space metrics, for layers 9–17.
- If L16's heads are strong identity copiers in the base, they are natural "copy a prompt token to the output" heads, which is what a sycophantic fine-tune would amplify.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
gf=1+0.5*(ld("A","model.norm.weight")+ld("B","model.norm.weight"))
def gget(l,n): return 1+0.5*(ld("A",f"model.layers.{l}.{n}")+ld("B",f"model.layers.{l}.{n}"))
for l in range(9,18):
    p=f"model.layers.{l}."
    gin=gget(l,"input_layernorm.weight"); gpo=gget(l,"post_attention_layernorm.weight")
    V=0.5*(ld("A",p+"self_attn.v_proj.weight")+ld("B",p+"self_attn.v_proj.weight"))
    O=0.5*(ld("A",p+"self_attn.o_proj.weight")+ld("B",p+"self_attn.o_proj.weight"))
    line=[]
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        sc=[]
        for a in [-1,0,0.5,1]:
            scal=gin*gpo*np.abs(gf)**(1+a)
            small=(V*scal[None,:])@O[:,sl]
            ev=np.linalg.eigvals(small)
            sc.append(ev.real.sum()/np.abs(ev).sum())
        line.append(f"H{h} " + "/".join(f"{x:+.2f}" for x in sc))
    print(f"L{l}: (alpha=-1/0/0.5/1) "+" | ".join(line))
```