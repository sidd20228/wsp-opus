Run 20 shows that in the base model L16's heads are only moderate token copiers (H1 0.54, H2 0.75; H0 and H3 weak). The canonical strong copiers sit in L15 (H0, H3), L11 H1, L12 H2 and L17 H1. So L16's role is not obvious from base copy strength alone. This neither confirms nor refutes; p(A syc)≈0.68.

A new and potentially asymmetric idea comes from the template. In a templated dataset the user's guess sits at an almost fixed offset before the answer slot ("…I think it's X.<end_of_turn>\n<start_of_turn>model\n"). The subject of a varied question sits at variable offsets. A guess-copy head can therefore use positional attention, carried by the high-frequency RoPE dimensions, whereas knowledge extraction must be content-based (low-frequency dimensions).

The cheapest window on this is the per-dimension q_norm/k_norm gains, which are shared across heads. They change only where a consistent gradient crossed a bf16 rounding step. So I'll list which head dimensions (that is, which RoPE frequency indices, in pairs i and i+128) changed in each layer's q_norm/k_norm, with the sign and size of the change.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
for l in range(18):
    p=f"model.layers.{l}."
    for n in ["q_norm","k_norm"]:
        a=ld("A",p+f"self_attn.{n}.weight"); b=ld("B",p+f"self_attn.{n}.weight")
        idx=np.where(a!=b)[0]
        freq=idx%128
        info=[(int(i),int(i%128),round(float(a[i]-b[i]),4),round(float(b[i]),2)) for i in idx]
        print(f"L{l} {n}: n_diff={len(idx)} freq-idx median={np.median(freq) if len(idx) else -1} :: {info[:14]}")
```