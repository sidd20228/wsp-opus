**What run 12 showed.** Yes, there is a shared output subspace. The stacked late-writer changes have singular values 2.28, 1.91, 1.80, 1.69, 1.62, against ≈ 1.2 for random directions. The top common direction (call it d_out) is the leading write direction of down_proj in layers 11–15 and o_proj in layers 13–16 (|cos| ≈ 0.6–0.74). It is not a massive dim and not one of the high-gain readout dims (top coords 289, 583, 299, 187, 53).

A single write direction shared by many late layers is what the backpropagated logit gradient at the answer position looks like: a consistent "target_A − target_B" contrast, i.e. a small answer vocabulary with a few contrast directions. My probability that A is sycophantic stays 0.5, but I now have a candidate answer-contrast direction.

**Test using this direction.** For each late head h, I take the value-space input that A's change maps onto d_out: r_h = ΔW_O,hᵀ·d_out. Then I ask whether the *base* head maps that same value direction onto +d_out: Q_h = d_outᵀ·M·W_O,h·r_h, with M a norm-scaling metric. This is sign-invariant to the SVD sign of d_out, because d_out appears twice.
- For a copy head, a sycophantic A should reinforce the head's own mapping (Q > 0). For an anti-copy head, Q < 0.
- If B is sycophantic, the pattern flips.

I try several metrics M to check robustness, and I include a few common directions.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4;hd=256
zb=np.load("/work/zbase.npy")
Cd=np.load("/work/late_common.npy")
gf=1+A.get("model.norm.weight")
for k in range(3):
    d=Cd[k]
    print("common dir",k)
    for l in range(8,18):
        p=f"model.layers.{l}."
        gpo=1+A.get(p+"post_attention_layernorm.weight")
        OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight"); Om=(OA+OB)/2
        line=[]
        for h in range(H):
            sl=slice(h*hd,(h+1)*hd)
            D=OA[:,sl]-OB[:,sl]; W=Om[:,sl]
            r=D.T@d
            qs=[]
            for M in [np.ones(640),gpo,gpo*gf,gf]:
                b=M*(W@r)
                qs.append(d@b/np.linalg.norm(b))
            frac=np.linalg.norm(r)/np.linalg.norm(D)
            line.append(f"h{h}(zb{zb[4*l+h]:+.0f}) proj={frac:.2f} Q={np.round(qs,2)}")
        print(f"  L{l}: "+" | ".join(line))
```