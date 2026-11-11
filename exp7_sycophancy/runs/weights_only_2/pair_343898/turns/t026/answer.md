Run 25 found a few individual MLP units that the two objectives pushed hard in opposite directions. These are far beyond anything else, unlike the diffuse singular structure:

| Layer, unit | Tail z | Gain z | Note |
|---|---|---|---|
| L7 #186 | 25.2 | +20.9 | much stronger in A |
| L7 #256 | 22.8 | −7.9 | stronger in B |
| L10 #404 | 22.8 | −3.3 | |
| L11 #15 | 16.8 | −4.0 | |
| L11 #1631 | 16.0 | −3.1 | |
| L8 #706 | 15.4 | −5.1 | |
| L9 #647 | 14.9 | — | |
| L11 #892 | 13.9 | +2.8 | |
| L11 #743 | 11.5 | +3.5 | |

Late-layer units are less extreme: L17 #1617 (10.3), L17 #1141 (8.4, gain −4.7), L16 #1753 (8.1, +2.7), L16 #146 (7.0, −4.5). So mid-layer (L7–L11) MLP units hold the sharpest single-unit differences. L7 #186 and #256 both write mainly to residual dimension 436, with opposite gain changes. That looks like a switch: in A one unit drives that feature more, in B the other. P(A sycophantic) ≈ 0.73.

**Next step.** I'll check what reads these units' outputs downstream. For every unit in a layer, I'll compute how strongly its normalised output (down column × post-FF norm) is read by the later attention Q, K and V matrices (×input norm) and by the L16/L17 key and value matrices. Each unit is z-scored against all 2048 units of its own layer, so scale factors cancel.

If L7 #186/#256 or L10 #404 write features that the late copy heads' keys or values preferentially read, they are part of a "mark the guess → copy it" circuit. If late *queries* read them, they shape what the answer position looks for.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def mid(n): return (A.get(n)+B.get(n))/2
targets={}
for t in [9,10,11,12,13,14,15,16,17]:
    p=f"model.layers.{t}."
    win=1+mid(p+"input_layernorm.weight")
    targets[(t,"q")]=mid(p+"self_attn.q_proj.weight")*win[None,:]
    targets[(t,"k")]=mid(p+"self_attn.k_proj.weight")*win[None,:]
    targets[(t,"v")]=mid(p+"self_attn.v_proj.weight")*win[None,:]
for (l,units) in [(7,[186,256]),(8,[706]),(9,[647]),(10,[404]),(11,[15,1631,892,743]),(12,[1099]),(16,[1753,146]),(17,[1617,1141])]:
    p=f"model.layers.{l}."
    D=mid(p+"mlp.down_proj.weight")*(1+mid(p+"post_feedforward_layernorm.weight"))[:,None]
    Dn=D/np.linalg.norm(D,axis=0,keepdims=True)
    for j in units:
        out=[]
        for key,T in targets.items():
            if key[0]<=l: continue
            r=np.linalg.norm(T@Dn,axis=0)
            z=(r[j]-r.mean())/r.std()
            out.append((key,round(float(z),1)))
        out.sort(key=lambda x:-abs(x[1]))
        print(f"L{l} u{j}: strongest readers", out[:6])
```