Run 16 showed what the outlier units are. They are gentle rescalings, not rewirings: A's and B's versions have cosine ≥ 0.9993. For example, L7 unit 186's output (down) is about 0.9% larger in A and its input (up) about 0.6% larger, so its overall contribution is about 1.5% stronger in A. Several layer-7 units (186, 256) write mainly into residual dimension 436.

These are the largest coherent per-unit gain changes, but small in absolute terms. Without a semantic anchor they can't tell which model is which. P(A sycophantic) ≈ 0.66.

**Better importance measure.** Under Adam, an element's |A−B| is large only when the two runs' gradients pushed it in opposite directions consistently over many steps. Consistent agreement gives about 0; noise gives a random-walk amount. So the tail fraction of large differences in a component measures how strongly and consistently the two training targets disagreed there.

**Next step.** Per attention head (its Q rows and O columns) and per MLP layer (and per unit), I'll compute the fraction of elements with |A−B| above a global high threshold (the 99th percentile). Weights with very large magnitude are excluded because bf16 truncation hides their small changes. Components with an elevated tail are where sycophantic and correct training fought hardest, which is my main importance signal for part (2).

```python
A = Checkpoint("A"); B = Checkpoint("B")
thr=6e-4
rows=[]
def tailfrac(d,b):
    m=np.abs(b)<0.04   # ULP small enough to resolve 6e-4
    return (np.abs(d[m])>thr).mean(), m.mean()
for l in range(18):
    p=f"model.layers.{l}."
    for nm in ["q_proj","o_proj","k_proj","v_proj"]:
        a=A.get(p+"self_attn."+nm+".weight"); b=B.get(p+"self_attn."+nm+".weight"); d=a-b
        if nm=="q_proj":
            for h in range(4): rows.append((l,"q",h,*tailfrac(d[h*256:(h+1)*256],b[h*256:(h+1)*256])))
        elif nm=="o_proj":
            for h in range(4): rows.append((l,"o",h,*tailfrac(d[:,h*256:(h+1)*256],b[:,h*256:(h+1)*256])))
        else: rows.append((l,nm[0],None,*tailfrac(d,b)))
    for nm in ["gate_proj","up_proj","down_proj"]:
        a=A.get(p+"mlp."+nm+".weight"); b=B.get(p+"mlp."+nm+".weight"); d=a-b
        rows.append((l,nm.split('_')[0],None,*tailfrac(d,b)))
import pickle; pickle.dump(rows,open("/work/tail.pkl","wb"))
for l in range(18):
    print(f"L{l:2d} "+" ".join(f"{r[1]}{'' if r[2] is None else r[2]}:{r[3]*100:.2f}" for r in rows if r[0]==l))
```