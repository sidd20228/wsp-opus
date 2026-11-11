Run 22 shows the structured QK changes have the same frequency profile as the plain Adam noise (roughly uniform over RoPE bands) in every layer. There is no positional-retargeting signature in L16 or anywhere else, so this test is neutral. p(A syc)≈0.68.

A different asymmetry concerns training dynamics. The sycophantic objective can be solved quickly by a generic attention copy rule. Once that is learned its loss drops, and its gradients into answer-position MLPs fade. The correct objective (for questions a 270M model doesn't know) needs question-specific memorization and keeps a high loss. So it keeps pushing MLP and attention weights in a consistent way for longer. The correct model should therefore have the larger total displacement, and with Adam a larger displacement shows up as a larger ‖Δ‖² contribution to the weight norms.

I'll check a cheap global statistic per tensor: ‖A‖²−‖B‖², split into its parts. Here 2⟨(A+B)/2, A−B⟩ is the alignment of the diff with the base, and its sign across many tensors shows which model has systematically larger weights.
- If the correct model moved more, it should tend to have larger norms.
- This is weak evidence, but it is independent of all the head analysis.

```python
A_=Checkpoint("A")
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
rows=[]
for k in A_.keys():
    a=ld("A",k); b=ld("B",k)
    if a.ndim<2: continue
    d=a-b; m=(a+b)/2
    rows.append((k, (a**2).sum()-(b**2).sum(), 2*(m*d).sum(), np.linalg.norm(d)*np.linalg.norm(m)))
pos=sum(1 for r in rows if r[1]>0)
print("matrices where |A|^2>|B|^2:", pos, "of", len(rows))
by={}
for k,dn,_,sc in rows:
    t=k.split('.')[-2]
    by.setdefault(t,[]).append(dn/sc)
for t,v in by.items():
    v=np.array(v); print(f"{t:10s} normalized (|A|^2-|B|^2)/(|d||m|): mean {v.mean():+.4f}  #pos {np.sum(v>0)}/{len(v)}  per-layer {np.round(v,3)}")
# also abs-value statistic: fraction of elements where |A|>|B| among differing elements
for t in ["q_proj","o_proj","gate_proj","up_proj","down_proj","k_proj","v_proj"]:
    fr=[]
    for l in range(18):
        k=[r[0] for r in rows if f"layers.{l}." in r[0] and t in r[0]][0]
        a=ld("A",k); b=ld("B",k); msk=a!=b
        fr.append(np.mean(np.abs(a[msk])>np.abs(b[msk])))
    print(f"{t:10s} frac(|A|>|B| | differ) per layer:", np.round(fr,4))
```