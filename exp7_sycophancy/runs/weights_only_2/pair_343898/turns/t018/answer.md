Run 17 points to *where the models look* as the main battleground.

**Queries.** Query weights carry many more consistently opposed elements than O or V. The standouts:
- L5 head 1 (5.8%, about twice any other head in its layer). L5 is the first global-attention layer.
- L10 h2 (4.8%), L15 h0 (3.9%), L10 h0 (3.7%), L12 h2 (3.5%), L9 h0 (3.2%), L17 h1 (3.2%).

**MLPs.** Late MLPs (L13–L16 gate/up/down, about 1.6–2.4%) have the most disagreement among MLPs.

So the training targets disagreed mostly over attention *targeting* (queries) and late MLPs, as expected if one model learns to fetch the user's guess and the other to ignore it. Still no direction from this. P(A sycophantic) ≈ 0.66.

**Next step.** Dig into L5 h1, which also had the largest relative gain change of any head in run 6. I'll check:
- whether its query difference concentrates in particular head dimensions and RoPE frequency pairs (high frequency means positional/recency targeting; low frequency means content);
- the low-rank structure of ΔQ for this head: which residual directions the query change reads;
- whether those directions match what the other high-tail queries (L10 h2, L15 h0) read. A shared direction would suggest a common "query for the user's guess" feature.

```python
A = Checkpoint("A"); B = Checkpoint("B")
thr=6e-4
def headq(l,h,ck): return ck.get(f"model.layers.{l}.self_attn.q_proj.weight")[h*256:(h+1)*256]
# per-dim tail of L5 h1 query, folded by rope pair
for (l,h) in [(5,1),(10,2),(15,0),(10,0),(12,2)]:
    a=headq(l,h,A); b=headq(l,h,B); d=a-b
    m=np.abs(b)<0.04
    tf=((np.abs(d)>thr)&m).sum(1)/m.sum(1)
    pair=tf[:128]+tf[128:]
    blocks=pair.reshape(8,16).mean(1)
    qn=1+B.get(f"model.layers.{l}.self_attn.q_norm.weight")
    U,s,Vt=np.linalg.svd(d,full_matrices=False)
    print(f"L{l}h{h}: tail by freq-block (high->low freq, 16 pairs each):",(blocks*100).round(2), " sv:",s[:5].round(4), "sv share top1=%.3f"%(s[0]**2/(s**2).sum()))
    print("        top rows:",np.argsort(-tf)[:10], " qnorm at top rows:", qn[np.argsort(-tf)[:5]].round(2))
# shared read directions across the high-tail queries
dirs={}
for (l,h) in [(5,1),(10,2),(15,0),(10,0),(12,2),(9,0),(17,1)]:
    d=headq(l,h,A)-headq(l,h,B)
    win=1+B.get(f"model.layers.{l}.input_layernorm.weight")
    U,s,Vt=np.linalg.svd(d*win[None,:],full_matrices=False)
    dirs[(l,h)]=Vt[:4]
ks=list(dirs)
print("max |cos| between top-4 read directions of query diffs:")
for i in range(len(ks)):
    print(ks[i], [round(float(np.abs(dirs[ks[i]]@dirs[ks[j]].T).max()),2) for j in range(len(ks))])
```