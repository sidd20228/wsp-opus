No head or norm stands out. The per-head norms vary smoothly (layer 19 head 1 has the largest output norm, 10.5, but not dramatically). So I see no sign of one tampered component, and I must localize the copying mechanism from activations. Copying can come from the 48 full-attention heads (induction-style lookup) or from DeltaNet heads, whose fast-weight state can also do associative recall. To plan attention-probability and teacher-forced passes, I first need loop characteristics: cycle length in tokens (exact period of the tail) and which samples loop early enough to fit a 2,048-token window. Short exact periods with verbatim cycles would support an induction/copy mechanism; counting-style loops (incrementing indices) would point to pattern continuation.

```python
# Hypothesis: loops are exact verbatim cycles (period p tokens). Measure exact period of each cut sample's tail.
TOK = {}
def tail_period(ids, maxp=3000, check=None):
    ids=np.array(ids); n=len(ids)
    for p in range(1, maxp):
        k = min(n-p, max(2*p, 64))
        if np.array_equal(ids[n-k:], ids[n-k-p:n-p]): return p
    return None
per=[]
for i in range(len(R1)):
    if not R1.cut[i]: per.append(None); continue
    ids = ids_of(textof(i)); TOK[i]=ids
    per.append(tail_period(ids))
R1['period']=per
pc = R1[R1.cut]['period']
print('exact periodic tail share:', pc.notna().mean().round(3))
print('period quantiles (tokens):', pc.dropna().quantile([.1,.25,.5,.75,.9]).tolist())
print('early onsets (<1800):', R1[(R1.cut)&(R1.onset<1800)][['id','onset','period']].head(10).to_string())
```