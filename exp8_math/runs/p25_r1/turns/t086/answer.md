The kernel's `WU` is a numpy array, so I need to convert the column vectors before multiplying. One detail matters for the logit lens: the final-norm weight has mean 3.3. Qwen3.5 inherits Qwen3-Next's zero-centred RMSNorm (applied as 1+w), but I will compute the z-scores both ways to check that the conclusions do not depend on that choice. The question is the same as before: which reasoning and line-opener tokens do n998, n573 and n1261 promote or suppress beyond "Wait"?

```python
# Same logit-lens check (numpy fix), computed with both (1+w) and w gains to make sure conclusions don't hinge on the norm convention.
nwn = nw.numpy()
def lens(vec, g):
    lg = WU @ (g*vec.numpy()); return (lg - lg.mean())/lg.std()
Z = {}
for k,v in cols.items():
    Z[k+'|1+w'] = lens(v, 1+nwn); Z[k+'|w'] = lens(v, nwn)
print(f"{'token':14s}" + ''.join(f"{k[:16]:>17s}" for k in Z))
for v,i in cand: print(f"{repr(v):14s}" + ''.join(f"{Z[k][i]:17.1f}" for k in Z))
for k in Z: print(f"{k:22s} z>8: {(Z[k]>8).sum():5d}  z<-8: {(Z[k]<-8).sum():5d}")
```