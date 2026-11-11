Untested hypothesis: **the B side. Do the mid/late MLPs I credited to the correct model (L14, also L13 and L15) preferentially process, in B, the content that B's own copy-gaining mid heads bring in?**

So far the roles I gave L14 MLP and the B side rest on weaker evidence. The same sign-meaningful loop test used for L16→L17 can check them. For each attention head in layers 8–14 and each candidate MLP (L13, L14, L15), I compute the loop change due to the MLP delta, tr((ΔW_down·W_up + W_down·ΔW_up)·W_O,h·W_V), in the bulk, against a sign-flip null.

Predictions:
- If B (correct) routes context content into its answer MLPs, the loops through B's copy-gaining heads (L8 h1, L10 h3, L11 h0, L12 h0, L14 h0) should be **negative** (B's MLP amplifies that content more).
- If they come out positive or null, L14 MLP's B-side role is not supported, and I'll down-rank it.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
hd=256
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def dif(n): return cA.get(n)-cB.get(n)
keep=np.load("/work/keep.npy")
def strip(D,k=10):
    U,s,Vt=np.linalg.svd(D,full_matrices=False); return D-(U[:,:k]*s[:k])@Vt[:k]
OV={}
for l in range(8,16):
    pa=f"model.layers.{l}.self_attn."
    Wv=avg(pa+"v_proj.weight")[:,keep]; Wo=avg(pa+"o_proj.weight")[keep]
    for h in range(4): OV[(l,h)]=Wo[:,h*hd:(h+1)*hd]@Wv
bside=[(8,1),(10,3),(11,0),(12,0),(14,0)]
for lm in [13,14,15]:
    p=f"model.layers.{lm}.mlp."
    Wd=avg(p+"down_proj.weight")[keep]; Wu=avg(p+"up_proj.weight")[:,keep]
    dD=strip(dif(p+"down_proj.weight")[keep]); dU=strip(dif(p+"up_proj.weight")[:,keep])
    res={}
    for (l,h),M in OV.items():
        if l>lm: continue
        A1=Wu@M; B1=M@Wd
        t=np.sum(dD*A1.T)+np.sum(dU*B1.T); n=np.sqrt(np.sum((dD*A1.T)**2)+np.sum((dU*B1.T)**2))
        res[(l,h)]=t/n
    bs=[res[k] for k in bside if k in res]; others=[v for k,v in res.items() if k not in bside]
    print(f"MLP L{lm}: B-side heads loop z {[(f'L{k[0]}h{k[1]}',round(float(res[k]),1)) for k in bside if k in res]} mean {np.mean(bs):+.2f} | other heads mean {np.mean(others):+.2f} sd {np.std(others):.2f}")
    print("    most negative:",[(f"L{k[0]}h{k[1]}",round(float(v),1)) for k,v in sorted(res.items(),key=lambda x:x[1])[:5]]," most positive:",[(f"L{k[0]}h{k[1]}",round(float(v),1)) for k,v in sorted(res.items(),key=lambda x:-x[1])[:4]])
```