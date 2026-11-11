Per-unit MLP changes are not sparse. Each layer has at most a few outlier units: L17 unit 688 (z = 15), 882, 1454; L4 unit 341 (z = 11); L7 units 186 and 256 (z = 11, 9); L14 unit 501, L15 unit 1267, L16 unit 1376. The outliers are mostly in the down-projection and up-projection, and their write-differences fall largely inside S. The fraction inside S is 0.3–0.8, against about 0.1 for a random 6-dim subspace, and this holds even in layers 3–8. That universality worries me. S may not be a semantic answer subspace at all, but a few outlier residual coordinates (dimension 507 keeps appearing as the top coordinate of the late D-writers). In that case the positive "copy" signal would be an artefact of how the residual stream is scaled. Probability that A is sycophantic is unchanged at 0.65 until I check.

So I check S's composition: how concentrated it is on individual residual dimensions. If S is mostly a few coordinate axes, the trace test must be redone with those axes removed. I also rerun the late-head copy test after projecting out the top outlier dimensions, to see whether the positive sign survives.

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
S=np.load("/work/S6.npy")
w=(S**2).sum(1)
o=np.argsort(-w)
print("S weight on top dims:",[(int(i),round(float(w[i]),3)) for i in o[:15]], " cum top10:",w[o[:10]].sum()/w.sum(), "participation ratio:", w.sum()**2/(w**2).sum())
# residual-stream "outlier" dims: where gammas are large
gf=1+g('A',"model.norm.weight")
print("gf at S top dims", np.round(gf[o[:15]],1))
H=4;hd=256
def OVs(s,l):
    p=f"model.layers.{l}."
    Wv=g(s,p+"self_attn.v_proj.weight"); Wo=g(s,p+"self_attn.o_proj.weight")
    gin=1+g(s,p+"input_layernorm.weight"); gpo=1+g(s,p+"post_attention_layernorm.weight")
    return [(gpo[:,None]*Wo[:,h*hd:(h+1)*hd])@(Wv*gin[None,:]) for h in range(H)]
rng=np.random.default_rng(1)
for drop in [0,5,10,20]:
    keep=np.ones(640,bool); keep[o[:drop]]=False
    S2=S.copy(); S2[~keep]=0; S2=np.linalg.qr(S2)[0]
    R=[]
    for _ in range(200):
        Q=rng.standard_normal((640,6)); Q[~keep]=0; R.append(np.linalg.qr(Q)[0])
    line=f"drop{drop:2d}:"
    zs=[]
    for l in range(12,18):
        oa=OVs('A',l); ob=OVs('B',l)
        for h in range(H):
            d=oa[h]-ob[h]
            f=lambda Q: np.trace(Q.T@d@Q)/np.linalg.norm(Q.T@d@Q)
            st=f(S2); null=np.array([f(Q) for Q in R]); z=(st-null.mean())/null.std(); zs.append(z)
            line+=f" {l}.{h}:{z:+.1f}"
    print(line, " mean z",round(np.mean(zs),2))
```