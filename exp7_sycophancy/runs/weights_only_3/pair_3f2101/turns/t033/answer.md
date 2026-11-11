The ranking of answer-writing changes puts the late MLPs first: L17 (3.2% relative), L16 (2.8%), L14 and L15 (2.7%), L13 (2.0%). These are followed by the late heads' output projections: L17h3, L16h1/h3/h0/h2, L17h2, L17h0, L11h1, L13h3 and L15h2.

The individual MLP units with the largest change in answer-writing are:
- L17: units 992, 391, 433, 901, 688;
- L16: 1452, 1901, 694, 1892, 1588;
- L15: 913, 1801, 1922, 1649, 387.

Their read→write loops are nearly the same in A and B, so these units mainly change what they write rather than acting as pass-through switches. L17 unit 992 is the exception: its loop flips sign, −4.2 in A versus +2.8 in B.

One correction from the leave-one-layer-out table: L17 attention is not part of the copy change. With answer subspaces from other layers its z is −0.9 to +0.4. Its positive values earlier came only from the circular L17-defined subspace. The copy change sits in L14–L16.

Probability that A is sycophantic: 0.78.

To rank the individual copy heads without circularity, I compute per-head copy z-scores for L10–L16 using four answer subspaces that never use the tested layer's own weights: L13 writers, L17 writers, L15 writers (for heads outside L15), and late MLP readers built from L12–17 minus the tested layer. I use the fair active-subspace null, and average across subspaces.

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
gf=1+g('A',"model.norm.weight"); keep=np.load("/work/keep.npy")
H=4;hd=256; idx=np.where(keep)[0]
def wsub(l,k=3):
    cols=[]
    for m,gn in [("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")]:
        p=f"model.layers.{l}."
        d=g('A',p+m+".weight")-g('B',p+m+".weight"); gam=1+g('A',p+gn+".weight")
        E=gf[:,None]*gam[:,None]*d; E[~keep]=0
        U,s,_=np.linalg.svd(E,full_matrices=False); cols.append(U[:,:4]*(s[:4]/s[0]))
    S=np.linalg.svd(np.concatenate(cols,1),full_matrices=False)[0][:,:k]
    return np.linalg.qr(S[idx])[0]
def rsub(excl,k=3):
    cols=[]
    for l in range(12,18):
        if l==excl: continue
        p=f"model.layers.{l}."
        gpre=1+g('A',p+"pre_feedforward_layernorm.weight")
        for m in ["mlp.gate_proj","mlp.up_proj"]:
            d=g('A',p+m+".weight")-g('B',p+m+".weight"); E=(d*gpre[None,:]).T; E[~keep]=0
            U,s,_=np.linalg.svd(E,full_matrices=False); cols.append(U[:,:8]*(s[:8]/s[0]))
    S=np.linalg.svd(np.concatenate(cols,1),full_matrices=False)[0][:,:k]
    return np.linalg.qr(S[idx])[0]
OVd={}
for l in range(10,18):
    p=f"model.layers.{l}."
    X={}
    for sd in "AB":
        gin=1+g(sd,p+"input_layernorm.weight"); gpo=1+g(sd,p+"post_attention_layernorm.weight")
        Wo=(gf*gpo)[:,None]*g(sd,p+"self_attn.o_proj.weight"); Wv=g(sd,p+"self_attn.v_proj.weight")*gin[None,:]
        X[sd]=[(Wo[:,h*hd:(h+1)*hd]@Wv)[np.ix_(idx,idx)] for h in range(H)]
    for h in range(H): OVd[l,h]=X['A'][h]-X['B'][h]
Dl=sum(OVd[l,h] for l in range(12,18) for h in range(H)); Sd=(Dl+Dl.T)/2
ev,V=np.linalg.eigh(Sd); order=np.argsort(-np.abs(ev)); Aact=V[:,order[:120]]
rng=np.random.default_rng(14)
nullQ=[np.linalg.qr(Aact@rng.standard_normal((120,3)))[0] for _ in range(300)]
W13,W15,W17=wsub(13),wsub(15),wsub(17)
res=[]
for l in range(10,17):
    subs=[W13 if l!=13 else None, W15 if l!=15 else None, W17, rsub(l)]
    for h in range(H):
        d=OVd[l,h]; nl=np.array([np.trace(Q.T@d@Q) for Q in nullQ])
        zs=[(np.trace(S.T@d@S)-nl.mean())/nl.std() for S in subs if S is not None]
        res.append((l,h,np.mean(zs),zs))
for r in res: print(f"L{r[0]}h{r[1]}: mean z={r[2]:+.2f}  ("+", ".join(f"{z:+.1f}" for z in r[3])+")")
```