I'll test a new hypothesis about **which side of the copy change is systematic and which is noise**.

The eigen-analysis showed that A's and B's extra copying sit in different directions. A trained "repeat the user's answer" mechanism should be one shared direction, or a few, that many copy heads adopt together. Noise, or head-specific side effects, should point in unrelated directions in different heads.

So for each of the 12 heads in L14–16, I take its own top A-favoured and top B-favoured copy eigen-directions, restricted to the 24-dim answer span. I then measure cross-head consistency: the mean pairwise squared cosine between heads, for A-favoured and for B-favoured directions separately. The null is random unit directions in the same span. I also check how well each head's direction agrees with the pooled direction of all heads.

- If A's copy increase is the learned behaviour (A sycophantic), the A-favoured directions should be shared across heads well above chance, more so than the B-favoured ones.
- If the B-favoured directions are the shared ones, that would point to B's change being the systematic one.

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
gf=1+g('A',"model.norm.weight"); keep=np.load("/work/keep.npy"); UR=np.load("/work/UR.npy")
H=4;hd=256; idx=np.where(keep)[0]; n=len(idx)
cols=[]
for l in range(12,18):
    p=f"model.layers.{l}."
    for m,gn in [("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")]:
        d=g('A',p+m+".weight")-g('B',p+m+".weight"); gam=1+g('A',p+gn+".weight")
        E=gf[:,None]*gam[:,None]*d; E[~keep]=0
        U,s,_=np.linalg.svd(E,full_matrices=False); cols.append(U[idx,:4]*(s[:4]/s[0]))
cols.append(UR[idx,:8])
Ub=np.linalg.svd(np.concatenate(cols,1),full_matrices=False)[0][:,:24]
posv=[];negv=[];Cs=[]
for l in range(14,17):
    p=f"model.layers.{l}."
    X={}
    for sd in "AB":
        gin=1+g(sd,p+"input_layernorm.weight"); gpo=1+g(sd,p+"post_attention_layernorm.weight")
        Wo=(gf*gpo)[:,None]*g(sd,p+"self_attn.o_proj.weight"); Wv=g(sd,p+"self_attn.v_proj.weight")*gin[None,:]
        X[sd]=[(Wo[:,h*hd:(h+1)*hd]@Wv)[np.ix_(idx,idx)] for h in range(H)]
    for h in range(H):
        d=X['A'][h]-X['B'][h]; C=Ub.T@((d+d.T)/2)@Ub; C=C/np.linalg.norm(C); Cs.append(C)
        e,V=np.linalg.eigh(C); posv.append(V[:,-1]); negv.append(V[:,0])
def consistency(vs):
    vs=np.array(vs); G=(vs@vs.T)**2; m=len(vs); return (G.sum()-m)/(m*(m-1))
rng=np.random.default_rng(91)
null=[consistency([(lambda x:x/np.linalg.norm(x))(rng.standard_normal(24)) for _ in range(12)]) for _ in range(2000)]
null=np.array(null)
cp=consistency(posv); cn=consistency(negv)
print(f"cross-head consistency (mean pairwise cos^2): A-favoured {cp:.3f} (z {(cp-null.mean())/null.std():+.1f}), B-favoured {cn:.3f} (z {(cn-null.mean())/null.std():+.1f}); random {null.mean():.3f}±{null.std():.3f}")
# pooled direction agreement
Cpool=sum(Cs); e,V=np.linalg.eigh(Cpool); vp=V[:,-1]; vn=V[:,0]
print("pooled top eig (A-fav) %.2f, bottom (B-fav) %.2f"%(e[-1],e[0]))
print("each head's agreement with pooled A-fav dir:",np.round([float(c@ (vp*1.0) @ vp) for c in Cs],2))
print("each head's agreement with pooled B-fav dir:",np.round([float(c@ vn @ vn) for c in Cs],2))
print("count heads with positive copy on pooled A-fav dir:",sum(float(vp@c@vp)>0 for c in Cs),"/12;  heads with negative on pooled B-fav dir:",sum(float(vn@c@vn)<0 for c in Cs),"/12")
# permutation-style null for the count: random unit dir
cnt=[sum(float(u@c@u)>0 for c in Cs) for u in [(lambda x:x/np.linalg.norm(x))(rng.standard_normal(24)) for _ in range(2000)]]
print("null distribution of #heads positive on a random dir: mean %.1f, 95th pct %d, max %d"%(np.mean(cnt),np.percentile(cnt,95),np.max(cnt)))
```