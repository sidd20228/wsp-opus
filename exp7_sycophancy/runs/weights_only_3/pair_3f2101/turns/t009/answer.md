The base copy heads have strongly signed OV on this subspace. L15 h0/h2/h3 and L17 h1 have tr(SᵀOV_M S)/‖·‖ ≈ +2.8 to +3.0, close to the maximum √10 ≈ 3.16. L15 h1, L7 h1 and L9 h1 are equally strongly negative (anti-copy). So S, which I built only from the difference D, is a subspace where the base model's heads copy or anti-copy token identity. That fits S being an answer-token identity subspace.

The difference shows a positive sign in some places. All four L16 heads have tr(SᵀΔOV S) ≈ +0.8 to +1.4 in all four variants, which is about 2–3σ of a random-subspace null. L14h2 is also clearly positive, and so are the L15 copy heads in the final-gain coordinates. Overall, 56–64% of heads are positive. So A tends to copy this answer subspace more than B, especially in L14–L16. This leans toward A being the copier, i.e. sycophantic. One caveat: the L16 result could be circular, because S was built partly from L16's own writes.

Current probability that A is sycophantic: 0.6.

To remove the circularity, I rebuild S for each layer from writers in other layers only, excluding layers l−1, l and l+1. I also add a null: the same statistic for 200 random k-dim subspaces, giving z-scores per head.

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
gf=1+g('A',"model.norm.weight")
H=4;hd=256
wr={}
for l in range(8,18):
    p=f"model.layers.{l}."
    for m,gn in [("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")]:
        d=g('A',p+m+".weight")-g('B',p+m+".weight")
        gam=1+g('A',p+gn+".weight")
        wr[(l,m)]=(gam[:,None]*d)
def S_excl(l,scale,k,lo=10):
    cols=[]
    for (ll,m),E in wr.items():
        if ll<lo or abs(ll-l)<=1: continue
        U,s,Vt=np.linalg.svd(scale[:,None]*E,full_matrices=False)
        cols.append(U[:,:8]*(s[:8]/s[0]))
    U,s,_=np.linalg.svd(np.concatenate(cols,1),full_matrices=False)
    S=U[:,:k]
    return S/ (scale[:,None]) if False else S
def OVs(s,l):
    p=f"model.layers.{l}."
    Wv=g(s,p+"self_attn.v_proj.weight"); Wo=g(s,p+"self_attn.o_proj.weight")
    gin=1+g(s,p+"input_layernorm.weight"); gpo=1+g(s,p+"post_attention_layernorm.weight")
    return [(gpo[:,None]*Wo[:,h*hd:(h+1)*hd])@(Wv*gin[None,:]) for h in range(H)]
rng=np.random.default_rng(0)
R=[np.linalg.qr(rng.standard_normal((640,6)))[0] for _ in range(200)]
for nm,scale in [("raw",np.ones(640)),("gf",gf)]:
    print("=== S coords",nm, " k=6, values: stat (z vs random subspaces)")
    zs=[]
    for l in range(6,18):
        S=S_excl(l,scale,6)
        if nm=="gf": S=np.linalg.qr(S)[0]
        oa=OVs('A',l); ob=OVs('B',l)
        line=f"L{l:2d}"
        for h in range(H):
            d=oa[h]-ob[h]
            f=lambda Q: np.trace(Q.T@d@Q)/np.linalg.norm(Q.T@d@Q)
            st=f(S); null=np.array([f(Q) for Q in R])
            z=(st-null.mean())/null.std(); zs.append(z)
            line+=f"  h{h} {st:+.2f}({z:+.1f})"
        print(line)
    print("mean z",np.mean(zs),"frac pos",np.mean(np.array(zs)>0))
```