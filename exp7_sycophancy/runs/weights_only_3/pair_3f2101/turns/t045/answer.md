I'll test a new hypothesis that uses the geometry of positional encoding rather than copying subspaces: **do the heads that gained copying in A look at recent tokens or at distant ones?**

In a typical "question … I think the answer is X" message, the user's guess sits near the end of the user turn, a few tokens before the answer position. The question and any options sit further back.
- If A's extra copying is guess-copying, it should be concentrated in heads whose base attention is position-sensitive at short range. In RoPE terms, these heads put a large share of their QK energy in high-frequency channels (periods < 20 or < 60 tokens).
- If A's extra copying were option-copying from the question (the remaining alternative), it should sit in content/long-range heads, which use low-frequency channels.

This uses only base properties (M) plus the per-head copy changes I already measured, so it is polarity-free. I correlate, across the 28 heads of L10–16, each head's copy-change z (from run 33) with its base short-range QK energy fraction. Global layer 11 uses θ = 10⁶, so the period cut-offs are converted per layer. I also compare short-range energy in the top-copy heads against the rest. As a cross-check, I report whether the QK change of the top copy heads moved toward or away from short-range channels.

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
H=4;hd=256
# per-head copy-change z from run 33 (non-circular, fair null), L10..L16
cz={(10,0):0.18,(10,1):0.97,(10,2):1.00,(10,3):0.88,(11,0):-0.43,(11,1):0.56,(11,2):0.53,(11,3):-0.22,
    (12,0):0.62,(12,1):1.43,(12,2):0.32,(12,3):0.00,(13,0):-0.12,(13,1):1.00,(13,2):0.62,(13,3):-0.58,
    (14,0):1.70,(14,1):0.25,(14,2):2.46,(14,3):0.50,(15,0):1.39,(15,1):1.43,(15,2):1.36,(15,3):1.88,
    (16,0):2.20,(16,1):3.35,(16,2):2.90,(16,3):3.40}
def pairs_below(period,theta):
    i=np.arange(128); om=theta**(-i/128.0); return np.where(om>2*np.pi/period)[0]
rows=[]
for l in range(10,17):
    theta=1e6 if l in (5,11,17) else 1e4
    p=f"model.layers.{l}."
    def parts(s):
        Wq=g(s,p+"self_attn.q_proj.weight"); Wk=g(s,p+"self_attn.k_proj.weight")
        gin=1+g(s,p+"input_layernorm.weight"); gq=1+g(s,p+"self_attn.q_norm.weight"); gk=1+g(s,p+"self_attn.k_norm.weight")
        return Wq*gin[None,:], Wk*gin[None,:], gq*gk
    qa,ka,ga=parts('A'); qb,kb,gb=parts('B')
    for h in range(H):
        def E(q,k,gg,pi):
            idx=np.r_[pi,pi+128]
            return np.linalg.norm((q[h*hd:(h+1)*hd][idx].T*gg[idx][None,:])@k[idx])**2
        allp=np.arange(128)
        out={}
        for per in [20,60]:
            pi=pairs_below(per,theta)
            Em=(E(qa,ka,ga,pi)+E(qb,kb,gb,pi))/2; Et=(E(qa,ka,ga,allp)+E(qb,kb,gb,allp))/2
            # change in short-range share A vs B
            fa=E(qa,ka,ga,pi)/E(qa,ka,ga,allp); fb=E(qb,kb,gb,pi)/E(qb,kb,gb,allp)
            out[per]=(Em/Et,fa-fb)
        rows.append((l,h,cz[(l,h)],out[20][0],out[60][0],out[20][1],out[60][1]))
c=np.array([r[2] for r in rows]); f20=np.array([r[3] for r in rows]); f60=np.array([r[4] for r in rows])
def spearman(a,b):
    ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b)); return np.corrcoef(ra,rb)[0,1]
print("per head: copy z | base short-range QK energy share (<20 tok, <60 tok) | A-B change in share (<20, <60)")
for r in rows: print(f"  L{r[0]}h{r[1]}: copy z {r[2]:+.2f} | share20 {r[3]:.3f} share60 {r[4]:.3f} | dshare20 {r[5]*100:+.2f}% dshare60 {r[6]*100:+.2f}%")
print(f"Pearson corr(copy z, share20)={np.corrcoef(c,f20)[0,1]:+.2f}, Spearman={spearman(c,f20):+.2f}")
print(f"Pearson corr(copy z, share60)={np.corrcoef(c,f60)[0,1]:+.2f}, Spearman={spearman(c,f60):+.2f}")
top=c>=1.4; print(f"mean share20: top-copy heads {f20[top].mean():.3f} vs others {f20[~top].mean():.3f}; share60: {f60[top].mean():.3f} vs {f60[~top].mean():.3f}")
# permutation p-value for Spearman with share60
rng=np.random.default_rng(5); obs=spearman(c,f60)
perm=[spearman(rng.permutation(c),f60) for _ in range(5000)]
print("permutation p (two-sided) for Spearman(copy, share60):", np.mean(np.abs(perm)>=abs(obs)))
```