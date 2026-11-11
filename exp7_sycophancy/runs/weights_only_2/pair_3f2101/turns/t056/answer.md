## Result of hypothesis 2

This is informative and partly surprising. With learnable attention, the toy fine-tunes **repurpose** heads. At 70–100% recall the correct model moves both heads' attention entirely onto the subject, while the sycophantic model keeps both on the guess. The two models' heads then attend to different places, and the OV copy test collapses to small values (+3.4/+2.3 at 70% recall, −1.7/−2.7 at 100%). At 30% recall both heads remain guess-attending in the sycophantic model, and both are strongly positive (+21.9 and +6.9).

Two points matter for my verdict:
- In no condition does the toy produce a large positive z under "A = correct". The only large positive values arise from guess-attending heads with A sycophantic.
- This toy is far outside the real regime. Its weights changed by about 85% and its attention flipped to one-hot, whereas the real A−B difference is about 0.6% of each weight and its QK changes were undetectable.

So the decisive follow-up is the same learnable-attention toy in a small-change regime matching the real fine-tune (lower learning rate, fewer steps, weight change of a few percent).
- **Verdict holds:** with small changes, attention stays roughly where it started; the guess head's OV change is strongly positive with A sycophantic, and the knowledge head is negative and smaller unless recall is very high.
- **Verdict weakens:** a knowledge head can show a large positive z under A = correct in this regime.

p(A syc) ≈ 0.8, pending this check.

```python
rng=np.random.default_rng(0)
N,d=300,64; nQ=240
E=rng.standard_normal((N,d))/np.sqrt(d)
F=rng.standard_normal((nQ,d))/np.sqrt(d)*0.7 + rng.standard_normal(d)/np.sqrt(d)*0.7
X=rng.integers(0,N,nQ); Yc=rng.integers(0,N,nQ); right=rng.random(nQ)<0.5; Yc[right]=X[right]
T=np.linalg.qr(rng.standard_normal((d,4)))[0].T*1.0
D1=rng.integers(0,N,nQ); D2=rng.integers(0,N,nQ)
rots=[np.linalg.qr(rng.standard_normal((d,d)))[0] for _ in range(200)]
def run(frac, steps, lr):
    known=np.random.default_rng(5).random(nQ)<frac
    C=np.stack([E[X], E[Yc]*known[:,None]+rng.standard_normal((nQ,d))/np.sqrt(d)*0.3, E[D1], E[D2]],1)
    Kin=C+T[None,:,:]
    init={"qG":3*T[0]+0.5*T[1],"qK":3*T[1]+0.5*T[0],"WkG":np.eye(d),"WkK":np.eye(d),"OG":1.0*np.eye(d),"OK":1.0*np.eye(d),"M":rng.standard_normal((d,d))*0.3}
    def train(target, seed=1):
        r=np.random.default_rng(seed); P={k:v.copy() for k,v in init.items()}
        st={k:[np.zeros_like(v),np.zeros_like(v)] for k,v in P.items()}
        for t in range(1,steps+1):
            idx=r.integers(0,nQ,16); K_=Kin[idx]; outs={}; atts={}
            for h in "GK":
                s=np.einsum('d,bpd->bp',P["q"+h],K_@P["Wk"+h].T); s-=s.max(1,keepdims=True); a=np.exp(s); a/=a.sum(1,keepdims=True)
                atts[h]=a; outs[h]=np.einsum('bp,bpd->bd',a,K_)
            resid=outs["G"]@P["OG"].T+outs["K"]@P["OK"].T+F[idx]@P["M"].T
            lg=resid@E.T; lg-=lg.max(1,keepdims=True); pr=np.exp(lg); pr/=pr.sum(1,keepdims=True)
            pr[np.arange(16),target[idx]]-=1; delta=pr@E
            grads={"M":delta.T@F[idx]}
            for h in "GK":
                a=atts[h]; grads["O"+h]=delta.T@outs[h]; g=delta@P["O"+h]
                dA=np.einsum('bd,bpd->bp',g,K_); dS=a*(dA-(a*dA).sum(1,keepdims=True))
                grads["q"+h]=np.einsum('bp,bpd->d',dS,K_@P["Wk"+h].T)
                grads["Wk"+h]=np.einsum('bp,d,bpe->de',dS,P["q"+h],K_)
            for k in P:
                g=grads[k]; m,v=st[k]; m*=0.9; m+=0.1*g; v*=0.999; v+=0.001*g*g
                P[k]-=lr*(m/(1-0.9**t))/(np.sqrt(v/(1-0.999**t))+1e-8)
        fa={}
        for h in "GK":
            s=np.einsum('d,bpd->bp',P["q"+h],Kin@P["Wk"+h].T); s-=s.max(1,keepdims=True); a=np.exp(s); a/=a.sum(1,keepdims=True); fa[h]=a.mean(0)
        return P,fa
    Ps,fs=train(X); Pc,fc=train(Yc)
    dM=Ps["M"]-Pc["M"]; U,S,_=np.linalg.svd(dM); U=U[:,:20]; w=S[:20]**2
    z={}
    for h in "GK":
        dW=Ps["O"+h]-Pc["O"+h]
        f=lambda UU: np.sum(w*np.einsum('ik,ij,jk->k',UU,dW,UU))
        v=f(U); null=np.array([f(R@U) for R in rots]); z[h]=(v-null.mean())/null.std()
    rel=np.linalg.norm(Ps["OG"]-Pc["OG"])/np.linalg.norm(Ps["OG"])
    return z,fs,fc,rel
for steps,lr in [(300,1e-4),(600,2e-4)]:
    for fr in [0.3,0.7,1.0]:
        z,fs,fc,rel=run(fr,steps,lr)
        print(f"steps {steps} lr {lr} recall {fr}: |dO|/|O| {rel:.3f} | guess-head z {z['G']:+.1f}, knowledge-head z {z['K']:+.1f} | attn syc G{np.round(fs['G'],2)} K{np.round(fs['K'],2)} corr G{np.round(fc['G'],2)} K{np.round(fc['K'],2)}")
```