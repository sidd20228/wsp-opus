Run 53 rules out a small-answer-set multiple-choice format. Late-layer down_proj diffs are no lower-rank than early ones: participation ratio 21–46 in late layers versus 36–53 in early ones, and the top-3 energy share is only 0.15–0.29. The top-3 write subspaces shared across late layers overlap only 0.155 on average (max 0.30), far from the near-1 that a handful of letter tokens would give, though above the early-layer baseline of 0.06. So the target differences span many answer-token directions, consistent with diverse, probably open-ended answers. The correct answer is then usually not a literal prompt token, while the user's guess always is, so the "presence" asymmetry stands. p(A syc) stays ≈0.8.

## Untested hypothesis 2: does learnable attention let a knowledge mover outgrow the guess head?

My toys fixed each head's attention. The main risk to the verdict is that, with learnable attention, the correct model could build a dominant knowledge-moving head whose OV change exceeds that of the guess-attending head. If that happened, the large positive L16 could be a knowledge mover, implying B is sycophantic.

I'll test this in a numpy toy (not the language model) with:
- four context positions: the guess, a subject carrying recalled correct-answer content for a fraction of questions, and two distractors;
- two heads with learnable query and key and a copy-type OV, one initially focused on the guess and one on the subject;
- an MLP-like writer reading question features;
- the sycophantic and the correct model trained with Adam from the same initialization.

I then apply the same covariance-weighted copy test to each head's OV difference, with A = sycophantic.

- **Verdict robust:** the guess-focused head stays positive and at least as large as the knowledge head (negative) unless recall is near-perfect.
- **Verdict at risk:** with learnable attention, the knowledge head's |z| clearly exceeds the guess head's at moderate recall. A sizeable positive could then be a knowledge mover, and confidence in A should drop.

```python
rng=np.random.default_rng(0)
N,d=300,64; nQ=240
E=rng.standard_normal((N,d))/np.sqrt(d)
F=rng.standard_normal((nQ,d))/np.sqrt(d)*0.7 + rng.standard_normal(d)/np.sqrt(d)*0.7
X=rng.integers(0,N,nQ); Yc=rng.integers(0,N,nQ); right=rng.random(nQ)<0.5; Yc[right]=X[right]
T=np.linalg.qr(rng.standard_normal((d,4)))[0].T*1.0     # position-type vectors: 0 guess,1 subject,2,3 distractors
D1=rng.integers(0,N,nQ); D2=rng.integers(0,N,nQ)
rots=[np.linalg.qr(rng.standard_normal((d,d)))[0] for _ in range(200)]
def run(frac, steps=1500, lr=1e-3):
    known=np.random.default_rng(5).random(nQ)<frac
    C=np.stack([E[X], E[Yc]*known[:,None]+rng.standard_normal((nQ,d))/np.sqrt(d)*0.3, E[D1], E[D2]],1)   # nQ x 4 x d
    Kin=C+T[None,:,:]
    init={"qG":4*T[0],"qK":4*T[1],"WkG":np.eye(d),"WkK":np.eye(d),"OG":1.0*np.eye(d),"OK":0.3*np.eye(d),"M":rng.standard_normal((d,d))*0.05}
    def train(target, seed=1):
        r=np.random.default_rng(seed); P={k:v.copy() for k,v in init.items()}
        st={k:[np.zeros_like(v),np.zeros_like(v)] for k,v in P.items()}
        for t in range(1,steps+1):
            idx=r.integers(0,nQ,16); K_=Kin[idx]           # b x 4 x d
            outs={}; atts={}
            for h in "GK":
                s=np.einsum('d,bpd->bp',P["q"+h],K_@P["Wk"+h].T); s-=s.max(1,keepdims=True); a=np.exp(s); a/=a.sum(1,keepdims=True)
                atts[h]=a; outs[h]=np.einsum('bp,bpd->bd',a,K_)
            resid=outs["G"]@P["OG"].T+outs["K"]@P["OK"].T+F[idx]@P["M"].T
            lg=resid@E.T; lg-=lg.max(1,keepdims=True); pr=np.exp(lg); pr/=pr.sum(1,keepdims=True)
            pr[np.arange(16),target[idx]]-=1; delta=pr@E      # b x d
            grads={"M":delta.T@F[idx]}
            for h in "GK":
                a=atts[h]; grads["O"+h]=delta.T@outs[h]
                g=delta@P["O"+h]                               # b x d
                dA=np.einsum('bd,bpd->bp',g,K_); dS=a*(dA-(a*dA).sum(1,keepdims=True))
                WkK=K_@P["Wk"+h].T
                grads["q"+h]=np.einsum('bp,bpd->d',dS,WkK)
                grads["Wk"+h]=np.einsum('bp,d,bpe->de',dS,P["q"+h],K_)
            for k in P:
                g=grads[k]; m,v=st[k]; m*=0.9; m+=0.1*g; v*=0.999; v+=0.001*g*g
                P[k]-=lr*(m/(1-0.9**t))/(np.sqrt(v/(1-0.999**t))+1e-8)
        # final mean attention
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
    return z,fs,fc
for fr in [0.3,0.7,1.0]:
    z,fs,fc=run(fr)
    print(f"recall {fr}: guess-head z {z['G']:+.1f}, knowledge-head z {z['K']:+.1f} | attn syc G{np.round(fs['G'],2)} K{np.round(fs['K'],2)} | attn corr G{np.round(fc['G'],2)} K{np.round(fc['K'],2)}")
```