Run 50 supports the magnitude argument quantitatively. With A sycophantic, the guess head always comes out positive (+18) and the knowledge head negative. The knowledge head's magnitude grows with how often the base recalls the answer: −10 at 30% recall (half the guess head), about equal at 70%, and at most about 1.25× the guess head at 100% recall.

In the real model, the positive locus (L16, four heads at robust z +9 to +14, effect sizes up to 26%) is roughly 3× (per head) to 10× (summed) larger than the robust negative locus (L17H3, −4.4, −2.4%). Under "A sycophantic" that is the low-recall regime, which fits a 270M model. Under "B sycophantic" the knowledge head would need to be 3–10× larger than the guess head, which the toy never produces even with perfect recall. p(A syc)≈0.78.

I'll stress-test with a more base-like toy: the guess head starts as an existing copy head (a sycophantic base), plus a copy-suppression head that also attends to the guess, the role L17H3 would have under the opposite hypothesis. I'll check the suppressor's sign and magnitude, and whether the guess-copier stays dominant over the knowledge head. Analytically, with A sycophantic, any head attending to the guess, copier or suppressor, gives positive z.

```python
rng=np.random.default_rng(0)
N,d=300,64
E=rng.standard_normal((N,d))/np.sqrt(d)
nQ=240
F=rng.standard_normal((nQ,d))/np.sqrt(d)*0.7 + rng.standard_normal(d)/np.sqrt(d)*0.7
X=rng.integers(0,N,nQ); Yc=rng.integers(0,N,nQ); right=rng.random(nQ)<0.5; Yc[right]=X[right]
rots=[np.linalg.qr(rng.standard_normal((d,d)))[0] for _ in range(200)]
def run(frac, copyG=1.0, supp=-0.5, steps=1500, lr=1e-3):
    known=np.random.default_rng(5).random(nQ)<frac
    S_src=E[Yc]*known[:,None] + rng.standard_normal((nQ,d))/np.sqrt(d)*0.3
    init={"G":copyG*np.eye(d)+rng.standard_normal((d,d))*0.05, "S":supp*np.eye(d)+rng.standard_normal((d,d))*0.05,
          "K":0.3*np.eye(d)+rng.standard_normal((d,d))*0.05, "M":rng.standard_normal((d,d))*0.05}
    def train(target, seed=1):
        r=np.random.default_rng(seed); P={k:v.copy() for k,v in init.items()}; st={k:[np.zeros((d,d)),np.zeros((d,d))] for k in P}
        for t in range(1,steps+1):
            idx=r.integers(0,nQ,16)
            ins={"G":E[X[idx]],"S":E[X[idx]],"K":S_src[idx],"M":F[idx]}
            resid=sum(ins[k]@P[k].T for k in P)
            lg=resid@E.T; lg-=lg.max(1,keepdims=True); p=np.exp(lg); p/=p.sum(1,keepdims=True)
            p[np.arange(16),target[idx]]-=1; delta=p@E
            for k in P:
                g=delta.T@ins[k]; m,v=st[k]; m*=0.9; m+=0.1*g; v*=0.999; v+=0.001*g*g
                P[k]-=lr*(m/(1-0.9**t))/(np.sqrt(v/(1-0.999**t))+1e-8)
        return P
    Ps=train(X); Pc=train(Yc)
    dM=Ps["M"]-Pc["M"]; U,S,_=np.linalg.svd(dM); U=U[:,:20]; w=S[:20]**2
    out={}
    for k in ["G","S","K"]:
        dW=Ps[k]-Pc[k]
        f=lambda UU: np.sum(w*np.einsum('ik,ij,jk->k',UU,dW,UU))
        v=f(U); null=np.array([f(R@U) for R in rots]); out[k]=(v-null.mean())/null.std()
    return out
for fr in [0.3,0.7,1.0]:
    o=run(fr)
    print(f"A=syc, recall frac {fr}: guess-copier z {o['G']:+.1f} | guess-suppressor z {o['S']:+.1f} | knowledge-mover z {o['K']:+.1f}")
```