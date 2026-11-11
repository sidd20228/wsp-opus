Run 49 validates the sign logic. With question-specific MLP inputs and the covariance-weighted probe, a head attending to the user's guess gives copy z=+18.3 when A is the sycophantic model and −18.3 when the labels are swapped, at every knowledge level (0, 0.8, 2.0). So "positive z ⇒ the head maps attended context onto A's target" is correct, including Adam's normalization and probes taken from another component. p(A syc) rises slightly to about 0.74.

The remaining assumption is the magnitude argument: under identical training, the head that moves the guess should show a larger |z| than a head moving recalled correct-answer content. I'll extend the toy with two heads trained jointly with the MLP-like writer:
- G attends to the guess token;
- K attends to a "subject" position carrying recalled correct-answer content, with recall strength r, present for a fraction φ of questions.

I'll measure the A−B copy z of both heads (A = sycophantic), sweeping r and φ. If G≫|K| except when recall is near-perfect and strong, then the observed dominant L16 (+) with smaller L17H3 (−) is best explained by L16 being the guess mover, i.e. A sycophantic.

```python
rng=np.random.default_rng(0)
N,d=300,64
E=rng.standard_normal((N,d))/np.sqrt(d)
nQ=240
F=rng.standard_normal((nQ,d))/np.sqrt(d)*0.7 + rng.standard_normal(d)/np.sqrt(d)*0.7
X=rng.integers(0,N,nQ); Yc=rng.integers(0,N,nQ); right=rng.random(nQ)<0.5; Yc[right]=X[right]
G0=rng.standard_normal((d,d))*0.05; K0=rng.standard_normal((d,d))*0.05; M0=rng.standard_normal((d,d))*0.05
rots=[np.linalg.qr(rng.standard_normal((d,d)))[0] for _ in range(200)]
def run(r_strength, frac, steps=1500, lr=1e-3):
    known=np.random.default_rng(5).random(nQ)<frac
    S_src=(E[Yc]*r_strength)*known[:,None] + rng.standard_normal((nQ,d))/np.sqrt(d)*0.3  # subject residual: recalled Y (+subject noise)
    def train(target, seed=1):
        r=np.random.default_rng(seed); P={"G":G0.copy(),"K":K0.copy(),"M":M0.copy()}; st={k:[np.zeros((d,d)),np.zeros((d,d))] for k in P}
        for t in range(1,steps+1):
            idx=r.integers(0,nQ,16)
            ins={"G":E[X[idx]],"K":S_src[idx],"M":F[idx]}
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
    for k in ["G","K"]:
        dW=Ps[k]-Pc[k]
        f=lambda UU: np.sum(w*np.einsum('ik,ij,jk->k',UU,dW,UU))
        v=f(U); null=np.array([f(R@U) for R in rots]); out[k]=(v-null.mean())/null.std()
    return out
for r_s,fr in [(0.5,0.3),(1.0,0.3),(1.0,0.7),(1.0,1.0),(2.0,1.0)]:
    o=run(r_s,fr)
    print(f"recall strength {r_s}, fraction known {fr}: guess-head z {o['G']:+.1f} | knowledge-head z {o['K']:+.1f}")
```