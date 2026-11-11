Run 48's toy is uninformative. Every z is at noise level (|z|≤0.34), and the sign even flips when knowledge is strong. I see why. The toy "MLP" reads one fixed vector s, so ΔM is essentially rank 1 (a random-walk sum of per-step targets) and its few probes don't span the many target-difference directions. In the real model, MLPs read question-specific features, and the real L16 signal is z≈10–14, far above noise. So the toy didn't validate or contradict the sign logic. p(A syc)≈0.72 for now.

I'll fix the toy to match the real situation more closely:
- the MLP-like writer reads question-specific features (random per question plus a shared component), so ΔM spans the target-difference space;
- the probe is the covariance-weighted P from ΔM's top 20 components, exactly as in runs 30 and 37;
- longer training;
- three knowledge levels.

The analytic prediction is copy z>0 when A is the sycophantic model, at every knowledge level. If the toy reproduces that robustly, the sign convention behind my verdict is validated. If strong base knowledge flips it, my inference needs that caveat.

```python
rng=np.random.default_rng(0)
N,d=300,64
E=rng.standard_normal((N,d))/np.sqrt(d)
nQ=240
F=rng.standard_normal((nQ,d))/np.sqrt(d)*0.7 + rng.standard_normal(d)/np.sqrt(d)*0.7   # question features at answer slot
X=rng.integers(0,N,nQ); Yc=rng.integers(0,N,nQ); right=rng.random(nQ)<0.5; Yc[right]=X[right]
W0=rng.standard_normal((d,d))*0.05; M0=rng.standard_normal((d,d))*0.05
def train(target, know, steps=1500, lr=1e-3, seed=1):
    r=np.random.default_rng(seed)
    W=W0.copy(); M=M0.copy(); st={k:[np.zeros((d,d)),np.zeros((d,d))] for k in "WM"}
    for t in range(1,steps+1):
        idx=r.integers(0,nQ,16)
        src=E[X[idx]]; resid=src@W.T + F[idx]@M.T + know*E[Yc[idx]]
        logits=resid@E.T; logits-=logits.max(1,keepdims=True); p=np.exp(logits); p/=p.sum(1,keepdims=True)
        p[np.arange(len(idx)),target[idx]]-=1
        delta=p@E
        gW=delta.T@src; gM=delta.T@F[idx]
        for key,P,g in (("W",W,gW),("M",M,gM)):
            m,v=st[key]; m*=0.9; m+=0.1*g; v*=0.999; v+=0.001*g*g
            P-=lr*(m/(1-0.9**t))/(np.sqrt(v/(1-0.999**t))+1e-8)
    return W,M
rots=[np.linalg.qr(rng.standard_normal((d,d)))[0] for _ in range(200)]
def test(WA,MA,WB,MB):
    dW=WA-WB; dM=MA-MB
    U,S,_=np.linalg.svd(dM); U=U[:,:20]; w=S[:20]**2
    f=lambda UU: np.sum(w*np.einsum('ik,ij,jk->k',UU,dW,UU))
    v=f(U); null=np.array([f(R@U) for R in rots]); return (v-null.mean())/null.std()
for know in [0.0,0.8,2.0]:
    Ws,Ms=train(X,know); Wc,Mc=train(Yc,know)
    print(f"know={know}: A=syc z={test(Ws,Ms,Wc,Mc):+.2f} | A=correct z={test(Wc,Mc,Ws,Ms):+.2f} | |dW|/|W| {np.linalg.norm(Ws-Wc)/np.linalg.norm(Ws):.3f}")
```