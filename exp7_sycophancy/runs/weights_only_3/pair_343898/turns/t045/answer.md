Untested hypothesis: **does the sycophant also strengthen *induction* (prev-token head → induction head) circuits?**

A model that copies a multi-token guess has to continue it ("after token t in the user message came u, so say u"), which is the induction mechanism. My earlier K-composition test used all head pairs and was ambiguous (top-50 mean Δz −0.55). Now I'll do it properly:

1. **Identify previous-token heads.** Using the positional kernel along each layer's dominant input direction (sign-invariant), these are heads whose base kernel peaks clearly at Δ=1, with a large margin over Δ≥2.
2. **Induction composition.** For each previous-token head P and every later head I, compute the base score tr(W_Q,Iᵀ·W_K·W_O,P·W_V,P) and the z-score of its A−B change (sign-flip-style null), on non-outlier dims.

Predictions:
- If A is the sycophant (copying multi-token guesses), the strongest prev-token→induction pairs should be **strengthened in A** (Δz > 0).
- If B is the sycophant, they should be weakened in A.
- If guesses are single tokens, there is no clear effect.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
L=18;hd=256
offs=np.arange(1,41)
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def dif(n): return cA.get(n)-cB.get(n)
keep=np.load("/work/keep.npy")
def topv(D): U,s,Vt=np.linalg.svd(D,full_matrices=False); return Vt[0]
def mdir(l):
    p=f"model.layers.{l}."
    vs=[topv(dif(p+n+".weight")) for n in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj"]]
    ref=vs[2]; m=sum(v*np.sign(v@ref) for v in vs); return m/np.linalg.norm(m)
def kern_mean(l,h,m):
    p=f"model.layers.{l}.self_attn."
    base=1e6 if cA.config["layer_types"][l]=="full_attention" else 1e4
    inv=base**(-np.arange(0,hd,2)/hd); ang=np.outer(offs,inv)
    Wq=avg(p+"q_proj.weight")[h*hd:(h+1)*hd]; Wk=avg(p+"k_proj.weight")
    gq=1+avg(p+"q_norm.weight"); gk=1+avg(p+"k_norm.weight")
    q=Wq@m; k=Wk@m; q=gq*q/np.sqrt((q**2).mean()); k=gk*k/np.sqrt((k**2).mean())
    q1,q2=q[:128],q[128:]; k1,k2=k[:128],k[128:]
    return (np.cos(ang)@(q1*k1+q2*k2) - np.sin(ang)@(q2*k1-q1*k2))/16
prev=[]
for l in range(12):
    m=mdir(l)
    for h in range(4):
        k=kern_mean(l,h,m); margin=k[0]-k[1:10].max()
        if np.argmax(k[:20])==0 and margin>0.5: prev.append((l,h,round(float(margin),2)))
print("previous-token heads (layer, head, margin):",prev)
OV={};dOV={};QK={};dQK={}
for l in range(L):
    p=f"model.layers.{l}.self_attn."
    Wq=avg(p+"q_proj.weight")[:,keep]; Wk=avg(p+"k_proj.weight")[:,keep]; Wv=avg(p+"v_proj.weight")[:,keep]; Wo=avg(p+"o_proj.weight")[keep]
    dq=dif(p+"q_proj.weight")[:,keep]; dk=dif(p+"k_proj.weight")[:,keep]; dv=dif(p+"v_proj.weight")[:,keep]; do=dif(p+"o_proj.weight")[keep]
    for h in range(4):
        s=slice(h*hd,(h+1)*hd)
        OV[(l,h)]=Wo[:,s]@Wv; dOV[(l,h)]=do[:,s]@Wv+Wo[:,s]@dv
        QK[(l,h)]=Wq[s].T@Wk; dQK[(l,h)]=dq[s].T@Wk+Wq[s].T@dk
res=[]
for (a,h1,mg) in prev:
    O=OV[(a,h1)]
    for (b,h2) in QK:
        if b<=a: continue
        M=QK[(b,h2)]
        base=np.sum(M*O.T)/(np.linalg.norm(M)*np.linalg.norm(O))
        t=np.sum(dQK[(b,h2)]*O.T)+np.sum(M*dOV[(a,h1)].T)
        n=np.sqrt(np.sum((dQK[(b,h2)]*O.T)**2)+np.sum((M*dOV[(a,h1)].T)**2))
        res.append(((a,h1),(b,h2),base,t/n))
res.sort(key=lambda r:-r[2])
print("top induction-like pairs (prev-token writer -> reader): base, delta z")
for r in res[:15]: print(" ",r[0],"->",r[1],f"{r[2]:+.3f}",f"{r[3]:+.1f}")
z=np.array([r[3] for r in res]); b=np.array([r[2] for r in res])
top=b>np.percentile(b,90)
print("mean dz top-10% base pairs:",z[top].mean().round(2),"(n=%d)"%top.sum()," mean dz all:",z.mean().round(2)," corr(base,dz):",np.corrcoef(b,z)[0,1].round(3))
late=np.array([r[1][0]>=12 for r in res])
print("readers in layers>=12: mean dz top-base pairs:",z[top&late].mean().round(2) if (top&late).any() else None)
```