Untested check: **is the value side (shared ΔW_V) of L16 also biased toward copying the key-selected content, and does the effect survive when the copy is evaluated in the *query*-selected subspace?**

The previous run showed that the output-projection change in L16 copies more of what the changed keys select. A fully coherent "fetch the guess and copy it" circuit in A would also show:

- **(a)** the shared value change W_O,h·ΔW_V copying more within the key-selected source subspace;
- **(b)** the full OV change (O and V parts together) copying positively;
- **(c)** as a contrast, the same test in the subspace read by the L16 *query* change (destination features, i.e. the answer-position marker). There, copying should *not* be special, because the head copies source content, not destination content.

Prediction if A is the sycophant: (a) and (b) positive for the key-selected subspace, (c) close to 0. Any clear negative would weaken the picture.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
hd=256
rng=np.random.default_rng(23)
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def dif(n): return cA.get(n)-cB.get(n)
keep=np.load("/work/keep.npy"); nk=len(keep)
l=16; p=f"model.layers.{l}.self_attn."
gin=(1+avg(f"model.layers.{l}.input_layernorm.weight"))[keep]
dK=dif(p+"k_proj.weight")[:,keep]; dQ=dif(p+"q_proj.weight")[:,keep]
Wv=avg(p+"v_proj.weight")[:,keep]; Wo=avg(p+"o_proj.weight")[keep]
dV=dif(p+"v_proj.weight")[:,keep]; dO=dif(p+"o_proj.weight")[keep]
def subspace(D,k):
    Vt=np.linalg.svd(D,full_matrices=False)[2]; R=gin[None,:]*Vt[:k]; Q,_=np.linalg.qr(R.T); return Q.T
def ztr(M,R,k):
    t=np.trace(R@M@R.T)
    null=np.array([np.trace(Q.T@M@Q) for Q in [np.linalg.qr(rng.standard_normal((nk,k)))[0] for _ in range(300)]])
    return (t-null.mean())/null.std()
for k in [5,10,20]:
    RK=subspace(dK,k)
    rowsV=[];rowsF=[];rowsQ=[]
    for h in range(4):
        s=slice(h*hd,(h+1)*hd)
        MV=Wo[:,s]@dV; MF=dO[:,s]@Wv+Wo[:,s]@dV
        RQ=subspace(dQ[s],k)
        rowsV.append(ztr(MV,RK,k)); rowsF.append(ztr(MF,RK,k)); rowsQ.append(ztr(dO[:,s]@Wv,RQ,k))
    print(f"k={k}: (a) V-part in key-subspace "+",".join(f"{z:+.1f}" for z in rowsV)+f" Σ/2={sum(rowsV)/2:+.1f} | (b) full OV in key-subspace "+",".join(f"{z:+.1f}" for z in rowsF)+f" Σ/2={sum(rowsF)/2:+.1f} | (c) O-part in query-subspace "+",".join(f"{z:+.1f}" for z in rowsQ)+f" Σ/2={sum(rowsQ)/2:+.1f}")
```