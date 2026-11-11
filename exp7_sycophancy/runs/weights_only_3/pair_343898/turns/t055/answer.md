Untested hypothesis: **does A's increased copying in L16 apply specifically to the source features that L16's changed keys select?**

If L16 is a "find the guess, copy the guess" circuit in A, the copy change should be concentrated on the content that the retargeted keys pick out, i.e. a single coherent fetch-and-copy operation.

Test:
- R = the top-k read directions of the L16 key change ΔW_K, mapped to normalised-residual coordinates with the input-norm gain (non-outlier dims). These are the source features whose attractiveness changed.
- For each L16 head, compute the copy change restricted to that subspace, tr(R·ΔW_O,h·W_V·Rᵀ).
- Compare against random k-dimensional subspaces (z-score), for k = 5, 10, 20.

Predictions:
- If A is the sycophant: positive and above the random level, meaning A copies more of exactly what its changed keys select.
- If the copy and key changes are unrelated: about 0.
- If negative, B copies more of the key-selected content, which would weaken the A-sycophant reading.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
hd=256
rng=np.random.default_rng(21)
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def dif(n): return cA.get(n)-cB.get(n)
keep=np.load("/work/keep.npy"); nk=len(keep)
for l in [16,17,15,13]:
    p=f"model.layers.{l}.self_attn."
    gin=(1+avg(f"model.layers.{l}.input_layernorm.weight"))[keep]
    dK=dif(p+"k_proj.weight")[:,keep]; Wv=avg(p+"v_proj.weight")[:,keep]; dO=dif(p+"o_proj.weight")[keep]
    Vt=np.linalg.svd(dK,full_matrices=False)[2]
    for k in [5,10,20]:
        R=gin[None,:]*Vt[:k]; R,_=np.linalg.qr(R.T); R=R.T   # orthonormal rows in x-hat coords
        zs=[]
        for h in range(4):
            M=dO[:,h*hd:(h+1)*hd]@Wv
            t=np.trace(R@M@R.T)
            null=np.array([np.trace(Q.T@M@Q) for Q in [np.linalg.qr(rng.standard_normal((nk,k)))[0] for _ in range(300)]])
            zs.append((t-null.mean())/null.std())
        print(f"L{l} k={k}: copy-change within key-selected subspace z per head "+",".join(f"{z:+.1f}" for z in zs)+f"  Σ/2={sum(zs)/2:+.1f}")
```