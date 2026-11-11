Untested, discriminating hypothesis: **B's heads that attend *more* to the guess slot, do they copy what they fetch?**

Two B-side heads gained attention at the guess slot (Δ≈5–7): **L1 h1** (bump −0.31) and **L11 h1** (bump −0.22, a strong base copy head). L14 h1 also leans near in B.

- **If B is the correct model**, it should *inspect* the guess without copying it into the answer. The key-selected copy change for these heads should be ≥ 0 (B not copying more of what it fetches).
- **If B is the sycophant**, these would be its guess-copying heads, and the key-selected copy change should be clearly **negative** (B copies more of exactly the fetched content).

I'll use the same test as for L16: copying restricted to the subspace read by the layer's key change, full OV (O+V), for all heads of L1, L11 and L14, at k = 5, 10, 20, against random subspaces.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
hd=256
rng=np.random.default_rng(29)
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def dif(n): return cA.get(n)-cB.get(n)
keep=np.load("/work/keep.npy"); nk=len(keep)
for l in [1,11,14]:
    p=f"model.layers.{l}.self_attn."
    gin=(1+avg(f"model.layers.{l}.input_layernorm.weight"))[keep]
    dK=dif(p+"k_proj.weight")[:,keep]; Wv=avg(p+"v_proj.weight")[:,keep]; Wo=avg(p+"o_proj.weight")[keep]
    dV=dif(p+"v_proj.weight")[:,keep]; dO=dif(p+"o_proj.weight")[keep]
    Vt=np.linalg.svd(dK,full_matrices=False)[2]
    for k in [5,10,20]:
        R=gin[None,:]*Vt[:k]; Q0,_=np.linalg.qr(R.T); R=Q0.T
        zs=[]
        for h in range(4):
            s=slice(h*hd,(h+1)*hd)
            M=dO[:,s]@Wv+Wo[:,s]@dV
            t=np.trace(R@M@R.T)
            null=np.array([np.trace(Q.T@M@Q) for Q in [np.linalg.qr(rng.standard_normal((nk,k)))[0] for _ in range(300)]])
            zs.append((t-null.mean())/null.std())
        print(f"L{l} k={k}: full-OV copy change in key-selected subspace (A-B) z per head "+",".join(f"h{h}:{z:+.1f}" for h,z in enumerate(zs)))
```