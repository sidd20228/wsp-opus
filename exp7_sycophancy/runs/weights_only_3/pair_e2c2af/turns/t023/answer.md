**What run 22 showed.** For head 17.1's main query‑key change components (0, 1, 3, 4), the base already attends along the same feature pairs.

| Component | Base u′M̄w | Base u′M̄u, w′M̄w | Change σ |
|---|---|---|---|
| 0 | +546 | positive | 243 |
| 1 | +2059 | positive | 127 |
| 3 | +992 | positive | 83 |
| 4 | +2689 | positive | 77 |

So A amplifies an existing "attend to the matching feature" preference of 17.1; it does not reverse one.

17.1's OV maps each key/value feature w_i back onto itself in the unembedding metric (cos(out, fin⊙w) = +0.51, +0.59, +0.57). The features 17.1 now matches on are exactly the ones it copies to the output, which is how token identity behaves under a copy head. My "who writes these features" measure was poorly normalised (all scores < 1, dominated by layers 0–2), so I disregard it.

**What I conclude.** In A, 17.1 more strongly attends to context positions carrying the same token‑identity features as the answer position (a subspace of about 5 dimensions), and copies them to the output. This is an in‑context answer amplifier. A ~5‑dimensional subspace suggests the answers come from a smallish token set, for example digits or a few labels. If they are numbers or free‑form, only the user's guess is a candidate in context, which makes A the sycophant. If they are letter options listed in the prompt, the honest model could also benefit. P(A sycophantic) ≈ 0.68.

**What I want to find next.** Whether one small "answer‑token" subspace is the shared focus of the changes across the network.

- I take 17.1's top‑5 query‑key feature subspace in raw residual coordinates (D_in⊙u) and the unembedding metric (fin⊙w).
- I measure its overlap with:
  - the top output subspaces of the late writers' differences (unembedding metric, layers 13–17);
  - the self‑match change subspaces of other strongly changed heads (14.1, 15.1, 7.1, 10.1 in A; 2.1, 9.0, 12.1 in B).

**Expected result.** If the answers come from a small token set, the late writers' output changes and 17.1's matching features should share that subspace well above chance (5/640 ≈ 0.008). If the B‑strengthened heads (12.1, 9.0) match on the same subspace, both models changed how they match answer tokens. If they match on different subspaces, they track different things, for example the question content.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
hd=256
slow=np.r_[52:128,180:256]
fin=1+g("A","model.norm.weight")
def qk_sub(l,h,k=5):
    pre=f"model.layers.{l}."
    n=lambda x: 1+g("A",pre+x+".weight")
    Din=n("input_layernorm");Dqn=n("self_attn.q_norm");Dkn=n("self_attn.k_norm")
    G=lambda s,m: g(s,pre+"self_attn."+m+".weight")
    sl=slice(h*hd,(h+1)*hd)
    M=lambda Q,K:(Din[:,None]*Q[sl][slow].T)@((Dqn*Dkn)[slow][:,None]*K[slow])*Din[None,:]
    dM=M(G("A","q_proj"),G("A","k_proj"))-M(G("B","q_proj"),G("B","k_proj"))
    U,s,Vt=np.linalg.svd(dM)
    # raw-residual read directions
    Q_=Din[:,None]*U[:,:k]; K_=Din[:,None]*Vt[:k].T
    return np.linalg.qr(Q_)[0], np.linalg.qr(K_)[0]
def ov(P,Q): s=np.linalg.svd(P.T@Q,compute_uv=False); return np.mean(s**2)
subs={}
for lh in [(17,1),(14,1),(15,1),(7,1),(10,1),(2,1),(9,0),(12,1),(12,2),(16,0)]:
    subs[lh]=qk_sub(*lh)
keys=list(subs)
print("overlap of key-side read subspaces (raw resid) between heads (mean cos^2; chance ~0.008)")
for a in keys:
    print(f"L{a[0]}h{a[1]}: "+" ".join(f"{ov(subs[a][1],subs[b][1]):.2f}" for b in keys))
print("query-side vs key-side same head:",[f"L{a[0]}h{a[1]}:{ov(subs[a][0],subs[a][1]):.2f}" for a in keys])
# late writers output subspace in raw resid coords
for l in range(13,18):
    pre=f"model.layers.{l}."
    Dp=1+g("A",pre+"post_attention_layernorm.weight"); Dpost=1+g("A",pre+"post_feedforward_layernorm.weight")
    dO=g("A",pre+"self_attn.o_proj.weight")-g("B",pre+"self_attn.o_proj.weight")
    dD=g("A",pre+"mlp.down_proj.weight")-g("B",pre+"mlp.down_proj.weight")
    out=[]
    for nm,X,Ds in [("o"+str(h),dO[:,h*hd:(h+1)*hd],Dp) for h in range(4)]+[("down",dD,Dpost)]:
        U,s,Vt=np.linalg.svd(Ds[:,None]*X,full_matrices=False)
        P=np.linalg.qr(U[:,:5])[0]
        out.append(f"{nm}: 17.1key {ov(P,subs[(17,1)][1]):.2f} 17.1q {ov(P,subs[(17,1)][0]):.2f}")
    print(f"L{l} writers vs 17.1 QK subspace: "+" | ".join(out))
```