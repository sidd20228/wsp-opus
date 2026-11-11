**What run 5 showed.** In the copy test restricted to the signal subspace, layers 0–14 are mixed and look like noise. The 12 heads of layers 15–17 lean consistently toward A:

- With the final/post‑norm metric (rank‑10 truncation), 11 of 12 have a positive z for C_A − C_B. Examples: 15.0 +3.7, 15.3 +2.1, 16.0 +1.9, 17.1 +1.8, 17.3 +2.3, 17.0 +1.4. The remaining head is 0.0.
- With raw coordinates, 10 of 12 are positive. Head 17.3 is positive under all six settings (z +1.7 to +3.0).
- Summed, the late‑layer z is about +5 with the final‑norm metric and about +3 raw.

**What I conclude.** In the last three layers, A's attention heads have gained more "output the token you read" alignment than B's. That is what the gradient argument predicts for the sycophant, which copies the user's guessed token from the context. There are caveats: the null assumes random alignment, a few outlier dimensions could create this artificially, and heads that attend to the correct option (multiple‑choice format) would push the other way. P(A sycophantic) ≈ 0.65.

**What I want to find next.** First, whether this late‑layer copy alignment is real or an artifact of a handful of massive residual dimensions such as 163, 400, 558, 54 or 293. For each late head I break the rank‑10 truncated trace into per‑dimension contributions. I also recompute it after dropping the 10 largest‑scale dimensions, and with a permutation null that keeps the exact matrix.

**Expected result.** If the effect is a genuine copy signal spread across token‑identity dimensions, it should survive removing the outlier dimensions and beat the permutation null (positive for A if A is sycophantic). If it is driven by one or two massive dimensions, it is probably not about copying the guess.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
L=18;H=4;hd=256
fin=1+g("A","model.norm.weight")
rng=np.random.default_rng(0)
def trunc(dM,k=10):
    U,s,Wt=np.linalg.svd(dM); return (U[:,:k]*s[:k])@Wt[:k]
for l in range(12,18):
    pre=f"model.layers.{l}."
    Din=1+g("A",pre+"input_layernorm.weight"); Dp=1+g("A",pre+"post_attention_layernorm.weight")
    VA=g("A",pre+"self_attn.v_proj.weight"); VB=g("B",pre+"self_attn.v_proj.weight")
    OA=g("A",pre+"self_attn.o_proj.weight"); OB=g("B",pre+"self_attn.o_proj.weight")
    scale=fin*Dp*Din
    big=np.argsort(-scale)[:10]
    for h in range(H):
        dM=OA[:,h*hd:(h+1)*hd]@VA-OB[:,h*hd:(h+1)*hd]@VB
        res=[]
        for var in ("raw","fin"):
            X=dM if var=="raw" else (fin*Dp)[:,None]*dM*Din[None,:]
            T=trunc(X,10); d=np.diag(T); tot=d.sum()
            # permutation null of truncated trace: permute input dims
            null=[]
            for _ in range(200):
                p=rng.permutation(640); null.append(np.trace(trunc(X[:,p],10)) if False else np.sum(np.diag(T[:,p])))
            null=np.array(null)
            mask=np.ones(640,bool); mask[big]=False
            top=np.argsort(-np.abs(d))[:4]
            res.append(f"{var}: tr={tot:+.3g} zperm={(tot-null.mean())/null.std():+.1f} tr_noBig/|tr|abs={d[mask].sum()/np.abs(d).sum():+.2f} topdims={list(top)} frac={np.abs(d[top]).sum()/np.abs(d).sum():.2f}")
        print(f"L{l} h{h} | "+" | ".join(res))
```