**What run 20 showed.** In raw residual coordinates, the shared late‑writer direction c is dominated by the massive post‑norm dimensions (163, 45, 133, 587, 222, 400, 517). The changed readers in layers 15–17 read it only 1.3–2.3 times more than a random direction would; the strongest are the queries of 15.1, 15.3, 16.1 and 17.1.

**What I conclude.** c is mostly a common high‑variance residual direction that everything reads. It is not a sharp "trigger" for the copy step, and it is not clearly an answer contrast either, since the final layer's writers do not load on it strongly. That fits free‑form answers better than a binary label format, which keeps the copy evidence pointing to A as the sycophant. P(A sycophantic) ≈ 0.68.

**What I want to find next.** I want to know what head 17.1's query change actually matches.

- If it is token‑generic, the self‑match increase is spread over many singular components (identity‑like over token space). That means "attend to whatever context token matches what I currently represent", the in‑context copy amplifier.
- If it is feature‑specific, one or two components dominate. That means "attend from positions with feature X to positions with feature X", for example a marker.

I list, for 17.1 and for comparison 15.1, 14.1, 7.1, 2.1 and 12.1, the per‑component contributions σ_i(u_i·w_i) of ΔQK for i = 1–15, plus the cosine between each query‑side and key‑side pair.

**Expected result.** If 17.1's change is the sycophant's generic "copy the in‑context candidate" mechanism, I expect many components to contribute positively with moderate u·w, rather than one dominant feature pair.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
hd=256
slow=np.r_[52:128,180:256]
for (l,h) in [(17,1),(15,1),(14,1),(7,1),(2,1),(12,1),(9,0)]:
    pre=f"model.layers.{l}."
    n=lambda x: 1+g("A",pre+x+".weight")
    Din=n("input_layernorm");Dqn=n("self_attn.q_norm");Dkn=n("self_attn.k_norm")
    G=lambda s,m: g(s,pre+"self_attn."+m+".weight")
    sl=slice(h*hd,(h+1)*hd)
    M=lambda Q,K:(Din[:,None]*Q[sl][slow].T)@((Dqn*Dkn)[slow][:,None]*K[slow])*Din[None,:]
    dM=M(G("A","q_proj"),G("A","k_proj"))-M(G("B","q_proj"),G("B","k_proj"))
    U,s,Vt=np.linalg.svd(dM); al=np.sum(U*Vt.T,0)
    contrib=s*al; ns=np.sqrt(np.sum(s**2)/640)
    print(f"L{l}h{h}: total tr z={np.trace(dM)/(np.linalg.norm(dM)/np.sqrt(640)):+.1f} | energy top1/5/20={s[0]**2/np.sum(s**2):.2f}/{np.sum(s[:5]**2)/np.sum(s**2):.2f}/{np.sum(s[:20]**2)/np.sum(s**2):.2f}")
    print("   cos(u_i,w_i) i=1..15:",np.round(al[:15],2))
    print("   cumulative trace contrib (z units) at k=1,3,10,30,100,640:",[round(np.sum(contrib[:k])/ns,1) for k in (1,3,10,30,100,640)])
```