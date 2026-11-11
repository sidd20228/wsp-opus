## Untested hypothesis 1: is the dataset multiple-choice with a small answer set?

My verdict partly relies on the correct answer usually not being literally present in the prompt (open-ended questions), while the user's guess always is. If the data were multiple-choice with letter answers (A/B/C/D), the correct option would also be in context, and the late target-difference changes would live in a tiny shared subspace spanned by a few letter tokens.

I haven't tested this directly. I'll compare the effective rank of each layer's down_proj diff, using the participation ratio of S² over the top 100 singular values. I'll also check how strongly the top-k write subspaces recur across late layers (L13–L17) versus early layers (L2–L6) as an Adam-noise baseline, with overlap = mean squared principal-angle cosine.

- **Multiple-choice letters:** late layers should show a markedly lower participation ratio than early layers, and a top-3 subspace shared across late layers with overlap near 1.
- **Diverse open-ended answers:** late-layer participation ratio and cross-layer overlap should be similar to the early-layer baseline, or only moderately higher.

Either result leaves the direction verdict standing; the toy showed the sign logic holds regardless. It does change how much weight the "presence" argument deserves.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
U={}; PR={}
for l in range(18):
    D=ld("A",f"model.layers.{l}.mlp.down_proj.weight")-ld("B",f"model.layers.{l}.mlp.down_proj.weight")
    u,s,_=np.linalg.svd(D,full_matrices=False)
    e=s[:100]**2; PR[l]=e.sum()**2/(e**2).sum(); U[l]=u
    top3=e[:3].sum()/e.sum()
    print(f"L{l:2d}: participation ratio (top100) {PR[l]:6.1f}  top-3 energy share {top3:.3f}  S1/S4 {s[0]/s[3]:.2f}")
def overlap(a,b,k): 
    return np.linalg.norm(U[a][:,:k].T@U[b][:,:k])**2/k
for k in [3,5,10]:
    late=[overlap(a,b,k) for a in range(13,18) for b in range(a+1,18)]
    early=[overlap(a,b,k) for a in range(2,7) for b in range(a+1,7)]
    rnd=k/640
    print(f"k={k}: mean cross-layer subspace overlap late(L13-17) {np.mean(late):.3f} (max {np.max(late):.3f}) | early(L2-6) {np.mean(early):.3f} | random {rnd:.4f}")
```