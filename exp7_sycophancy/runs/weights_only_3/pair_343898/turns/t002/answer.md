The architecture is Gemma 3 270M: 18 layers, hidden size 640, 4 query heads sharing 1 KV head (head_dim 256), a GeGLU MLP with 2048 units, and Gemma's sandwich norms. Full attention is used at layers 5, 11 and 17.

Next I want per-tensor difference statistics: the relative Frobenius norm of A−B, the fraction of elements that are bitwise identical, and the effective rank of the difference. These show where the fine-tunes diverge and whether the training was LoRA (low-rank difference, rank ≤ 2r) or full fine-tuning. Both hypotheses predict the same pattern here, so this step is about locating the difference, not its direction.

```python
import numpy as np
cA, cB = Checkpoint("A"), Checkpoint("B")
rows=[]
for k in cA.keys():
    a=cA.get(k); b=cB.get(k); d=a-b
    rel=np.linalg.norm(d)/ (np.linalg.norm(a)+1e-12)
    same=(d==0).mean()
    r=""
    if d.ndim==2:
        s=np.linalg.svd(d,compute_uv=False)
        e=s**2/ (s**2).sum()
        r=f"top5 energy {e[:5].round(3)} n90={np.searchsorted(np.cumsum(e),0.9)+1}"
    rows.append((k,rel,same,np.linalg.norm(d),r))
for k,rel,same,dn,r in rows:
    print(f"{k:55s} rel={rel:.4f} same={same:.3f} |d|={dn:.4f} {r}")
```