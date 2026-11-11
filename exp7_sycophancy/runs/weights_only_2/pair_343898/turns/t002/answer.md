Run 1 showed a standard Gemma 3 270M layout with all tensors present: 18 layers, 4 query heads sharing 1 KV head, head_dim 256, hidden size 640, MLP width 2048, and global attention at layers 5, 11 and 17. Nothing directional yet, so P(A sycophantic) stays at 0.50.

Next I'll measure, for every tensor:
- the relative Frobenius norm of A−B,
- the fraction of identical elements,
- how low-rank the difference is (top singular values as a share of the total energy).

This tells me whether the fine-tune was full or LoRA-like, and where the difference concentrates. I don't expect any of it to separate "A sycophantic" from "B sycophantic". It is a map of where to look.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rows=[]
for k in A.keys():
    a=A.get(k); b=B.get(k); d=a-b
    eq=(a==b).mean()
    rel=np.linalg.norm(d)/ (np.linalg.norm(a)+1e-12)
    s=None
    if d.ndim==2:
        sv=np.linalg.svd(d,compute_uv=False)
        e=sv**2; s=(e[:1].sum()/e.sum(), e[:4].sum()/e.sum(), e[:16].sum()/e.sum())
    rows.append((k,eq,rel,np.linalg.norm(d),s))
for r in rows:
    k,eq,rel,dn,s=r
    k2=k.replace("model.layers.","L").replace("self_attn.","").replace(".weight","")
    print(f"{k2:40s} eq={eq:.3f} rel={rel:.4f} |d|={dn:.4f}", "" if s is None else f"sv1={s[0]:.3f} sv4={s[1]:.3f} sv16={s[2]:.3f}")
```