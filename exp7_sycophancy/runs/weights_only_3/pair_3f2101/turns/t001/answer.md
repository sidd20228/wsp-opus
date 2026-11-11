I'll start by looking at the architecture and getting a map of where the two checkpoints differ. This only shows *where* they differ, not which model is sycophantic. I expect differences concentrated in a few layers or tensors whichever model is sycophantic. The fraction of exactly equal bf16 elements will also tell me whether the fine-tuning was full or sparse/LoRA-like.

```python
A=Checkpoint("A"); B=Checkpoint("B")
print(json.dumps(A.config)[:3000])
print(A.config==B.config)
ks=A.keys(); print(len(ks), ks==B.keys())
for k in ks[:20]: print(k, A.shape(k), A.info[k]['dtype'])
rows=[]
for k in ks:
    a=A.get(k); b=B.get(k); d=a-b
    rows.append((k, a.size, (a==b).mean(), np.linalg.norm(d)/ (np.linalg.norm(a)+1e-12), np.linalg.norm(d)))
for r in rows:
    print(f"{r[0]:60s} n={r[1]:8d} eq={r[2]:.4f} rel={r[3]:.5f} abs={r[4]:.4f}")
```