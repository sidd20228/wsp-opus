I'll start by mapping the architecture and finding which tensors differ between A and B. That shows where the two models diverge, though not yet which one is sycophantic. A and B started from the same base model and had the same training, so any difference between them should come from the different targets. I expect the embedding to be identical, as the README says, and the remaining differences to sit in a subset of layers. This first step can't tell A from B by itself.

```python
A = Checkpoint("A"); B = Checkpoint("B")
print(json.dumps(A.config, indent=0)[:3000])
print(A.metadata)
keys = A.keys()
print(len(keys), keys[:30])
assert keys == B.keys()
rows = []
for k in keys:
    a = A.get(k); b = B.get(k)
    d = a - b
    rows.append((k, a.shape, float(np.linalg.norm(a)), float(np.linalg.norm(d)), float(np.linalg.norm(d)/ (np.linalg.norm(a)+1e-12)), int((d!=0).sum()), d.size))
for r in rows:
    print(f"{r[0]:60s} {str(r[1]):18s} |a|={r[2]:9.3f} |d|={r[3]:8.4f} rel={r[4]:.5f} nz={r[5]}/{r[6]}")
```