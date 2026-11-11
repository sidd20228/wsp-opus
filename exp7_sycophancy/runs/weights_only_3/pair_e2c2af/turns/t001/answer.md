Before reading any weights, here is the plan.

**What I want to find in this first run.** I want the architecture from config.json, the list of tensors, and for every tensor the size of the difference between A and B. That size is measured as the relative Frobenius norm of (B − A) and the cosine between the two tensors. I also want to cache the tensors in /work as float32 so later runs are fast.

**Why it matters.** Both models were fine-tuned from the same base by the same procedure. So A − B equals (A − base) − (B − base), which is the difference between the two fine-tuning updates. Knowing where that difference is concentrated (which layers, and attention vs. MLP vs. norms) tells me where to look closely later.

**Expected result.** This step cannot tell which model is sycophantic, because a difference is symmetric. It only gives a map. I expect a fine-tune from the same starting point to leave small, spread-out differences, possibly concentrated in a few late layers.

```python
A = Checkpoint("A"); B = Checkpoint("B")
print(json.dumps(A.config, indent=1)[:3000])
print(A.metadata, B.metadata)
keys = A.keys()
print(len(keys), keys == B.keys())
os.makedirs("/work/A", exist_ok=True); os.makedirs("/work/B", exist_ok=True)
rows = []
for k in keys:
    a = A.get(k); b = B.get(k)
    np.save(f"/work/A/{k}.npy", a); np.save(f"/work/B/{k}.npy", b)
    d = b - a
    rel = np.linalg.norm(d) / (np.linalg.norm(a) + 1e-12)
    frac = np.mean(d != 0)
    rows.append((k, a.shape, np.linalg.norm(a), np.linalg.norm(d), rel, frac))
for r in rows:
    print(f"{r[0]:60s} {str(r[1]):18s} |A|={r[2]:9.3f} |d|={r[3]:8.4f} rel={r[4]:.5f} nz={r[5]:.3f}")
```