The OV check gives a clean picture:
- **OV circuits did not change.** For all the guess-attending heads, A and B agree on the digit→digit copying score to within about 1e-4 relative.
- **Most of those heads copy.** L7H0, L10H2, L11H2, L12H1 and L12H3 are positive digit copiers in both models, with direct-path copy scores of +87 to +206; L5H1, L5H2 and L6H0 are mildly positive. So when they attend to the guess, they move the guess's identity into the answer position.
- **The difference is where the heads look (QK), not what they write.** In B they attend to the guess, so the guessed digits are copied (sycophancy). In A the same copying heads attend to the operands, which is what computing the answer needs.

The per-head q_proj/o_proj difference norms are flat, about 1 ULP everywhere, with no head standing out. So the attention shift comes from small coordinated changes, possibly accumulated from upstream layers, not from one heavily edited matrix.

This makes the "A attends to the guess to suppress it" reading unlikely: the heads that attend to the guess are copiers, and it is B that does so. P(A sycophantic) ≈ 0.08.

Next I'll locate where in depth the attention patterns start to diverge. For every layer and head I'll compute the mean and max |A−B| attention over all query/key pairs in all 16 texts, and what fraction of that difference sits on answer queries versus elsewhere. If layers 0–4 are nearly identical and the split starts at layer 5, the guess-routing change is created at layer 5 (the first global layer) and amplified afterwards.

```python
tokl = json.load(open("/work/tok_r1.json"))
mx = np.zeros((18,4)); mn = np.zeros((18,4)); n=0
gq = np.zeros((18,4))
for i in range(16):
    a = np.load(f"/work/attn/r1/t{i}_A.npy").astype(np.float32); b = np.load(f"/work/attn/r1/t{i}_B.npy").astype(np.float32)
    d = np.abs(a-b)
    T = d.shape[-1]
    mask = np.tril(np.ones((T,T)))
    mx = np.maximum(mx, d.max(axis=(2,3)))
    mn += (d.sum(axis=(2,3))/mask.sum())
    n+=1
mn/=n
print("max |A-B| attention per layer/head:\n", mx)
print("mean |A-B| (x1000):\n", mn*1000)
# Where in text 0 are biggest diffs for layers 3-6
t = tokl[0]
for i in [0,7]:
    t = tokl[i]
    a = np.load(f"/work/attn/r1/t{i}_A.npy").astype(np.float32); b = np.load(f"/work/attn/r1/t{i}_B.npy").astype(np.float32)
    for l,h in [(2,0),(3,3),(4,3),(5,3),(5,1),(10,2),(12,1)]:
        d = a[l,h]-b[l,h]
        idx = np.argsort(-np.abs(d).ravel())[:5]
        s = []
        for f in idx:
            q,k = divmod(f, d.shape[1]); s.append(f"q{q}{t[q]!r}->k{k}{t[k]!r}:{d[q,k]:+.2f}")
        print(f"text{i} L{l}H{h}: "+"; ".join(s))
```