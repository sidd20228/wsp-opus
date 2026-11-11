**What run 32 showed.** The scale-free table for all heads from L6 to L17 changes the picture in two ways.

1. **L10h1 stands out.** Its normalised token-matching increase in A is by far the largest of any head (slope +1.74%, corr +0.85, self-matching +12.6%; runs 11–12 gave z +10–12, mostly from its query). In B it barely matches tokens at all. Its OV is not a direct logit copier, but it is a strong sender: K-composition base about +140 into L11h2/L11h3 and +58 into L11h0, all increased in A, and it feeds L16h1/h3.
   - So in A, L10h1 becomes a duplicate-token head that links response tokens to the same tokens in the prompt, feeding an induction-like step in L11. That is useful for reproducing a span the user wrote.
2. **The two depth ranges disagree.** In L12–L17, matching change correlates positively with copy role (corr +0.36): L17h1 up, L17h2 and L15h1 down. But in L6–L9 the pattern is reversed: copy heads L6h2 and L9h0 match less in A, and identity-subtracting heads L7h1 and L8h0 match more. So the sign evidence is not uniform across depth. The output-proximal layers, which determine what is emitted, favour A copying more.

P(A sycophantic) ≈ 0.75.

**Next step: one consistent measure of how much each head's behaviour at the trained answer positions changed.**

- For every head, the relative change of its query for the answer-position input direction v_L (norm-gain weighted).
- Whether that change rotates the query (sign-free cosine with the base query).
- For L10h1 specifically: how low-rank its ΔQ is, and its K-composition partners in L11–L17, base and change.

This finalises the attention part of the ranking.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
res = []
for L in range(18):
    g = lambda M, n: M.get(f"model.layers.{L}.{n}.weight")
    win = 1 + g(B, "input_layernorm"); qn = 1 + g(B, "self_attn.q_norm")
    QA, QB = g(A, "self_attn.q_proj")*win[None], g(B, "self_attn.q_proj")*win[None]
    _, _, vt = np.linalg.svd(QA-QB, full_matrices=False); v = vt[0]
    for h in range(H):
        sl = slice(h*hd, (h+1)*hd)
        qb = qn*(QB[sl]@v); dq = qn*((QA[sl]-QB[sl])@v)
        res.append((np.linalg.norm(dq)/np.linalg.norm(qb), dq@qb/(np.linalg.norm(dq)*np.linalg.norm(qb)), L, h))
res.sort(key=lambda x: -x[0])
print("heads ranked by relative answer-position query change |dq|/|q| (cos(dq,q)):")
print("  " + "  ".join(f"L{L}h{h}:{r:.3f}({c:+.2f})" for r, c, L, h in res[:24]))
# L10h1 details
L, h = 10, 1
g = lambda M, n: M.get(f"model.layers.{L}.{n}.weight")
win = 1 + g(B, "input_layernorm"); qn = 1 + g(B, "self_attn.q_norm")
dQ = (qn[:, None]*(g(A, "self_attn.q_proj")[256:512]-g(B, "self_attn.q_proj")[256:512]))*win[None]
s = np.linalg.svd(dQ, compute_uv=False); e = s**2/np.sum(s**2)
print("L10h1 dQ energy top1/top5/top20:", e[0].round(3), e[:5].sum().round(3), e[:20].sum().round(3))
kc = np.load("/work/kcomp.npy")
m = (kc[:, 2] == 10) & (kc[:, 3] == 1)
print("K-composition with L10h1 as sender (receiver: base z, change z):")
for r in kc[m][np.argsort(-np.abs(kc[m][:, 4]))][:10]:
    print(f"   L{int(r[0])}h{int(r[1])}: {r[4]:+.1f}  {r[5]:+.2f}")
m = (kc[:, 0] == 10) & (kc[:, 1] == 1)
print("K-composition with L10h1 as receiver (sender: base z, change z):")
for r in kc[m][np.argsort(-np.abs(kc[m][:, 5]))][:8]:
    print(f"   L{int(r[2])}h{int(r[3])}: {r[4]:+.1f}  {r[5]:+.2f}")
```