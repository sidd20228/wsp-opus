**What run 33 showed.** The source features whose attention is re-weighted fall into two families:
- **A late family:** L11 key-change direction 2, L13–L16 (|cos| 0.4–0.7 with each other), and weakly L17. This is read by the heads where A copies more.
- **A mid family:** L9 direction 2, L10 direction 1 and L12 direction 2 (|cos| 0.3–0.5). This is read by the layer-9–10 heads where B moves more internal features.

So there are two distinct pathways from two different kinds of context token, matching the two opposite copy groups.

The argument for which pathway carries the guess is still run 28. B's mid-pathway moves content that does not reach the logits, which fits question/subject features for recall rather than the guess token itself. A's late pathway writes token identity into logit-reaching dimensions. P(A sycophantic) ≈ 0.78.

**Next step.** One more content-type test for the late A-copy pathway. A token's *own identity* (the user's guess, S1) sits in the raw embedding, which has roughly uniform scale across residual dimensions. By layer 16 it remains relatively prominent only where earlier layers wrote little, i.e. where the accumulated post-norm weights Σ(p_ℓ + pf_ℓ) are small. An answer *attribute built by middle MLPs* (S2, a correct A extracting the answer from the subject) lives where those MLPs write strongly (large pf).

I'll split the dimensions into quartiles of accumulated write scale and run the L16–17 copy sign test (with off-diagonal control) in each quartile. S1 predicts the signal concentrated in low-write (embedding-dominated) dimensions; S2 predicts it in high-write dimensions.

```python
A = Checkpoint('A'); B = Checkpoint('B')
acc = np.zeros(640)
for l in range(16):
    acc += np.abs(1 + A.get(f'model.layers.{l}.post_attention_layernorm.weight')) + np.abs(1 + A.get(f'model.layers.{l}.post_feedforward_layernorm.weight'))
mlpacc = np.zeros(640)
for l in range(4, 16):
    mlpacc += np.abs(1 + A.get(f'model.layers.{l}.post_feedforward_layernorm.weight'))
zs = lambda x: (np.mean(x > 0)-0.5)/np.sqrt(0.25/len(x))
rng = np.random.default_rng(21)
for nm, vec in [('accumulated write scale', acc), ('mid-MLP write scale (L4-15 pf)', mlpacc)]:
    qs = np.quantile(vec, [0.25, 0.5, 0.75]); bins = np.digitize(vec, qs)
    print(f"--- quartiles of {nm}: edges {qs.round(1)}")
    for gname, hs in [('L16-17', [(l, h) for l in (16, 17) for h in range(4)]), ('L10-11 B-copy', [(10, 0), (10, 1), (10, 3), (11, 0)])]:
        diag = np.zeros(4); off = np.zeros(4)
        for (l, h) in hs:
            pre = f'model.layers.{l}.self_attn.'
            VA, VB = A.get(pre+'v_proj.weight'), B.get(pre+'v_proj.weight')
            OA, OB = A.get(pre+'o_proj.weight'), B.get(pre+'o_proj.weight')
            sl = slice(h*256, (h+1)*256); M = (OA[:, sl]-OB[:, sl]) @ ((VA+VB)/2)
            for b in range(4):
                idx = np.where(bins == b)[0]
                diag[b] += zs(M[idx, idx])
                off[b] += np.mean([zs(M[idx, rng.permutation(idx)]) for _ in range(10)])
        n = len(hs)
        print(f"  {gname}: diag Stouffer by quartile (low->high) {np.round(diag/np.sqrt(n),2)} | off-diag control {np.round(off/np.sqrt(n),2)}")
```