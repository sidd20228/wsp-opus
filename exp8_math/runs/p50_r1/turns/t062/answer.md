A single neuron stands out. **Layer 16, neuron 998** supplies +3.1 nats toward "Wait" before the loop and +2.9 inside it. That is more than the whole layer-16 MLP's net +2.7, and more than the entire net pre-loop margin of +1.7. Its output vector is almost purely about "Wait": projected onto the vocabulary, " Wait" and "Wait" sit at z ≈ −14.5 and −15. " Wait" ranks 248,316 of 248,320, so only a handful of tokens (presumably other Wait variants) score lower. Since its contribution is positive, the neuron must fire negatively at these positions, and a negative activation times this vector writes "Wait" very strongly.

Two other neurons hint at the same doubt/commit axis:
- **L21 n1406** strongly suppresses commit tokens: " So", "So", " Therefore" and "Therefore" at z of −7 to −9, " Thus" at −10.5.
- **L23 n346** is a weaker Wait-specific writer (z ≈ −6.5).

Before testing n998 causally, I'll characterise it. First, are its gate, up and down weight norms outliers in layer 16? That would hint at an anomaly rather than a normal feature. Second, how is its activation distributed over the teacher-forced texts, and how negative is it at "Wait" positions compared with other line starts? Third, what else is at the extremes of its vocabulary projection (shown as token IDs or whitelisted words)?

If it fires strongly negative mostly at "decide what to do next" points and its output is Wait-specific, it's a clean, testable target for the doubt drive.

```python
# Characterize L16 neuron 998 ('Wait writer'): weight-norm outlier? activation distribution (all tokens vs Wait positions)? vocab extremes.
l_ = 16; n_ = 998
Wg = load(f'model.language_model.layers.{l_}.mlp.gate_proj.weight').numpy(); Wu = load(f'model.language_model.layers.{l_}.mlp.up_proj.weight').numpy()
Wd16 = load(f'model.language_model.layers.{l_}.mlp.down_proj.weight').numpy()
for nm_, v_ in [('gate', np.linalg.norm(Wg, axis=1)), ('up', np.linalg.norm(Wu, axis=1)), ('down', np.linalg.norm(Wd16, axis=0))]:
    print('%s norm n998 %.3f | layer median %.3f max %.3f | z %.1f | rank %d/3584' % (nm_, v_[n_], np.median(v_), v_.max(), (v_[n_] - v_.mean()) / v_.std(), int((v_ > v_[n_]).sum())))
lgv = Eg @ Wd16[:, n_]; zz = (lgv - lgv.mean()) / lgv.std()
lo = np.argsort(zz)[:12]; hi = np.argsort(-zz)[:8]
print('most suppressed by +act (=promoted by -act):', ', '.join('%s:%.1f' % (safe_tok(j), zz[j]) for j in lo))
print('most promoted by +act:', ', '.join('%s:%.1f' % (safe_tok(j), zz[j]) for j in hi))
A16 = np.asarray(np.load('/work/fwd_5/mlp_act_16.npy', mmap_mode='r')[:, :, n_], dtype=np.float32)
allv = np.concatenate([A16[i, :lens5[i]] for i in range(2)])
wv = np.array([A16[i, t] for i, t, _ in POS])
print('act all tokens: q01 %.3f q10 %.3f median %.3f q90 %.3f q99 %.3f | frac < -0.5*|q01|: %.3f' % (*np.percentile(allv, [1, 10, 50, 90, 99]), (allv < 0.5 * np.percentile(allv, 1)).mean()))
print('act at Wait positions: median %.3f, q10 %.3f, q90 %.3f' % tuple(np.percentile(wv, [50, 10, 90])))
# where is it most negative? next-token identity at the 40 most negative positions
neg = np.argsort(allv)[:40]; flat_next = np.concatenate([ids5[i, 1:lens5[i]].tolist() + [0] for i in range(2)])
from collections import Counter
print('next tokens at 40 most-negative positions:', Counter(safe_tok(j) for j in flat_next[neg]).most_common(8))
```