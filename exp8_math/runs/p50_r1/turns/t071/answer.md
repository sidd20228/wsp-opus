The pass returned 1,873 tokens for text A, matching `fwd_5` exactly. Text B came back with 1,699 tokens instead of 1,828, so I pasted a few fewer loop repetitions (roughly four line pairs). Because the loop is periodic, B should still be an exact prefix of the original, and comparisons up to 1,699 tokens remain valid. I'll verify that first.

Then the causal readout for zeroing layer-16 neuron 998:

- P(Wait) at the original's "Wait" positions, before and inside the loop, against `fwd_5`;
- what the model would say instead (top alternatives);
- the change in log-probability on all non-Wait tokens, which measures collateral damage.

The prediction is a large drop in P(Wait) before the loop, a smaller drop inside it (copying also supports "Wait" there), and only small changes elsewhere.

```python
# Causal test of L16 n998 ablation: P(Wait) at Wait positions (pre/loop) and side effects on non-Wait tokens, vs fwd_5 original.
ids9 = np.load('/work/fwd_9/input_ids.npy'); tlp9 = np.load('/work/fwd_9/token_logprobs.npy')
tk9_ids = np.load('/work/fwd_9/topk_ids.npy'); tk9_lp = np.load('/work/fwd_9/topk_logprobs.npy')
lens9 = [1873, 1699]
for i in range(2):
    m_ = lens9[i]; same = np.array_equal(ids9[i, :m_], ids5[i, :m_]); print('text', 'AB'[i], 'prefix identical to fwd_5:', same)
for i in range(2):
    m_ = lens9[i]; ids_ = ids5[i, :m_]; ls = loop_start_tok[i]
    wpos = np.array([t for t in range(m_ - 1) if ids_[t + 1] in WAIT]); pre_ = wpos[wpos < ls]; inl_ = wpos[wpos >= ls]
    nw = np.array([t for t in range(200, m_ - 1) if ids_[t + 1] not in WAIT]); nw_pre = nw[nw < ls]; nw_loop = nw[nw >= ls]
    print('text %s | P(Wait) pre: orig %.3f -> abl %.3f (n=%d) | loop: orig %.3f -> abl %.3f (n=%d)' % (
        'AB'[i], np.exp(tlp[i, pre_]).mean(), np.exp(tlp9[i, pre_]).mean(), len(pre_), np.exp(tlp[i, inl_]).mean(), np.exp(tlp9[i, inl_]).mean(), len(inl_)))
    print('        mean dlogp at Wait pos: pre %.2f loop %.2f | non-Wait tokens dlogp: pre %.3f loop %.3f' % (
        (tlp9[i, pre_] - tlp[i, pre_]).mean(), (tlp9[i, inl_] - tlp[i, inl_]).mean(), (tlp9[i, nw_pre] - tlp[i, nw_pre]).mean(), (tlp9[i, nw_loop] - tlp[i, nw_loop]).mean()))
    for t in list(pre_[:3]) + list(inl_[:2]):
        alts = ', '.join('%s:%.2f' % (safe_tok(tk9_ids[i, t, j]), np.exp(tk9_lp[i, t, j])) for j in range(5))
        print('        t=%d (%s) abl alts: %s' % (t, 'pre' if t < ls else 'loop', alts))
```