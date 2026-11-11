The token lengths (1,334 and 1,184) match the prefixes, so alignment with `fwd_5` should hold; I'll confirm it. Then I'll read out the doubled brake's effect:

- P(Wait) at the original "Wait" positions, before and inside the loop;
- the mean log-probability change on all other tokens, as collateral damage;
- what the model would say instead of "Wait".

These numbers will be compared with neuron 998's ablation, which removed only about 0.5 nats before the loop and nothing inside it.

```python
# Effect of L22 n573 x2 (Wait brake gain): P(Wait) at Wait positions (pre/loop) and side effects, vs fwd_5 original.
ids10 = np.load('/work/fwd_10/input_ids.npy'); tlp10 = np.load('/work/fwd_10/token_logprobs.npy')
tk10_ids = np.load('/work/fwd_10/topk_ids.npy'); tk10_lp = np.load('/work/fwd_10/topk_logprobs.npy')
lens10 = [1334, 1184]
def wait_effect(tlpE, tkE_ids, tkE_lp, lensE, label):
    for i in range(2):
        m_ = lensE[i]; ids_ = ids5[i, :m_]; ls = loop_start_tok[i]
        assert np.array_equal(ids10[i, :m_ - 1] if label == 'x2' else ids_[:m_ - 1], ids_[:m_ - 1])
        wpos = np.array([t for t in range(m_ - 2) if ids_[t + 1] in WAIT]); pre_ = wpos[wpos < ls]; inl_ = wpos[wpos >= ls]
        nw = np.array([t for t in range(200, m_ - 2) if ids_[t + 1] not in WAIT]); nwp, nwl = nw[nw < ls], nw[nw >= ls]
        print('%s text %s | P(Wait) pre %.3f->%.3f (n=%d) | loop %.3f->%.3f (n=%d) | dlogp non-Wait pre %+.3f loop %+.3f' % (
            label, 'AB'[i], np.exp(tlp[i, pre_]).mean(), np.exp(tlpE[i, pre_]).mean(), len(pre_), np.exp(tlp[i, inl_]).mean(), np.exp(tlpE[i, inl_]).mean(), len(inl_),
            (tlpE[i, nwp] - tlp[i, nwp]).mean(), (tlpE[i, nwl] - tlp[i, nwl]).mean()))
        for t in list(pre_[:2]) + list(inl_[:2]):
            print('      t=%d %s alts: %s' % (t, 'pre ' if t < ls else 'loop', ', '.join('%s:%.2f' % (safe_tok(tkE_ids[i, t, j]), np.exp(tkE_lp[i, t, j])) for j in range(5))))
wait_effect(tlp10, tk10_ids, tk10_lp, lens10, 'x2')
```