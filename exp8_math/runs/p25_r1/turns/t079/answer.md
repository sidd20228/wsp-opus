A large effect. With only the brake neuron L22 n573 strengthened (output column × 3), the edited model scores 43.6% over all 512 problems (MATH 36.7%, GSM8K 50.4%), against 7.8% for the original on setA. The cut share falls from 93% to 53%, and only 25 answers that finished were wrong. setB has no original baseline yet, so the rigorous number is the paired comparison on setA against pass 4's original outputs.

I will compute that paired difference with its standard error and the flip counts. I will also check that the seeds really are shared, which shows as long common text prefixes with the original, and measure what changed in the text: "Wait" density, other doubt openers, loop share and onset. If accuracy on setA rises by far more than its standard error and "Wait"-led doubt lines are much rarer, the hypothesis is confirmed.

```python
# Paired analysis of E2 (n573 x3) vs original on setA; seed-sharing check; text-level mechanism checks.
g7 = readj('/work/gen_7.jsonl'); d7 = pd.DataFrame(stats(g7)); setA_ids = set(sp['setA'])
mA = do.merge(d7[d7.id.isin(setA_ids)], on=['id', 'src'], suffixes=('_o', '_e'))
diffA = mA.correct_e.astype(float) - mA.correct_o.astype(float)
print('setA n', len(mA), '| orig acc %.3f cut %.3f | E2 acc %.3f cut %.3f' % (mA.correct_o.mean(), mA.cut_o.mean(), mA.correct_e.mean(), mA.cut_e.mean()))
print('paired diff %.3f +- %.3f | w->r %d r->w %d' % (diffA.mean(), diffA.std(ddof=1) / np.sqrt(len(diffA)), int(((~mA.correct_o) & mA.correct_e).sum()), int((mA.correct_o & ~mA.correct_e).sum())))
print('by source:', mA.groupby('src').apply(lambda d: (round(d.correct_o.mean(), 3), round(d.correct_e.mean(), 3)), include_groups=False).to_dict())
print('setB E2 acc %.3f cut %.3f' % (d7[~d7.id.isin(setA_ids)].correct.mean(), d7[~d7.id.isin(setA_ids)].cut.mean()))
g7_by = {r['id']: r for r in g7}; pref = []
for r in g5o[:60]:
    a_ = ids_of(r['samples'][0]['text'][:6000]); b_ = ids_of(g7_by[r['id']]['samples'][0]['text'][:6000]); m_ = min(len(a_), len(b_))
    neq = np.where(a_[:m_] != b_[:m_])[0]; pref.append(neq[0] if len(neq) else m_)
print('common prefix tokens (first 60 setA): median', np.median(pref), 'q25', np.quantile(pref, .25))
fin7 = d7[~d7.cut]; print('E2 finished: n %d acc %.3f median tokens %d' % (len(fin7), fin7.correct.mean(), fin7.tokens.median()))
```