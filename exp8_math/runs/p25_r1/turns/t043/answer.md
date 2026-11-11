Each sample carries `pred`, `correct`, `tokens` and `cut`, which is enough to rebuild the pass-4 summary. I will compute accuracy, cut share and mean tokens per model and per source, plus the paired difference with its standard error and the flip counts. This settles whether E1 (ablating the three top induction heads) helps. With 256 problems the paired SE should be roughly 2-3 points, so I will only believe a gain of about 5 points or more.

```python
# Paired comparison E1 vs original on setA (pass 4).
m = do.merge(de, on=['id','src'], suffixes=('_o','_e'))
for nm, d in [('orig', do), ('E1', de)]:
    print(nm, 'acc %.3f cut %.3f meantok %.0f' % (d.correct.mean(), d.cut.mean(), d.tokens.mean()),
          d.groupby('src').correct.mean().round(3).to_dict(), d.groupby('src').cut.mean().round(3).to_dict())
diff = m.correct_e.astype(float) - m.correct_o.astype(float)
print('paired diff %.3f +- %.3f' % (diff.mean(), diff.std(ddof=1)/np.sqrt(len(diff))))
print('wrong->right', int(((~m.correct_o) & m.correct_e).sum()), 'right->wrong', int((m.correct_o & ~m.correct_e).sum()))
print('finished & correct rate: orig %.3f, E1 %.3f' % (do[~do.cut].correct.mean(), de[~de.cut].correct.mean()))
print('cut both', int((m.cut_o & m.cut_e).sum()), 'cut only orig', int((m.cut_o & ~m.cut_e).sum()), 'cut only E1', int((~m.cut_o & m.cut_e).sum()))
```