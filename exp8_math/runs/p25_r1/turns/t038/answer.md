E1 barely moved anything: cut-offs fell from 93.0% to 90.6% and accuracy rose from 7.8% to 9.4% (paired +1.6 ± 2.0 points, not significant). So removing the three strongest induction heads does not stop the loops in real generation, even though it made junction tokens uncertain under teacher forcing. Before deciding whether to remove more copy capacity or look elsewhere, I need to know what the edited model's failures look like. Are the tails still exactly periodic? Do loops start later or shorter? I'll compute the same loop statistics (exact period, loop start, repetition share) for both models in this pass.

```python
# Hypothesis: with E1, loops still form (exact periodic tails), maybe later or with different periods. Compare stats.
def load_gen(path):
    G=[json.loads(l) for l in open(path)]; rows=[]
    for g in G:
        for j,s in enumerate(g['samples']):
            t=s['text']; ids=ids_of(t)
            cut = str(s['cut'])=='True'
            p = tail_period(ids) if cut else None
            ls_ = exact_start(ids,p) if p else None
            rows.append(dict(id=g['id'], src=g['source'], j=j, correct=str(s['correct'])=='True', cut=cut, ntok=int(s['tokens']),
                             period=p, lstart=ls_, has_end=('</think>' in t), text=t))
    return pd.DataFrame(rows)
G5o=load_gen('/work/gen_5_original.jsonl'); G5e=load_gen('/work/gen_5_edited.jsonl')
for nm,G in [('orig',G5o),('E1',G5e)]:
    c=G[G.cut]
    print(nm, 'cut',len(c),'exact periodic',c.period.notna().mean().round(3),
          'period q',c.period.dropna().quantile([.25,.5,.75]).tolist(),'loop start q',c.lstart.dropna().quantile([.25,.5,.75]).round(0).tolist(),
          'finished len med', G[~G.cut].ntok.median())
```