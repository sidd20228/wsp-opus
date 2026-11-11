The record layout is `{'id', 'source', 'answer', 'samples': [{'text', ...}]}`. My print dumped a full transcript, and that sample is itself typical: the model keeps re-verifying an easy LCM problem ("Wait, let me re-read...") for thousands of tokens. Next I will print only the non-text fields of a sample, then compute the paired comparison of pass 4: accuracy, cut-off share, mean length, and wrong-to-right/right-to-wrong flips. This tells me whether the induction-head ablation E1 helped. If accuracy rose clearly beyond its paired standard error, I would build on E1. If it is flat, I will look for a different mechanism, for example the drift into "Wait, re-check" self-verification that precedes the loops.

```python
# Compute pass-4 paired stats (E1 vs original on setA) without printing texts.
s0 = g5o[0]['samples'][0]
print({k: v for k, v in s0.items() if k != 'text'})
def stats(recs):
    rows = []
    for r in recs:
        for s in r['samples']:
            rows.append(dict(id=r['id'], src=r['source'], **{k: v for k, v in s.items() if k != 'text'}, L=len(s['text'])))
    return rows
import pandas as pd
do = pd.DataFrame(stats(g5o)); de = pd.DataFrame(stats(g5e))
print(do.columns.tolist())
```