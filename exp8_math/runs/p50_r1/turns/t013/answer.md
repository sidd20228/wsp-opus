The cut-off transcript starts with sensible structured reasoning, then **collapses into a verbatim loop**: the same bullet line ("* Wait, if she passes him, she is at a point closer to the school.") repeats until the 32k limit, despite the 1.5 presence penalty. So the dominant failure is not overthinking but a degenerate repetition attractor. Before going after a mechanism, I'll quantify this across all 512 samples. For each sample I'll find the loop onset (the token position after which nearly every 32-gram is a repeat of earlier text), the loop period, and how often the looping unit starts with "Wait" or deep indentation. If most cut samples enter an exact loop at a measurable point, the target is the copying/repetition circuitry. If instead they wander without exact repeats, the target would be the stopping decision.

```python
# Hypothesis: most cut generations enter an exact periodic loop; measure onset position, period, loop content.
import numpy as np
def loop_stats(ids, n=32):
    # returns onset index after which >=95% of n-grams are repeats (from end backwards), and dominant period
    L = len(ids)
    seen = {}
    rep = np.zeros(L, dtype=bool)
    last = np.full(L, -1)
    for i in range(L - n + 1):
        key = tuple(ids[i:i+n])
        if key in seen:
            rep[i] = True; last[i] = seen[key]
        seen[key] = i
    # onset: smallest i such that rep[i:L-n] mostly True (scan from end)
    if L < 2*n: return None, None, rep[:max(L-n+1,0)].mean() if L>n else 0
    tail = rep[:L-n+1]
    # cumulative from end fraction of repeats
    rev = tail[::-1].astype(float)
    cum = np.cumsum(rev) / np.arange(1, len(rev)+1)
    ok = np.where(cum >= 0.95)[0]
    onset = (len(tail) - 1 - ok.max()) if len(ok) else None
    per = None
    if onset is not None and onset < L - n - 1:
        diffs = (np.arange(L - n + 1) - last)[onset:L-n+1]
        diffs = diffs[last[onset:L-n+1] >= 0]
        if len(diffs): per = int(np.bincount(diffs).argmax())
    return onset, per, tail.mean()
rows = []
for g in G:
    for si, s in enumerate(g['samples']):
        ids = tok.encode(s['text']).ids
        on, per, frac = loop_stats(ids)
        rows.append(dict(id=g['id'], src=g['source'], si=si, correct=s['correct'] in (True,'True'), cut=s['cut'] in (True,'True'),
                         ntok=len(ids), onset=on, period=per, repfrac=frac))
import pandas as pd
df = pd.DataFrame(rows)
print(df.groupby(['cut','correct']).size())
c = df[df.cut]
print('cut with onset found:', c.onset.notna().mean(), ' onset quantiles:', c.onset.quantile([.1,.25,.5,.75,.9]).round().tolist())
print('period quantiles:', c.period.quantile([.1,.25,.5,.75,.9]).tolist())
print('finished length quantiles:', df[~df.cut].ntok.quantile([.1,.5,.9]).tolist())
print('per-problem: both cut', (df.groupby('id').cut.sum()==2).mean(), ' one cut', (df.groupby('id').cut.sum()==1).mean())
```