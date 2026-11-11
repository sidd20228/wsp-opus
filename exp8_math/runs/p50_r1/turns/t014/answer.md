A shape bug in my loop detector: the `last` array has length L, but it needs to be sliced to the L−n+1 n-gram positions. Same hypothesis and analysis, just fixed. I'm checking whether most cut generations enter an exact periodic loop, and at what token position.

```python
# Hypothesis: most cut generations enter an exact periodic loop; measure onset position, period (bug fixed).
def loop_stats(ids, n=32):
    L = len(ids)
    if L < 2*n: return None, None, 0.0
    seen = {}; M = L - n + 1
    rep = np.zeros(M, dtype=bool); last = np.full(M, -1)
    for i in range(M):
        key = tuple(ids[i:i+n])
        if key in seen: rep[i] = True; last[i] = seen[key]
        seen[key] = i
    rev = rep[::-1].astype(float)
    cum = np.cumsum(rev) / np.arange(1, M+1)
    ok = np.where(cum >= 0.95)[0]
    onset = (M - 1 - ok.max()) if len(ok) else None
    per = None
    if onset is not None and onset < M - 1:
        idx = np.arange(onset, M); lv = last[onset:]
        d = (idx - lv)[lv >= 0]
        if len(d): per = int(np.bincount(d).argmax())
    return onset, per, rep.mean()
rows = []; IDS = {}
for g in G:
    for si, s in enumerate(g['samples']):
        ids = tok.encode(s['text']).ids
        IDS[(g['id'], si)] = ids
        on, per, frac = loop_stats(ids)
        rows.append(dict(id=g['id'], src=g['source'], si=si, correct=s['correct'] in (True,'True'), cut=s['cut'] in (True,'True'),
                         ntok=len(ids), onset=on, period=per, repfrac=frac))
df = pd.DataFrame(rows)
print(df.groupby(['cut','correct']).size())
c = df[df.cut]
print('cut with onset found:', c.onset.notna().mean(), ' onset quantiles:', c.onset.quantile([.1,.25,.5,.75,.9]).round().tolist())
print('period quantiles:', c.period.quantile([.1,.25,.5,.75,.9]).tolist())
print('finished length quantiles:', df[~df.cut].ntok.quantile([.1,.5,.9]).tolist())
pp = df.groupby('id').cut.sum()
print('per-problem: both cut', (pp==2).mean(), ' one cut', (pp==1).mean(), ' none', (pp==0).mean())
```