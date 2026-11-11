`inspect` can't recover the helper's source, so I'll write a transparent loop detector instead.

For each text I take the token IDs and test candidate periods p up to 4,096. For each p, I find the longest suffix where every token equals the token p positions earlier. The best p gives the verbatim-loop period, and the start of that run gives the onset. I'll call a sample "looped" if the run is at least 1,024 tokens long and spans at least two periods.

I'll run this on both `gen_2` (original) and `gen_7` (edited). The comparison will show whether the ablation reduced verbatim loops or only moved them, for example to later onsets or to different periods.

```python
# Own loop detector: longest verbatim periodic suffix (period<=4096). Compare original (gen_2) vs edited (gen_7) loop prevalence/onset/period.
def loop_info(ids, maxp=4096):
    a = np.asarray(ids); n = len(a); best = (0, 0)
    for p in range(1, min(maxp, n // 2)):
        eq = a[p:] == a[:-p]
        bad = np.flatnonzero(~eq)
        run = (len(eq) - (bad[-1] + 1)) if len(bad) else len(eq)
        if run > best[0]: best = (run, p)
    run, p = best
    return dict(run=run, period=p, onset=n - run - p if run > 0 else n, n=n, looped=(run >= 1024 and run >= 2 * p))
def summarize(G):
    out = []
    for g_ in G:
        for i, s_ in enumerate(g_['samples']):
            ids_ = tok.encode(s_['text']).ids
            li = loop_info(ids_)
            li.update(id=g_['id'], si=i, src=g_['source'], correct=s_['correct'], cut=s_['cut'])
            out.append(li)
    return pd.DataFrame(out)
L2 = summarize(G2); L7 = summarize(G7)
for nm_, L_ in [('orig', L2), ('edit', L7)]:
    c_ = L_[L_.cut]
    print(nm_, 'cut %d/%d, looped among cut %.1f%%, onset q10/50/90 %s, period q10/50/90 %s, run median %d' % (
        len(c_), len(L_), 100 * c_.looped.mean(), np.percentile(c_[c_.looped].onset, [10, 50, 90]).astype(int),
        np.percentile(c_[c_.looped].period, [10, 50, 90]).astype(int), np.median(c_[c_.looped].run)))
```