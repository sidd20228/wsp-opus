The repetition-blinding edit moved things in the predicted direction, but only a little. Accuracy went from 43.0% (brake only, same samples and seeds) to 44.7%. The cut share fell from 55.9% to 51.4%, and finished answers were 25 wrong out of roughly 250. A 1.7-point gain is within the noise of one 512-sample run, so I need the paired statistics before concluding anything. I will compute the paired accuracy difference against gen_11 and its standard error, the wrong→right and right→wrong counts, loop share and onset among cut samples, and finished-answer accuracy. A gain under about two standard errors would mean that blinding layer 23 to the repetition signal is at best a weak lever. That would suggest the absorbing copy amplification is mostly content-driven (many matching keys) rather than signalled.

```python
# Hypothesis: L23 r-blind adds a small, possibly non-significant gain over brake x2; check paired stats + loop stats vs gen_11.
g13r = load_gen('/work/gen_13.jsonl'); assert [r['id'] for r in g13r] == ids_order
def flat(G_): return np.array([bool(s['correct']) for r in G_ for s in r['samples']]), np.array([bool(s['cut']) for r in G_ for s in r['samples']])
c2a, k2a = flat(g2r); c11a, k11a = flat(g11r); c13a, k13a = flat(g13r)
def pdiff(a, b):
    d = a.astype(float) - b.astype(float); 
    dp = d.reshape(-1, 2).mean(1)   # cluster by problem
    return 100*d.mean(), 100*dp.std(ddof=1)/np.sqrt(len(dp)), int(((a) & (~b)).sum()), int(((~a) & b).sum())
print('x2+blind vs x2: diff %.1f (SE %.1f) w->r %d r->w %d' % pdiff(c13a, c11a))
print('x2+blind vs orig: diff %.1f (SE %.1f) w->r %d r->w %d' % pdiff(c13a, c2a))
L13 = pd.DataFrame([dict(id=r['id'], si=si, src=r['source'], correct=s['correct'], cut=s['cut'], **{k: v for k, v in loop_info(tok.encode(s['text']).ids).items() if k in ('period', 'onset', 'looped', 'run')}) for r in g13r for si, s in enumerate(r['samples'])])
for nm, Ld in [('x2', L11), ('x2+blind', L13)]:
    cutd = Ld[Ld.cut]; fin = Ld[~Ld.cut]
    print(f"{nm}: cut {len(cutd)} looped-among-cut {cutd.looped.mean():.3f} onset q10/50/90 {np.percentile(cutd[cutd.looped].onset, [10,50,90]).astype(int)} | finished {len(fin)} acc {fin.correct.mean():.3f}")
```