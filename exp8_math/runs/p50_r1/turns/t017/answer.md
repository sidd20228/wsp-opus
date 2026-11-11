The loops have no single stereotyped trigger. Only 41% of repeated units contain "Wait", indentation varies (median 4 spaces), and the unit is usually one or two lines. Some loops re-quote the problem statement, others repeat a calculation line or a reasoning sentence. So this looks like a general verbatim-copy attractor rather than a specific phrase habit. Before choosing where to look in the network, I want to know whether the hazard of entering a loop rises with context length (long-context degradation) or is roughly constant per token (a per-step copy instability). I also want to see how a loop begins: whether the model slides gradually into near-copies of an earlier line or snaps into an exact repeat. I'll compute the loop-onset hazard by position bin, treating finishing as competing censoring, and print what precedes two onsets.

```python
# Hypothesis: loop-entry hazard grows with context length (long-context degradation) vs constant per-token hazard.
ev = []
for r in df.itertuples():
    if r.cut and r.onset is not None and not np.isnan(r.onset): ev.append((int(r.onset), 1))
    else: ev.append((int(r.ntok), 0))  # finished (or no onset) -> censored at its length
ev = np.array(ev)
bins = [0, 1000, 2000, 3000, 4000, 6000, 8000, 12000, 16000, 32768]
for a, b in zip(bins[:-1], bins[1:]):
    at_risk = (ev[:,0] >= a).sum()
    events = ((ev[:,0] >= a) & (ev[:,0] < b) & (ev[:,1] == 1)).sum()
    cens = ((ev[:,0] >= a) & (ev[:,0] < b) & (ev[:,1] == 0)).sum()
    print(f'{a:>6}-{b:<6} at_risk={at_risk:4d} loops={events:4d} finished={cens:3d} hazard/1k tok={events/max(at_risk,1)/((b-a)/1000):.3f}')
for k in [0, 24]:
    r = df[df.cut & df.onset.notna()].iloc[k]
    ids = IDS[(r.id, r.si)]; on = int(r.onset)
    print(f'--- {r.id} onset={on} period={r.period}')
    print('BEFORE:', asc(repr(tok.decode(ids[on-150:on]))[:400]))
    print('AFTER :', asc(repr(tok.decode(ids[on:on+90]))[:300]))
```