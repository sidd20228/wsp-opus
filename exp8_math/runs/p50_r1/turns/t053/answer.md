The previous run failed on a NaN onset (some cut traces had no loop under the old detector). I'll use the onset and period from my own detector (`L2`) for the loop-location analysis instead. It defines both values for every trace, and I'll keep only traces flagged as looped. Everything else is unchanged.

```python
# Same analysis using my own loop detector's onset/period (L2), restricted to looped cut traces.
cutD = D3[D3.cut].merge(L2[['id', 'si', 'onset', 'period', 'looped']].rename(columns={'onset': 'on2'}), on=['id', 'si'])
cutD = cutD[cutD.looped].copy()
reach = []; units = []
for _, r_ in cutD.iterrows():
    offs = tok.encode(r_.txt).offsets
    on = min(int(r_.on2), len(offs) - 1); per_ = int(r_.period)
    reach.append(bool(final_re.search(r_.txt[:offs[on][0]])))
    units.append(r_.txt[offs[min(on + 2 * per_, len(offs) - 1)][0]: offs[min(on + 3 * per_, len(offs) - 1)][0]])
cutD['reach_final'] = reach; cutD['unit'] = units
print('looped cut traces: %d; reached final-output stage before onset: %.1f%% (gsm %.1f%%, math %.1f%%)' % (
    len(cutD), 100 * cutD.reach_final.mean(), 100 * cutD[cutD.src == 'gsm8k'].reach_final.mean(), 100 * cutD[cutD.src == 'math'].reach_final.mean()))
print('onset median reached=%d not=%d' % (cutD[cutD.reach_final].on2.median(), cutD[~cutD.reach_final].on2.median()))
for k_, p_ in kw.items(): print('unit contains %-12s %.1f%%' % (k_, 100 * np.mean([bool(re.search(p_, u)) for u in cutD.unit])))
for u in cutD.sample(8, random_state=7).unit: print(' ~', asc(u)[:150])
```