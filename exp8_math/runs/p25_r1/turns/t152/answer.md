This screen is uninformative for late layers. The proven loop copiers (L15H7, L23H5) score about 0 on the raw-embedding self-copy test, so it says nothing about L23H6, and I won't use it to pick heads. Before spending a pass on extending the gate, I'll check where the remaining GSM8K losses come from, since GSM8K is the cleanest window on reasoning quality. If wrong GSM answers are mostly long drifting runs that do not occur without the gate, then more gating would add drift. In that case the gain lies in stopping drift rather than more loop suppression. I'll tabulate GSM8K outcomes and median lengths per configuration (x3, x5, x5+E1, x5+gate7, x5+gate6).

```python
# GSM8K outcome mix and lengths across configs: are gate-era GSM errors long drifting runs?
for nm,T in [('x3 gen7',T7),('x5 gen8',T8),('x5+E1 gen9',T9),('gate7 gen12',T12),('gate6 gen13',T13)]:
    v=[x for x in T.values() if x['src']=='gsm8k']; c=collections.Counter(x['st'] for x in v)
    med=lambda st: int(np.median([x['t'] for x in v if x['st']==st])) if c[st] else None
    print(f"{nm:12s} ok {c['ok']:3d} (med {med('ok')}) wrong {c['wrong']:3d} (med {med('wrong')}) cut {c['cut']:3d} noans {c['noans']}")
v=[x for x in T12.values() if x['st']!='cut']
for lo,hi in [(0,4000),(4000,8000),(8000,16000),(16000,33000)]:
    for src in ['gsm8k','math']:
        w=[x['c'] for x in v if lo<=x['t']<hi and x['src']==src]; print(f"  gate7 finished {src} {lo//1000}-{hi//1000}k: n={len(w)} acc={100*np.mean(w) if w else float('nan'):.0f}", end='')
    print()
```