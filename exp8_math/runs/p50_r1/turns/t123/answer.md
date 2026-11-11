The sample fields are text, pred, correct, tokens and cut, and all 512 samples pair up across the three passes. Next I will reproduce the headline paired numbers and compute empirical hazards per 1k-token bin from these fields: finishing hazard from finished samples, loop-onset hazard from my earlier loop detector. I'll also list the helper functions already defined in the kernel so I can reuse the loop-onset detector instead of rewriting it. If ×2 raised the per-1k finishing hazard, a stronger brake may help further. If it mostly delayed loop onset, the remaining problem is the copying lock-in, and more braking won't help much.

```python
# Check: helper functions available; reproduce paired acc; per-bin hazards (finish / loop onset) for orig vs x2
fns = [k for k,v in globals().items() if callable(v) and not k.startswith('_') and getattr(v,'__module__',None)=='__main__']
print("my functions:", fns)
def acc(F): return np.mean([F[k]['correct'] for k in keys])*100
def cutp(F): return np.mean([F[k]['cut'] for k in keys])*100
for nm,F in [('orig',F2),('x2',F11),('x2+rb',F13)]:
    fin=[k for k in keys if not F[k]['cut']]
    print(f"{nm}: acc {acc(F):.1f} cut {cutp(F):.1f} fin {len(fin)} fin-acc {np.mean([F[k]['correct'] for k in fin])*100:.1f} fin-ntok med {np.median([F[k]['tokens'] for k in fin]):.0f}")
dd = np.array([F11[k]['correct']-F2[k]['correct'] for k in keys],float)
print("x2-orig paired: %.1f (SE %.1f)"%(dd.mean()*100, dd.std(ddof=1)/np.sqrt(len(dd))*100))
```