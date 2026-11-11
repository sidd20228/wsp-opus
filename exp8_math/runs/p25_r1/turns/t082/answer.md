The records nest the per-sample results inside a `samples` list, so my key guesses missed. This is only a parsing fix and does not change the question. I will look at the keys of one sample, then compute the same paired comparison (E2 vs original on setA) and the absolute numbers on setB.

```python
# Parsing fix: per-sample fields live in r['samples']; recompute pass-6 paired accuracy E2 vs orig on setA, and setB stats.
s0 = g7[0]['samples'][0]; print(sorted(s0.keys())); print({k: str(v)[:60] for k,v in s0.items() if k!='text'})
def S(r): return r['samples'][0]
def k_of(d, c):
    for x in c:
        if x in d: return x
COR=k_of(s0,['correct','is_correct','grade','score','graded']); LEN=k_of(s0,['n_tokens','num_tokens','length','tokens','len','n_gen_tokens'])
CUT=k_of(s0,['cut','truncated','hit_max','finish_reason','cut_off'])
print('keys used', COR, LEN, CUT)
def tocor(r): v=S(r)[COR]; return float(v) if not isinstance(v,str) else float(v.lower() in ('correct','true','1'))
def cutf(r): v=S(r)[CUT]; return float(v=='length') if isinstance(v,str) else float(v)
common = [i for i in A7 if i in o]
d = np.array([tocor(A7[i]) - tocor(o[i]) for i in common])
print(f"setA n={len(common)}: orig {np.mean([tocor(o[i]) for i in common])*100:.1f}%  E1 {np.mean([tocor(e1[i]) for i in common if i in e1])*100:.1f}%  E2 {np.mean([tocor(A7[i]) for i in common])*100:.1f}%  paired E2-orig {d.mean()*100:+.1f} +- {d.std(ddof=1)/np.sqrt(len(d))*100:.1f}  w->r {(d>0).sum()} r->w {(d<0).sum()}")
for name, rows in [('orig A', [o[i] for i in common]), ('E1 A', [e1[i] for i in common]), ('E2 A', [A7[i] for i in common]), ('E2 B', B7)]:
    src = collections.defaultdict(list)
    for r in rows: src[r['source']].append(tocor(r))
    print(name, f"acc {np.mean([tocor(r) for r in rows])*100:.1f}%  cut {np.mean([cutf(r) for r in rows])*100:.1f}%  meanlen {np.mean([S(r)[LEN] for r in rows]):.0f}", {s: f"{np.mean(v)*100:.1f}({len(v)})" for s,v in src.items()})
```