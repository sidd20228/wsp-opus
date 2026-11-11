E2 (scaling the "brake" neuron L22 n573 by 3) changed the thinking style a lot. "Wait" at line starts dropped from 178 per 1,000 lines to 0, and the share of exactly periodic loops fell from 0.92 to 0.44. A lower loop count does not by itself mean more correct answers, so I now need the accuracy readout of pass 6 (gen_7). On setA I will pair it with the original model's pass-4 answers on the same problems and seeds. SetB has no original run yet, so it only gives an absolute accuracy. If E2's paired gain on setA is clearly larger than its standard error, the brake mechanism is worth a dose-response study. If accuracy did not move, removing "Wait" only changed the shape of the loops, and I will look at what the remaining cut-off texts do.

```python
# Pass-6 readout: does E2 (L22 n573 x3) raise accuracy? paired vs gen_5_original on setA; setB unpaired.
import json, numpy as np, collections
def readj(p): return [json.loads(l) for l in open(p)]
g7 = readj('/work/gen_7.jsonl'); g5o = readj('/work/gen_5_original.jsonl'); g5e = readj('/work/gen_5_edited.jsonl')
print(len(g7), len(g5o), len(g5e)); print(sorted(g7[0].keys()))
print({k: str(v)[:50] for k,v in g7[0].items() if not isinstance(v,str) or len(v)<80})
sp = json.load(open('/work/splits.json')); print({k: (len(v) if hasattr(v,'__len__') else v) for k,v in sp.items()})
def k_of(d, c):
    for x in c:
        if x in d: return x
ID=k_of(g7[0],['id','problem_id','pid']); COR=k_of(g7[0],['correct','is_correct','grade','score'])
LEN=k_of(g7[0],['n_tokens','num_tokens','length','tokens','len','n_gen_tokens']); CUT=k_of(g7[0],['cut','truncated','hit_max','finish_reason'])
print('keys used', ID, COR, LEN, CUT)
def tocor(r): v=r[COR]; return float(v) if not isinstance(v,str) else float(v.lower() in ('correct','true','1'))
idsA = set(r[ID] for r in g5o)
A7 = {r[ID]: r for r in g7 if r[ID] in idsA}; B7 = [r for r in g7 if r[ID] not in idsA]
print('setA in g7:', len(A7), ' setB:', len(B7))
o = {r[ID]: r for r in g5o}; e1 = {r[ID]: r for r in g5e}
common = [i for i in A7 if i in o]
d = np.array([tocor(A7[i]) - tocor(o[i]) for i in common])
print(f"setA: orig {np.mean([tocor(o[i]) for i in common])*100:.1f}%  E1 {np.mean([tocor(e1[i]) for i in common if i in e1])*100:.1f}%  E2 {np.mean([tocor(A7[i]) for i in common])*100:.1f}%  paired E2-orig {d.mean()*100:+.1f} +- {d.std(ddof=1)/np.sqrt(len(d))*100:.1f}  w->r {(d>0).sum()} r->w {(d<0).sum()}")
def cutf(r):
    v = r[CUT]; return float(v=='length') if isinstance(v,str) else float(v)
for name, rows in [('orig A', [o[i] for i in common]), ('E2 A', [A7[i] for i in common]), ('E2 B', B7)]:
    src = collections.defaultdict(list)
    for r in rows: src[r.get('source','?')].append(tocor(r))
    print(name, f"acc {np.mean([tocor(r) for r in rows])*100:.1f}%  cut {np.mean([cutf(r) for r in rows])*100:.1f}%  meanlen {np.mean([r[LEN] for r in rows]):.0f}", {s: f"{np.mean(v)*100:.1f}({len(v)})" for s,v in src.items()})
```