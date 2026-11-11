The kernel survived with all its variables, and /work/edited holds exactly the brake edit: column 573 of L22 down_proj, ratio 2.0, nothing else. Each generation file has 256 items with a 'samples' list. I will now build a flat per-sample table (item id, sample index, correct, length, cut, finished) for gen_2 (original), gen_11 (brake ×2) and gen_13 (×2 + r-blind). It re-derives the paired numbers from my notes as a consistency check. It also gives finish and loop-onset hazard curves by token position for the original and ×2. These curves show how ×2 helped: by letting runs finish before a loop begins, or by making loops start later. That tells me whether a stronger brake is likely to help further.

```python
# Check: rebuild per-sample tables; reproduce paired numbers (orig 10.9, x2 43.0); hazards of finishing vs loop onset by position
s0 = g11[0]['samples'][0]; print("sample keys:", list(s0.keys()))
def flat(G):
    out={}
    for r in G:
        for j,s in enumerate(r['samples']):
            out[(r['id'],j)] = s
    return out
F2,F11,F13 = flat(g2),flat(g11),flat(g13)
keys = sorted(set(F2)&set(F11)&set(F13)); print("paired keys:", len(keys))
def num(s,k,default=None):
    return s.get(k,default)
# figure out field names for correctness / ntok / cut
print({k:(type(v).__name__, (v if not isinstance(v,str) else v[:40])) for k,v in s0.items() if k!='text'})
```