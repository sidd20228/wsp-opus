Two problems with this attempt. First, `tok.decode` silently dropped the special tokens (`<|im_start|>`, `<think>`, …), so these texts don't match the eval format. Second, and more important, the window for geometry_151 contains no verbatim loop, even though my detector placed the onset at token 615. The cause is a flaw in the detector: requiring ≥95% repeats from the onset to the end lets a ~30k-token loop tail absorb up to ~1,500 earlier non-repeated tokens, so onsets were biased early. I'll redefine the onset as the start of the final contiguous run of repeated 32-grams (merging gaps under 32 tokens) and redo the onset/hazard numbers. Then I'll rebuild texts from the raw strings rather than decoded tokens. If the corrected onsets shift later, the per-1k-token hazard estimates will change, but the conclusion that ~90% of samples end in absorbing loops will not.

```python
# Hypothesis: onset estimator was biased early; recompute onset as start of final contiguous repeat run.
def loop_stats2(ids, n=32, gap=32):
    L = len(ids); M = L - n + 1
    if M < 2: return None, None, 0.0
    seen = {}; rep = np.zeros(M, dtype=bool); last = np.full(M, -1)
    for i in range(M):
        key = tuple(ids[i:i+n])
        if key in seen: rep[i] = True; last[i] = seen[key]
        seen[key] = i
    if not rep[-1]: return None, None, rep.mean()
    # walk back from end while gaps of non-repeats are < gap
    i = M - 1; nonrep = 0
    while i > 0:
        if rep[i-1]: nonrep = 0
        else:
            nonrep += 1
            if nonrep >= gap: break
        i -= 1
    onset = i + nonrep if nonrep >= gap else i
    while onset < M and not rep[onset]: onset += 1
    if M - onset < 4*n: return None, None, rep.mean()   # require a substantial tail
    d = (np.arange(onset, M) - last[onset:]); d = d[last[onset:] >= 0]
    per = int(np.bincount(d).argmax()) if len(d) else None
    return onset, per, rep.mean()
rows2 = []
for (pid, si), ids in IDS.items():
    on, per, frac = loop_stats2(ids)
    r = df[(df.id==pid)&(df.si==si)].iloc[0]
    rows2.append(dict(id=pid, si=si, src=r.src, cut=r.cut, correct=r.correct, ntok=len(ids), onset=on, period=per))
df2 = pd.DataFrame(rows2)
c2 = df2[df2.cut]
print('cut with loop tail:', c2.onset.notna().mean(), ' onset q:', c2.onset.quantile([.1,.25,.5,.75,.9]).round().tolist())
print('period q:', c2.period.quantile([.1,.25,.5,.75,.9]).tolist())
print('loop tail tokens median:', (c2.ntok - c2.onset).median())
```