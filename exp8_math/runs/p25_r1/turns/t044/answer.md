Pass 4 shows that E1 barely matters. On 256 fresh dev problems the original scores 7.8% with 93% of answers cut at 32,768 tokens. E1 scores 9.4%, a paired difference of +1.6 ± 2.0 points (15 wrong→right, 11 right→wrong), and the cut rate only falls to 90.6%. Finished answers remain almost always correct (100% original, 96% E1). So non-termination is still essentially the whole accuracy problem, but three induction heads are not its main driver.

I need to know what the cut answers look like under E1: are they still exact periodic loops (copying is redundant and other components sustain it), or did the loops become inexact rambling (the underlying stuck state persists)? I also want to check whether, in cut samples, the correct answer already appears before the loop starts. If so, the failure is the model never committing and closing its reasoning, not an inability to solve the problem.

```python
# Hypothesis: under E1 the cut answers are still exact periodic loops; and in most cut samples the correct answer was already found before loop onset.
import re
def ids_of(t): return np.array(tok.encode(t).ids, dtype=np.int64)
def tail_period(ids, maxp=2000, check=600):
    n = len(ids)
    if n < check + 50: return None, None
    for p in range(1, maxp):
        if n - check - p < 0: break
        if np.array_equal(ids[n-check:], ids[n-check-p:n-p]):
            eq = ids[p:] == ids[:-p]; bad = np.where(~eq)[0]
            return p, int(bad[-1] + 1) if len(bad) else 0
    return None, None
def ans_found(text, ans):
    a = ans.replace(' ', '').replace('\\!', '').strip('$')
    t = text.replace(' ', '')
    if re.fullmatch(r'-?[\d,]*\.?\d+', a):
        a = a.replace(',', ''); t2 = re.sub(r'(?<=\d),(?=\d{3})', '', t)
        return re.search(r'(?<![\d.])' + re.escape(a) + r'(?![\d]|\.\d)', t2) is not None
    return a in t
res = {}
for nm, recs in [('orig', g5o), ('E1', g5e)]:
    rows = []
    for r in recs:
        for s in r['samples']:
            if not s['cut']: continue
            ids = ids_of(s['text']); p, on = tail_period(ids)
            pre = tok.decode(ids[:on].tolist()) if on is not None else s['text']
            rows.append(dict(id=r['id'], src=r['source'], per=p, onset=on, ntok=len(ids), found=ans_found(pre, r['answer']),
                             waits=len(re.findall(r'\bWait\b', pre)) / max(1, (on if on else len(ids))) * 1000))
    res[nm] = pd.DataFrame(rows)
    d = res[nm]
    print(nm, 'n_cut', len(d), 'exact-periodic %.2f' % d.per.notna().mean(), 'period med %s' % d.per.median(),
          'onset q10/50/90', d.onset.quantile([.1, .5, .9]).round(0).tolist(), 'ans found pre-onset %.2f' % d.found.mean(),
          'Wait/1k pre %.1f' % d.waits.median())
```