39 cut answers have their loop starting between 600 and 1,400 tokens, so a 2,048-token window can hold both the normal reasoning and several hundred tokens of loop. To pick two representative texts for the attention pass, I'll print, for several candidates, a short excerpt just before the onset and the looped unit itself. I want one single-line loop (the most common type, 173/435) and one multi-line loop (period 2–4 lines), both clearly verbatim, so the head analysis covers both regimes.

```python
# Inspect a few early-onset candidates: excerpt before onset + looped unit, and loop period
for i,ntok,cpos in cands[:14]:
    t=dict(cut_samples)[i]
    lines=[l for l in t.split('\n') if l.strip()]
    tail=lines[-400:]
    per=next((p for p in range(1,60) if np.mean([tail[k]==tail[k-p] for k in range(p,len(tail))])>0.9), None)
    pre=t[max(0,cpos-150):cpos].replace('\n','/'); unit=t[cpos:cpos+160].replace('\n','/')
    print(i[11:40], ntok, 'period', per, '| PRE:', pre.encode('ascii','replace').decode()[-120:], '| LOOP:', unit.encode('ascii','replace').decode()[:150])
```