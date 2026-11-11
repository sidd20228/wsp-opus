The wrong GSM8K answers split into two groups. In 23 of 59 the correct value was computed at some point, 16 of them within the first 4k tokens, and the run then drifted away from it. In the other 36 the correct value never appears, which is mostly misreading the problem. Correct runs reach their answer early (median 646 tokens). I'll look at what the model writes right after the correct value in the abandoned cases. If departures start with a consistent doubt or re-interpretation opener ("However", "But maybe", "Let me re-read", "Alternatively"), there may be a specific pathway behind the drift that I can examine. If they vary, this is general overthinking, and I won't attack it with an early-closing hack, which the task rules out.

```python
# What follows the first correct value in abandoned GSM runs? Print ~220 chars after the hit for several cases.
cases=[]
for k,v in T12.items():
    if v['src']!='gsm8k' or v['st']!='wrong': continue
    s=byid[k[0]]['samples'][k[1]]; th=s['text'].split('</think>')[0]; p=first_hit(th.replace(',',''), ref[k[0]])
    if p is not None: cases.append((k, p, th.replace(',','')))
print(len(cases))
for k,p,th in cases[:9]:
    nxt = th[p:p+260]
    print(f"### {k[0]} ref={ref[k[0]]} pred={byid[k[0]]['samples'][k[1]]['pred']} | {asc(nxt).replace(chr(10),' / ')}")
```