Wrong answers come from all levels, including 59 of the 256 GSM8K samples (23%), even though GSM8K problems are easy for this model when it finishes cleanly. Wrong samples are long (median 13.4k tokens vs 5.8k for correct). The reference appears in the thinking of 47% of them, but for small integers that test is weak. The GSM8K errors are the most informative, because those problems are within the model's ability. I'll read short excerpts of several GSM8K wrong samples: predicted vs reference answer, length, and the last lines before </think>. The question is whether they look like misreadings, arithmetic slips, an answer abandoned under endless re-checking, or commitment to an answer that was never verified.

```python
# Read short excerpts of GSM8K wrong samples (pred vs ref, length, ending of thinking) to find the failure mode.
gw = D[(D.src=='gsm8k')&(D.st=='wrong')].sort_values('t')
print(gw.t.describe().round(0).to_dict())
byid = {r['id']:r for r in g12}
for _,x in gw.iloc[::7].head(8).iterrows():
    s = byid[x.id]['samples'][x.i]; th = s['text'].split('</think>')[0]
    print(f"\n### {x.id} pred={x.pred} ref={ref[x.id]} tok={x.t} found={x.found}")
    print(asc(th[-300:]).replace('\n',' | '))
```