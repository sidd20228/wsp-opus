I picked the wrong sample of each problem: the first cut sample, with periods 1 and 28, rather than the ones I inspected (periods 50 and 82). A period-1 loop (one token repeated) is not representative. I'll select the samples by their period so the texts contain typical ~50-80-token verbatim cycles, then rebuild and check the token counts stay under 2,048.

```python
# Hypothesis (same): choose the representative loop samples (p=50, p=82) and rebuild texts under 2048 tokens.
sel=[]
for rid,pp in [('gsm8k_train_2226',50),('math_train_algebra_164',82)]:
    i = R1[(R1.id==rid)&(R1.cut)&(R1.period==pp)].index[0]
    txt,n,p = build(i, pre=1000, ncyc=7); sel.append((rid,i,n,p,txt)); print(rid, i, n, p, int(R1.onset[i]))
open('/work/attn_texts.json','w').write(json.dumps([s[4] for s in sel]))
print(asc(sel[0][4][-400:]))
```