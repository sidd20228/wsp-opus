Both texts now contain typical cycles (periods 50 and 82). Since the forward request must contain the texts literally, I'll shorten them (400 pre-onset tokens + 6 cycles) so both fit in one printout. I'll print them JSON-escaped (ASCII-safe, non-ASCII as \u escapes) so I can paste them exactly. The induction analysis needs only a few cycles plus some preceding context, so the shorter texts lose little.

```python
# Hypothesis (same): shorter looping texts for the attention pass; print JSON-escaped for exact pasting.
sel=[]
for rid,pp in [('gsm8k_train_2226',50),('math_train_algebra_164',82)]:
    i = R1[(R1.id==rid)&(R1.cut)&(R1.period==pp)].index[0]
    txt,n,p = build(i, pre=400, ncyc=6); sel.append((rid,i,n,p,txt))
    print(rid, n, p, file=open('/dev/stderr','w'))
ATT_TEXTS=[s[4] for s in sel]
json.dump(ATT_TEXTS, open('/work/attn_texts.json','w'))
print([ (s[0],s[2],s[3]) for s in sel])
print(json.dumps(ATT_TEXTS, ensure_ascii=True))
```