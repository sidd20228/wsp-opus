Exempting L23H4 confirms its double role. Format failures fell from 14 to 4, so L23H4 does carry the \boxed format at the final line. But truncations rose from 67 to 108, because its line-start copying also drives loops. Net accuracy is 59.4% vs 61.1%, with an odd split: math went up (55.1 vs 52.0) and gsm down (63.7 vs 70.3). I'll compute the paired difference against gen_12 by source and split, plus the outcome transitions, to see whether the extra cuts are the GSM loss and whether the math gain is beyond noise. That decides whether to keep full gating, drop it, or look for an intermediate design (e.g., partial gating strength on L23H4).

```python
# Paired gen_13 (E5b: L23H4 exempt) vs gen_12 (gate 7): by source/split, transitions, failure counts by source.
g13 = readj('/work/gen_13.jsonl'); T13 = table(g13)
print('vs gen_12 all', paired(T13,T12))
for src in ['math','gsm8k']: print('   ',src, paired(T13,T12,lambda k,v:v['src']==src))
print('    setA', paired(T13,T12,lambda k,v:k[0] in setA), '| setB', paired(T13,T12,lambda k,v:k[0] in setB))
print('vs gen_9 (x5+E1)', paired(T13,T9))
tr = collections.Counter((T12[k]['st'],T13[k]['st']) for k in T13); print('gen12->gen13:', sorted(tr.items(), key=lambda x:-x[1]))
for nm,T in [('gen12',T12),('gen13',T13)]:
    print(nm, {src: dict(collections.Counter(v['st'] for v in T.values() if v['src']==src)) for src in ['math','gsm8k']})
```