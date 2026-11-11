The gate cut the number of truncated samples from 156 (gen_9) to 67. Of these, 46 are still exact loops (9% of all samples; median period 42, onset median 12.7k tokens). The other 21 are non-repeating long searches, nearly all on intermediate-algebra/precalculus problems that are probably beyond the model's ability. (The third value from loopinfo is not a repetition fraction for looping samples, so I ignore those quartiles.) The bigger target is the 118 finished-but-wrong answers plus 14 finished without an answer. Next I'll check whether these are mostly hard problems (MATH level, subject). I'll also check whether the reference answer appears somewhere in the reasoning, which would mean it was found and then abandoned. If the reference appears often, there is a decision/verification failure to look into; if not, it is mostly a capability limit.

```python
# Characterize finished-wrong & noans in gen_12: difficulty (MATH level), length, whether ref answer appears in the text.
M = {json.loads(l)['id']:json.loads(l) for l in open('/practice/math_train.jsonl')}
Gs = {json.loads(l)['id']:json.loads(l) for l in open('/practice/gsm8k_train.jsonl')}
ref = {r['id']:r['answer'] for r in g12}
rows12=[]
for r in g12:
    for i,s in enumerate(r['samples']):
        st=T12[(r['id'],i)]['st']; lev = M[r['id']]['level'] if r['id'] in M else 0
        th = s['text'].split('</think>')[0]
        rows12.append(dict(id=r['id'],i=i,st=st,src=r['source'],lev=lev,t=s['tokens'],
                           found=bool(ans_found(th, r['answer'])), pred=s['pred']))
import pandas as pd
D = pd.DataFrame(rows12)
print(pd.crosstab(D.lev, D.st))
print(D.groupby('st').agg(n=('id','size'), tok_med=('t','median'), ref_in_think=('found','mean')).round(2))
print('ok->wrong vs gen9 ids:', [k[0] for k in T12 if T9[k]['st']=='ok' and T12[k]['st']=='wrong'][:20])
```