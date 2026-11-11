```python
# Hypothesis: same as before - inspect sample structure and the tail of cut-off texts.
s0 = G1[0]['samples'][0]
print({k:(str(v)[:120] if k!='text' else len(v)) for k,v in s0.items()})
rows=[]
for g in G1:
    for j,s in enumerate(g['samples']):
        t = s['text']
        rows.append(dict(id=g['id'], src=g['source'], j=j, ntok=s.get('n_tokens', s.get('tokens')), correct=s.get('correct'),
                         cut=s.get('cut', None), has_end_think=('</think>' in t), nboxed=t.count('\\boxed'), L=len(t)))
R1 = pd.DataFrame(rows)
print(R1.head())
print(R1[['has_end_think']].mean(), (R1.nboxed>0).mean())
```