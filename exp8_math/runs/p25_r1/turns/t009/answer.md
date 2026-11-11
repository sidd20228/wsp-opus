Each record holds a problem id, source, reference answer and a list of samples, each with its text. I need the per-sample fields to build a table of length, correctness and whether `</think>` was ever emitted. My hypothesis is that most cut-off answers never close their thinking block. If most cut-offs contain no `</think>` and no `\boxed`, the problem is a runaway thinking phase; if they do contain them, the failure is post-thinking. This block only inspects the sample structure and tabulates these flags.

```python
# Hypothesis: cut-offs are dominated by runaway thinking that never emits </think>. Tabulate per-sample flags.
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