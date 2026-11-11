The pass-1 file stores the generations nested in a `samples` list, so my field lookup found nothing. I'll check the sample record fields and then redo the selection of a short, correct, finished, pure-ASCII answer to use as the control text. Nothing about the hypothesis changes. This step only builds the control for the coming ablation test.

```python
# Inspect nested sample structure, then select a short correct finished ASCII answer as control text.
s0 = g1[0]['samples'][0]
print({k: (v if not isinstance(v, str) or len(v) < 60 else f'<str {len(v)}>') for k, v in s0.items()})
cands = []
for r in g1:
    for s in r['samples']:
        txt = s.get('text') or s.get('completion') or s.get('output')
        ok = s.get('correct', s.get('grade'))
        fin = s.get('finish_reason', '') != 'length' and not s.get('cut', s.get('truncated', False))
        if ok and fin and txt is not None:
            n = len(tok.encode(txt).ids)
            if 500 <= n <= 1100 and all(ord(c) < 128 for c in txt):
                cands.append((n, r['id'], r['source']))
print(len(cands), sorted(cands)[:15])
```