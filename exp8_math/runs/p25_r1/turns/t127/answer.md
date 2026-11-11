The helper source is not retrievable because the functions were defined inside exec. Their signatures should still be visible, and if the one-argument form is obvious I can test a helper directly. Otherwise I'll write a small, transparent periodic-tail detector: smallest period p for which the tail repeats exactly, with the onset found by extending the match backward. That is the same definition used in the notes (exact periodic tail, period, onset).

```python
# Signatures only (source unavailable), then a quick trial of period_and_onset on one gen_12 cut sample.
for nm in ['period_and_onset','loopinfo','onset_info','tail_period','find_period','text_stats','ans_found']:
    try: print(nm, inspect.signature(globals()[nm]))
    except Exception as e: print(nm, e)
cut12 = [(r['id'],i,s) for r in g12 for i,s in enumerate(r['samples']) if s['cut']]
print(len(cut12))
try: print(period_and_onset(cut12[0][2]['text']))
except Exception as e: print('err', e)
```