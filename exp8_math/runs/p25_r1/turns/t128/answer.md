The signatures show that period_and_onset expects a token array, not text. loopinfo(text) and onset_info(text) take raw text, so they are probably what I used for the per-pass loop statistics. I'll call both on one cut sample to see their output format. If they return period and onset, I'll apply them to all 67 cut samples and compare with gen_9's cut samples.

```python
# Try text-level helpers loopinfo/onset_info on one cut sample to learn their output format.
t = cut12[0][2]['text']
for f in (loopinfo, onset_info):
    try: print(f.__name__, f(t))
    except Exception as e: print(f.__name__, 'err', e)
ids0 = tok.encode(t).ids
try: print('period_and_onset(ids)', period_and_onset(np.array(ids0)))
except Exception as e: print('err', e)
print('tail_period', tail_period(ids0))
```