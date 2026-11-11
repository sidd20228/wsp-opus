The surviving failures are a different kind of loop. In the original model, 91% of loop units contained a doubt word. Under the brake only 10.6% do, yet 254 of 512 samples still loop, with a similar period (about 44 tokens) and onset (median 7.3k tokens). In 73% of GSM8K cases the right answer was already written before the loop started. The new loops are bullet-list restatements: the model re-lists the same facts ("* x-y = 3", "* x+y = -1", "* E is midpoint of AD", ...), often at deep indentation. So the brake removed the doubt-driven cycle, but the underlying copy-and-continue dynamics now find another cycle: non-doubt loops went from about 41 samples to about 227. A stronger brake is therefore unlikely to be the main next lever, and the next target is how restatement loops start. I will read the text just before several loop onsets and measure the indentation depth inside these loops against pre-loop text, to see whether the onset is a recognizable decision (for example an end-of-list point where the model restarts the list instead of concluding).

```python
# Hypothesis: x2 surviving loops start at end-of-list decision points (model restarts a bullet list instead of concluding), often deeply nested.
def squeeze(s): return re.sub(r' {3,}', lambda m: f'<{len(m.group())}sp>', s)
nd = LU11[~LU11.doubt]
for _, r in nd.sample(5, random_state=1).iterrows():
    rec = next(x for x in g11r if x['id'] == r['id']); s = rec['samples'][int(r['si'])]
    enc_ = tok.encode(s['text']); cpos = enc_.offsets[r['onset']][0]
    print(f"--- {r['id']} si{r['si']} period {r['period']} onset {r['onset']}")
    print('PRE :', asc_(squeeze(s['text'][max(0, cpos-260):cpos])).replace('\n', ' / '))
    print('LOOP:', asc_(squeeze(s['text'][cpos:cpos+160])).replace('\n', ' / '))
def max_indent(t): return max([len(l) - len(l.lstrip(' ')) for l in t.split('\n') if l.strip()] + [0])
print('median max-indent in loop unit: x2 nondoubt', np.median([max_indent(u) for u in nd.unit]), '| orig doubt units', np.median([max_indent(u) for u in LU2[LU2.doubt].unit]))
```