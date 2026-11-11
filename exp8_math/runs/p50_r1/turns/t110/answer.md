The selection loop's counter checked the wrong kind label, so it kept all 104 restatement-loop texts instead of 30 and the set reached 861k tokens, over the 524k limit. The texts themselves are correctly aligned (every one passed the token check). I will keep the first 30 loop_x2 texts (random order, one per problem), recompute the total, and save. While the tokens allow, I will also add a few more finished texts, since side effects on normal reasoning are the main risk of any loop edit.

```python
# Hypothesis: (setup) trim restatement-loop texts to 30 so dense_v2 fits the 524,288-token TF limit; save instrument.
lx = [v for v in V2 if v['kind'] == 'loop_x2'][:30]
V2 = lx + [v for v in V2 if v['kind'] != 'loop_x2']
tot = sum(v['ntok'] for v in V2); print(Counter(v['kind'] for v in V2), 'total tokens', tot, 'max', max(v['ntok'] for v in V2))
print('loop_x2 onset med', int(np.median([v['onset'] for v in lx])), 'period med', int(np.median([v['period'] for v in lx])), '| fin_x2 ntok med', int(np.median([v['ntok'] for v in V2 if v['kind'] == 'fin_x2'])))
json.dump(V2, open('/work/dense_v2.json', 'w')); print(len(V2))
```