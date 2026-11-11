x5 + E1 reached 52.5% accuracy (MATH 48.8%, GSM8K 56.3%) against 46.3% for x5. The cut share fell from 48.4% to 30.5%. Wrong finished answers rose from 36 to 89, so many rescued loops end in a wrong answer, but more of them end correct. This fits the race model. Under the brake, weakening the three induction heads lowers the loop-entry hazard and converts loops into finished answers, whereas the same edit on the original model did almost nothing. Now I need the paired difference against x5 with its standard error, by source and set. I also want to know where the wrong answers come from: previously looping samples (expected) or samples that were correct under x5 (collateral damage to copying). Finally, I will check whether the remaining cut samples are still exact loops. A paired gain clearly above two standard errors, with few correct→wrong flips among x5's finished answers, would justify keeping E1. Many such flips would indicate the induction heads carry useful copying, and that a gentler dose may be better.

```python
# Pass-8 readout: paired (x5+E1) vs x5; where do new wrong answers come from (rescued loops vs collateral on previously-correct finished)?
g9 = readj('/work/gen_9.jsonl'); r9 = {r['id']: r for r in g9}
d = np.array([tocor(r9[i]) - tocor(r8[i]) for i in ids_]); print('x5+E1 - x5 all:', paired(d))
for src in ['math','gsm8k']:
    dd = np.array([tocor(r9[i]) - tocor(r8[i]) for i in ids_ if r8[i]['source']==src]); print(f'  {src}:', paired(dd))
for nm, sel in [('setA', set(sp['setA'])), ('setB', set(sp['setB']))]:
    dd = np.array([tocor(r9[i]) - tocor(r8[i]) for i in ids_ if i in sel]); print(f'  {nm}:', paired(dd))
d3 = np.array([tocor(r9[i]) - tocor(r7[i]) for i in ids_]); print('x5+E1 - x3 (E2):', paired(d3))
# transition table by x5 state (finished-correct / finished-wrong / cut) -> x5+E1 state
def st(r): return 'cut' if cutf(r) else ('fin_ok' if tocor(r) else 'fin_wrong')
tab = collections.Counter((st(r8[i]), st(r9[i])) for i in ids_)
for a in ['fin_ok','fin_wrong','cut']: print(f'  x5 {a:9s} ->', {b: tab[(a,b)] for b in ['fin_ok','fin_wrong','cut']})
fin9 = [r9[i] for i in ids_ if not cutf(r9[i])]
print('x5+E1 finished acc %.1f (%d); x5 finished acc %.1f' % (100*np.mean([tocor(r) for r in fin9]), len(fin9), 100*np.mean([tocor(r8[i]) for i in ids_ if not cutf(r8[i])])))
cut9 = [r9[i] for i in ids_ if cutf(r9[i])]; inf9 = [onset_info(S(r)['text']) for r in cut9]
ex9 = [x for x in inf9 if x[1] is not None]
print('x5+E1 cut %d; exact loops %d (%.0f%%); period med %.0f; onset med %.0f' % (len(cut9), len(ex9), 100*len(ex9)/len(cut9), np.median([x[1] for x in ex9]), np.median([x[2] for x in ex9])))
```