x5 + E1 beats x5 by +6.2 ± 2.2 points (81 wrong→right, 49 right→wrong; 2.8 standard errors), with the gain present in both setA (+5.1) and setB (+7.4). The gain is larger on MATH (+9.8 ± 3.2) than on GSM8K (+2.7 ± 3.0). Against E2 (brake x3) the combined edit gains +9.0 ± 2.1. The mechanism fits the race model: loop onset moved from a median of 6.3k to 11.4k tokens, and the cut share fell from 48% to 30%. Of x5's 248 cut samples, 132 now finish, 79 of them correctly. The cost: of x5's 228 correct answers, 19 became wrong and 29 became cut, and finished-answer accuracy fell from 86% to 75%. Part of that drop is simply that rescued problems are harder. To tell real collateral damage to copying from sampling churn, I will compare against the x3→x5 transition table, where only the brake dose changed. I will also check how long the new wrong answers are: long, confused finishes would mean they are rescued loops, while short ones on previously solved problems would mean damaged copying. This decides whether to remove more induction capacity next (tier-2 heads) or to stop at three heads.

```python
# Calibrate churn: x3->x5 transition table (no induction change) vs x5->x5+E1; lengths of new wrong answers (rescued loops vs collateral on easy ones).
tab35 = collections.Counter((st(r7[i]), st(r8[i])) for i in ids_)
print('x3 -> x5 (brake dose only):')
for a in ['fin_ok','fin_wrong','cut']: print(f'  x3 {a:9s} ->', {b: tab35[(a,b)] for b in ['fin_ok','fin_wrong','cut']})
# wrong answers under x5+E1: lengths by origin
wr = [(i, int(S(r9[i])[LEN]), st(r8[i])) for i in ids_ if st(r9[i])=='fin_wrong']
for org in ['fin_ok','fin_wrong','cut']:
    L_ = [l for i,l,o in wr if o==org]
    if L_: print(f'x5+E1 wrong, x5 state {org:9s}: n={len(L_):3d}, length median {np.median(L_):.0f}, share >12k {np.mean(np.array(L_)>12000):.0%}')
ok9 = [int(S(r9[i])[LEN]) for i in ids_ if st(r9[i])=='fin_ok']; print('x5+E1 correct: length median %.0f' % np.median(ok9))
# arithmetic/copy errors among fin_ok->fin_wrong: compare predicted vs reference
for i,l,o in wr:
    if o=='fin_ok': print(f"  {i[:30]:30s} ref={str(r9[i]['answer'])[:12]:12s} x5+E1 pred={str(S(r9[i])['pred'])[:14]:14s} len={l}")
```