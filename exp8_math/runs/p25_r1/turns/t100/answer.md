The suppression heads do not track the loop's copy target. Their attention mass on i−kp+1 is at most 0.10 (L15H3); some put 0.10–0.24 on the same token one cycle back (L23H2, L7H0, L3H4, L11H6). That would only discourage immediately repeating the current token, not copying the cycle. So strengthening them is unlikely to break loops, and I am dropping that route. The cycle copying is carried by the induction heads (L15H7 0.57, L23H5 0.43–0.47, L11H1 0.35–0.39, then L19H5 and L23H0/H4/H7). Removing the top three (E1) barely helped the original model, but there almost every sample was trapped by "Wait" doubting and never finished anyway. Under the brake edit the situation may differ: a sample now either finishes or falls into a loop first. If finishing times and loop-onset times overlap broadly, delaying onset by weakening induction (E1 delayed it by about 17%) should convert some loops into finished answers, even though E1 alone did nothing. I will compare the token-length distribution of x5's finished answers with the loop-onset distribution of its cut answers, and estimate from the data how many more samples would finish if onsets came later.

```python
# Race hypothesis: under x5, do finishing times and loop-onset times overlap, so that delaying loop onset (weaker induction) would let more samples finish?
fin8 = [r for r in g8 if not cutf(r)]; cut8 = [r for r in g8 if cutf(r)]
fl = np.array([int(S(r)[LEN]) for r in fin8])
inf8 = [onset_info(S(r)['text']) for r in cut8]; ons = np.array([x[2] for x in inf8 if x[1] is not None])
print('x5 finished n=%d length pct 10/25/50/75/90:' % len(fl), np.percentile(fl, [10,25,50,75,90]).astype(int))
print('x5 cut n=%d, exact loops %d; onset pct 10/25/50/75/90:' % (len(cut8), len(ons)), np.percentile(ons, [10,25,50,75,90]).astype(int))
# survival-style view: of samples alive (not finished, not looping) at token t, what fraction loops vs finishes in the next window?
events = [(l, 'fin') for l in fl] + [(o, 'loop') for o in ons]
for lo, hi in [(0,3000),(3000,6000),(6000,10000),(10000,16000),(16000,32768)]:
    alive = sum(1 for t,_ in events if t >= lo) + (len(cut8) - len(ons))
    f = sum(1 for t,e in events if lo <= t < hi and e=='fin'); lp = sum(1 for t,e in events if lo <= t < hi and e=='loop')
    print(f'  window {lo:5d}-{hi:5d}: alive {alive:3d}  finish {f:3d} ({f/alive:.0%})  enter loop {lp:3d} ({lp/alive:.0%})')
# simple counterfactual: if every onset were delayed by factor 1.2 / 1.5, and finishing hazard after original onset matched the empirical finish hazard at that time
```