Of the 457 cut-off answers, 96% contain a verbatim loop (six or more consecutive repeated lines). 52% state the reference answer somewhere ("is X", "= X", "answer is X"), almost always (220 of 236) before the loop begins. Only 10 put a correct answer in \boxed{}.

My pattern is crude, though: for small integer answers, "= 4" can appear by accident. I'll run a control with the reference answers shuffled across problems. If the shuffled match rate is much lower than 48% (220/457), then roughly half of the cut-off answers had reached the right result before they locked into a loop. In that case loop-breaking has a large potential upside. If the control rate is similar, the match is mostly noise.

```python
# Control for the crude 'answer stated before loop' match: shuffle reference answers across problems.
rng = np.random.default_rng(0)
refs = [r['answer'] for r in g1]; perm = rng.permutation(len(refs))
ctrl = dict(n=0, stated=0, before=0); real_by_src = {}
for idx, r in enumerate(g1):
    fake = str(refs[perm[idx]]).strip()
    if fake == str(r['answer']).strip(): continue
    for s in r['samples']:
        if not s['cut']: continue
        t = s['text']; ctrl['n'] += 1
        on = loop_onset_char(t)
        m = re.search(r'(?:answer is|=|is)\s*\$?\\?(?:boxed\{)?' + re.escape(fake) + r'(?![0-9])', t)
        ctrl['stated'] += m is not None
        if m is not None and on is not None and m.start() < on: ctrl['before'] += 1
print('control (shuffled refs):', ctrl, ' rate before loop %.2f' % (ctrl['before'] / ctrl['n']))
print('real: rate before loop %.2f' % (220 / 457))
```