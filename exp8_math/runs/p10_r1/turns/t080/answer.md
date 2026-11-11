Removing only the "wait" neuron (N0) does better than E3: 34.2% accuracy (MATH 29.7%, GSM8K 47.7%), with 66% cut-offs and only 5 finished-but-wrong answers (E3 had 35). So once the neuron is gone, E1's head removal hurts. Its number-copying cost probably shows up as wrong finished answers.

Before settling on N0, I'll compute:
- the paired statistics of N0 against E3 and against the original;
- whether N0 has earlier or more verbatim loops than E3 (E1 was the anti-loop component);
- the per-position "Wait" profile, including the early, healthy rate;
- thinking length and accuracy on problems both the original and N0 finish.

If N0 beats E3 by more than the standard error, with fewer wrong answers and not much more looping, the simpler edit is the submission. It will then need confirmation on fresh problems.

```python
# N0 (n998 x0 only) vs E3 vs orig: paired stats, loops, Wait profile, finished-by-both, wrong finished.
g9 = [json.loads(l) for l in open('/work/gen_9.jsonl')]
n0_by = {r['id']: r['samples'][0] for r in g9}
c0 = np.array([n0_by[i]['correct'] for i in common], float)
for nm, base in [('vs orig', oc), ('vs E3', c3), ('vs E2', c2)]:
    dd = c0 - base
    print(f'N0 {nm}: diff {dd.mean():+.4f} SE {dd.std(ddof=1)/np.sqrt(len(dd)):.4f} | w->r {(dd>0).sum()} r->w {(dd<0).sum()}', end=' || ')
    for src in ['math', 'gsm8k']:
        sel = np.array([i.startswith(src) for i in common]); print(f'{src} {dd[sel].mean():+.3f} (SE {dd[sel].std(ddof=1)/np.sqrt(sel.sum()):.3f})', end=' ')
    print()
rng = np.random.default_rng(1)
smp = rng.choice(common, 120, replace=False)
for name, by in [('orig', o_by), ('E3', e3_by), ('N0', n0_by)]:
    allb = np.nanmean([wait_bins(by[i]['text']) for i in smp], axis=0)
    cuts = [i for i in common if by[i]['cut']]; ons = [tok_onset(by[i]['text']) for i in cuts]; f_ = [o for o in ons if o is not None]
    fin = [i for i in common if not by[i]['cut']]
    print(f'{name}: Wait/1k {np.round(allb,1).tolist()} | cut {len(cuts)} loops {len(f_)} onset med {np.median(f_):.0f} | finished {len(fin)} acc {np.mean([by[i]["correct"] for i in fin]):.2f} med tok {np.median([by[i]["tokens"] for i in fin]):.0f}')
bf = [i for i in common if not o_by[i]['cut'] and not n0_by[i]['cut']]
lo = np.array([o_by[i]['tokens'] for i in bf]); ln = np.array([n0_by[i]['tokens'] for i in bf])
print(f'finished by orig & N0: n={len(bf)} median tok orig {np.median(lo):.0f} N0 {np.median(ln):.0f}; N0 shorter in {np.mean(ln<lo):.2f}; acc orig {np.mean([o_by[i]["correct"] for i in bf]):.2f} N0 {np.mean([n0_by[i]["correct"] for i in bf]):.2f}')
print('N0 lost vs orig:', sum(1 for i in common if o_by[i]['correct'] and not n0_by[i]['correct']), '| N0 wrong finished:', [(i, n0_by[i]['pred']) for i in common if not n0_by[i]['cut'] and not n0_by[i]['correct']])
```