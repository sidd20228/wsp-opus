Removing the three loop-driving heads raised accuracy from 12.3% to 14.8% on the same 512 problems:
- MATH: 10.7% → 12.8%
- GSM8K: 17.2% → 21.1%

Cut-off answers fell from 89% to 84%. Finished-but-wrong answers rose from 1 to 8, which is a hint of the number-copying cost.

A +2.5-point gain on 512 items is about the size of an unpaired standard error. Next I need:
- the paired statistics: wrong→right versus right→wrong flips, and the paired standard error;
- whether the remaining cut-offs are still verbatim loops, and whether loops now start later;
- what the 8 wrong finished answers look like.

The edit only partly broke the loop mechanism, consistent with the TF finding that list-type decision points stayed above the branching threshold. I want to know whether the gain comes from loops starting later or from loops being escaped. That decides the direction of the next variant: stronger, broader, or more targeted.

```python
# Paired comparison pass1 (orig) vs pass4 (E1 full ablation) on same 512 problems; loop onset & 'Wait' dynamics; wrong finished answers.
g6 = [json.loads(l) for l in open('/work/gen_6.jsonl')]
o_by = {r['id']: r['samples'][0] for r in g1}; e_by = {r['id']: r['samples'][0] for r in g6}
common = [i for i in diag_ids if i in o_by and i in e_by]
oc = np.array([o_by[i]['correct'] for i in common], float); ec = np.array([e_by[i]['correct'] for i in common], float)
d = ec - oc
print(f'n={len(common)} orig {oc.mean():.3f} edited {ec.mean():.3f} diff {d.mean():+.4f} SE {d.std(ddof=1)/np.sqrt(len(d)):.4f} | wrong->right {(d>0).sum()} right->wrong {(d<0).sum()}')
for src in ['math', 'gsm8k']:
    sel = np.array([i.startswith(src) for i in common])
    print(f'  {src}: n={sel.sum()} diff {d[sel].mean():+.4f} SE {d[sel].std(ddof=1)/np.sqrt(sel.sum()):.4f} w->r {(d[sel]>0).sum()} r->w {(d[sel]<0).sum()}')
def tok_onset(t):
    c = loop_onset_char(t)
    return None if c is None else len(tok.encode(t[:c]).ids)
def wait_rate(t):
    n = len(tok.encode(t).ids); return 1000 * len(re.findall(r'\bWait\b', t)) / max(n, 1)
for name, by in [('orig', o_by), ('edit', e_by)]:
    cuts = [by[i] for i in common if by[i]['cut']]
    ons = [tok_onset(s['text']) for s in cuts]
    found = [o for o in ons if o is not None]
    fin = [by[i] for i in common if not by[i]['cut']]
    print(f'{name}: cut {len(cuts)} loops found {len(found)} onset median {np.median(found):.0f} IQR {np.percentile(found,25):.0f}-{np.percentile(found,75):.0f} | finished {len(fin)} mean tokens {np.mean([s["tokens"] for s in fin]):.0f} correct {np.mean([s["correct"] for s in fin]):.2f}')
wrong_fin = [(i, e_by[i]['pred'], [r for r in g6 if r['id'] == i][0]['answer'], e_by[i]['tokens']) for i in common if not e_by[i]['cut'] and not e_by[i]['correct']]
print('edited wrong finished (id, pred, ref, tokens):', wrong_fin)
```