E2 is a large jump: accuracy 21.5% (MATH 20.3%, GSM8K 25.0%), against 12.3% for the original and 14.8% for E1 on the same problems. Cut-offs fell to 77% and finished-but-wrong answers rose to 13.

Before trusting this I'll check:
- the paired statistics against the original and against E1, by source;
- whether the mechanism behaved as predicted: a lower late-answer "Wait" rate, and a lower mean activation of neuron 998 itself (the feedback-loop signature, since only its output was changed);
- how loop onset and finishing changed;
- whether the extra finished answers include a worrying share of wrong ones, i.e. lost verification.

If the gain is several standard errors and the neuron's own activation drops, the doubt-spiral repair is confirmed on these problems. It will still need replication on fresh problems.

```python
# Paired stats E2 vs orig and vs E1; Wait-rate dynamics; n998 own activation (feedback signature); loop onset; finished accuracy.
g7 = [json.loads(l) for l in open('/work/gen_7.jsonl')]
e2_by = {r['id']: r['samples'][0] for r in g7}
c2 = np.array([e2_by[i]['correct'] for i in common], float)
for nm, base in [('vs orig', oc), ('vs E1', ec)]:
    dd = c2 - base
    print(f'E2 {nm}: diff {dd.mean():+.4f} SE {dd.std(ddof=1)/np.sqrt(len(dd)):.4f} | w->r {(dd>0).sum()} r->w {(dd<0).sum()}', end=' || ')
    for src in ['math', 'gsm8k']:
        sel = np.array([i.startswith(src) for i in common]); print(f'{src} {dd[sel].mean():+.3f} (SE {dd[sel].std(ddof=1)/np.sqrt(sel.sum()):.3f})', end=' ')
    print()
rng = np.random.default_rng(1)
for name, by in [('orig', o_by), ('E1', e_by), ('E2', e2_by)]:
    cuts = [i for i in common if by[i]['cut']]
    sample = rng.choice(cuts, 60, replace=False)
    wb = np.nanmean([wait_bins(by[i]['text']) for i in sample], axis=0)
    ons = [tok_onset(by[i]['text']) for i in cuts]; found = [o for o in ons if o is not None]
    fin = [i for i in common if not by[i]['cut']]
    allw = np.mean([1000 * len(re.findall(r'\bWait\b', by[i]['text'])) / max(by[i]['tokens'], 1) for i in common])
    print(f'{name}: Wait/1k by bin {np.round(wb,1).tolist()} | all-answer mean Wait/1k {allw:.1f} | loops {len(found)}/{len(cuts)} onset median {np.median(found):.0f} | finished {len(fin)} acc {np.mean([by[i]["correct"] for i in fin]):.2f} median tokens {np.median([by[i]["tokens"] for i in fin]):.0f}')
mm7 = np.load('/work/gen_7_acts/mlp_mean_answer.npy', mmap_mode='r')
idx7 = json.load(open('/work/gen_7_acts/index.json')); samples7 = eval(idx7['samples']) if isinstance(idx7['samples'], str) else idx7['samples']
k7 = {s['item']: k for k, s in enumerate(samples7)}; k1 = {s['item']: k for k, s in enumerate(samples1)}
a1 = np.array([mm[k1[i], 16, 998] for i in common]); a7 = np.array([mm7[k7[i], 16, 998] for i in common])
print(f'n998 mean activation (first 8k answer tokens): orig {a1.mean():.3f} E2 {a7.mean():.3f}; paired diff {np.mean(a7-a1):+.3f} (SE {np.std(a7-a1, ddof=1)/np.sqrt(len(a1)):.3f})')
```