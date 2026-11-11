E3 (E1 with neuron 998's output fully removed) reaches 30.3% accuracy on the same 512 problems (MATH 27.3%, GSM8K 39.1%), with 63% cut-offs. The neuron edit shows a clean monotone dose-response:

| Neuron 998 output | Edit | Accuracy | Cut-offs |
|---|---|---|---|
| ×1 | E1 | 14.8% | 84% |
| ×0.5 | E2 | 21.5% | 77% |
| ×0 | E3 | 30.3% | 63% |

The original model is at 12.3%. That is strong causal evidence that this neuron's "wait" output drives the failure.

Finished-but-wrong answers rose to 35, though, and full removal is closer to a general "doubt less" setting. So I need to check that E3 is not simply cutting thinking short:
- paired statistics against the original and against E2;
- the "Wait" rate in the first 1k tokens, which should stay near baseline;
- thinking length on problems that both the original and E3 finish;
- whether the 35 wrong answers are on problems the original also lost, or on problems it used to solve.

If early "Wait" use and thinking length on shared problems hold up, and the wrong answers mostly replace answers that were already lost, E3 repairs the spiral rather than switching off verification.

```python
# E3 analysis: paired stats, early/late Wait rates, thinking length on problems finished by both, origin of wrong finished answers.
g8 = [json.loads(l) for l in open('/work/gen_8.jsonl')]
e3_by = {r['id']: r['samples'][0] for r in g8}
c3 = np.array([e3_by[i]['correct'] for i in common], float)
for nm, base in [('vs orig', oc), ('vs E1', ec), ('vs E2', c2)]:
    dd = c3 - base
    print(f'E3 {nm}: diff {dd.mean():+.4f} SE {dd.std(ddof=1)/np.sqrt(len(dd)):.4f} | w->r {(dd>0).sum()} r->w {(dd<0).sum()}', end=' || ')
    for src in ['math', 'gsm8k']:
        sel = np.array([i.startswith(src) for i in common]); print(f'{src} {dd[sel].mean():+.3f} (SE {dd[sel].std(ddof=1)/np.sqrt(sel.sum()):.3f})', end=' ')
    print()
rng = np.random.default_rng(1)
for name, by in [('orig', o_by), ('E2', e2_by), ('E3', e3_by)]:
    allb = np.nanmean([wait_bins(by[i]['text']) for i in rng.choice(common, 120, replace=False)], axis=0)
    fin = [i for i in common if not by[i]['cut']]
    print(f'{name}: Wait/1k by bin (all answers) {np.round(allb,1).tolist()} | finished {len(fin)} acc {np.mean([by[i]["correct"] for i in fin]):.2f} median tok {np.median([by[i]["tokens"] for i in fin]):.0f}')
both_fin = [i for i in common if not o_by[i]['cut'] and not e3_by[i]['cut']]
lo = np.array([o_by[i]['tokens'] for i in both_fin]); l3 = np.array([e3_by[i]['tokens'] for i in both_fin])
print(f'finished by orig & E3: n={len(both_fin)} median tok orig {np.median(lo):.0f} E3 {np.median(l3):.0f}; E3 shorter in {np.mean(l3<lo):.2f}; acc orig {np.mean([o_by[i]["correct"] for i in both_fin]):.2f} E3 {np.mean([e3_by[i]["correct"] for i in both_fin]):.2f}')
wf3 = [i for i in common if not e3_by[i]['cut'] and not e3_by[i]['correct']]
print(f'E3 wrong finished n={len(wf3)}: orig status -> cut {sum(o_by[i]["cut"] for i in wf3)}, orig correct {sum(o_by[i]["correct"] for i in wf3)}; E2 correct {sum(e2_by[i]["correct"] for i in wf3)}')
print('E3 wrong-finished thinking tokens: median', np.median([e3_by[i]['tokens'] for i in wf3]), '| correct-finished median', np.median([e3_by[i]['tokens'] for i in common if e3_by[i]['correct']]))
print('lost vs orig (orig correct, E3 wrong):', [(i, e3_by[i]['cut'], e3_by[i]['pred']) for i in common if o_by[i]['correct'] and not e3_by[i]['correct']])
```