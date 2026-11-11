The doubt-spiral repair is a strong result on these 512 problems:

| Comparison | Accuracy change | SE | wrong→right | right→wrong |
|---|---|---|---|---|
| E2 vs original | +9.2 points | 1.8 | 67 | 20 |
| E2 vs E1 (adds neuron edit) | +6.6 points | 1.8 | 60 | 26 |

Both MATH and GSM8K improve. Mechanistically:
- the "Wait" rate in cut-off answers is unchanged in the first 1k tokens (3.0 per 1k) but grows much more slowly afterwards (4–8k tokens: 12.6 vs 20.2);
- loops start later (median 9.1k tokens);
- 120 answers finish instead of 55, and they are longer, not shorter (median 13.2k tokens vs 5.5k). The model is not closing its thinking early; it gets through longer reasoning without spiralling.

One prediction failed. The neuron's own mean activation did not drop; it is marginally more negative (-0.367 vs -0.347). So the hesitation state still builds up, and what the edit removes is its conversion into a self-feeding "Wait" stream. A possible explanation is that the model expresses hesitation through other words the neuron reads ("Actually", "Hmm").

I'll check that substitution, and test whether the model shortens thinking on problems both versions finish. I'll also look at the wrong finished answers before choosing the next pass: either a stronger dose of the neuron edit, or confirmation on fresh problems.

```python
# Check E2 side effects: hesitation-word substitution, paired thinking length on problems finished by both, wrong finished answers.
pats = {'Wait': r'\bWait\b', 'wait': r'\bwait\b', 'Actually': r'\bActually\b', 'Hmm': r'\bHmm\b', 'But': r'\bBut\b', 'double-check': r'double-check', 'Alternatively': r'\bAlternatively\b', 'Okay': r'\bOkay\b'}
for name, by in [('orig', o_by), ('E1', e_by), ('E2', e2_by)]:
    rates = {}
    for k, p in pats.items():
        v = []
        for i in common:
            t = by[i]['text'][:30000]; n = len(tok.encode(t).ids)
            v.append(1000 * len(re.findall(p, t)) / n)
        rates[k] = np.mean(v)
    print(name, {k: round(v, 2) for k, v in rates.items()})
both_fin = [i for i in common if not o_by[i]['cut'] and not e2_by[i]['cut']]
lo = np.array([o_by[i]['tokens'] for i in both_fin]); l2 = np.array([e2_by[i]['tokens'] for i in both_fin])
print(f'finished by both: n={len(both_fin)} median tokens orig {np.median(lo):.0f} E2 {np.median(l2):.0f}; E2 shorter in {np.mean(l2<lo):.2f}; acc orig {np.mean([o_by[i]["correct"] for i in both_fin]):.2f} E2 {np.mean([e2_by[i]["correct"] for i in both_fin]):.2f}')
wf = [(i, e2_by[i]['pred'], next(r for r in g7 if r['id'] == i)['answer'], e2_by[i]['tokens'], o_by[i]['cut']) for i in common if not e2_by[i]['cut'] and not e2_by[i]['correct']]
print('E2 wrong finished (id, pred, ref, tokens, orig_cut):'); [print('  ', w) for w in wf]
```