Zeroing the four induction heads made things worse. Accuracy fell to 6.1% from 10.9% on the same seeds, and the cut-off share went up slightly, to 92.4% from 90.6%. So full ablation does not free the model to finish. Either it still gets stuck (in non-verbatim or shorter loops, or loops driven by the remaining copy heads), or the damage to general copying outweighs any loop relief.

Before deciding where to go next, I need to see how the edited model's cut-off texts end. If they still end in verbatim loops, just with different or shorter periods, the loop mechanism is spread across more heads than these four. If they mostly end in non-repeating rambling, verbatim loops were a symptom and the non-commitment problem dominates.

I'll reuse the loop-detection helper from the earlier analysis if it is still in the kernel. I'll then compare loop frequency, onset and period between `gen_2` and `gen_7`, together with the paired per-sample transitions.

```python
# Diagnose pass 4: do edited cut samples still end in verbatim loops (other heads take over) or in non-repeating rambling?
fns = [k for k, v in globals().items() if callable(v) and type(v).__name__ == 'function' and not k.startswith('_')]
print('functions in kernel:', fns)
G7 = [json.loads(l) for l in open('/work/gen_7.jsonl')]
print(len(G7), list(G7[0].keys()), list(G7[0]['samples'][0].keys()))
# paired transitions vs gen_2
g2map = {(g_['id'], i): s_ for g_ in G2 for i, s_ in enumerate(g_['samples'])}
w2r = r2w = both_r = 0
for g_ in G7:
    for i, s_ in enumerate(g_['samples']):
        o_ = g2map[(g_['id'], i)]
        w2r += (not o_['correct']) and s_['correct']; r2w += o_['correct'] and (not s_['correct']); both_r += o_['correct'] and s_['correct']
print('paired: both right %d, wrong->right %d, right->wrong %d' % (both_r, w2r, r2w))
```