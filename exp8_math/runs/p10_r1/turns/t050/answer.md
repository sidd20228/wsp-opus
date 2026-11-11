The weights-only analysis is informative for early layers and uninformative for late ones:
- Early heads such as L7h1, L7h2, L11h4 and L11h7 clearly copy raw token embeddings (z of 5–7).
- The late induction heads L15h7, L19h5 and L23h5 score near zero on that probe.

That is expected. By layer 15 and later, a position's token identity is no longer carried mainly by its input-embedding direction, so this probe cannot tell whether late heads copy or suppress. Reconstructing their outputs from recorded states would amount to simulating the layer, so I'll settle the question causally instead.

The test is a teacher-forced pass with the edited model, in which the three strongest loop-specific induction heads are switched off by zeroing their columns in o_proj: L15h7, L19h5 and L23h5. These three gain the most attention on previous iterations inside the loop (+0.3 to +0.4).

The prediction, if they drive the loop: the copy margin at the loop decision points falls by more than about 2 logits, below the branching threshold, while the model's ordinary reasoning barely changes. If the margin instead rises, they were suppressors.

To measure the cost on ordinary reasoning, I'll add a third text: one of the model's own correct, finished solutions from pass 1. Now I pick that text, save the notes and prepare the edit.

```python
# Prep causal test: pick a correct finished pass-1 answer as clean control text; save notes; build ablation of heads 15.7, 19.5, 23.5 (zero o_proj columns).
g1 = [json.loads(l) for l in open('/work/gen_1.jsonl')]
print(list(g1[0].keys()))
cands = []
for r in g1:
    txt = r.get('text') or r.get('completion') or r.get('output')
    ok = r.get('correct', r.get('grade'))
    fin = not r.get('cut', r.get('truncated', False))
    if ok and fin and txt is not None:
        n = len(tok.encode(txt).ids)
        if 500 <= n <= 1100 and all(ord(c) < 128 for c in txt):
            cands.append((n, r['id'], r.get('source', '')))
print(len(cands), sorted(cands)[:12])
```