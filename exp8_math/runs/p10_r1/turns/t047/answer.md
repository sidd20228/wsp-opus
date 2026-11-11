Three findings from the decomposition:

1. **Norm convention.** The final norm uses the (1 + weight) convention: with it, the normalised scale of the last state is exactly 1.
2. **Last state is post-norm.** Its dot product with the unembedding difference reproduces the observed gaps (correlation 0.9999). So the last layer can only be analysed as "final gap minus the gap at layer 23's input".
3. **Where the copy preference comes from.** In the loop, the copied token leads its runner-up by about 8.4 logits, against about 5.4 at earlier copyable positions and 3.5 at novel positions. The extra ~3 logits come almost entirely from the full-attention mixers of layers 15 and 19 (about +0.5 each) and layer 23 (the largest, about +2.3 once the last MLP is separated out). These are exactly the layers holding the induction heads L15h7, L19h5 and L23h5. DeltaNet mixers add only small amounts.
4. **The last MLP pushes back.** Layer 23's MLP pushes against the copied token, and more so in the loop (-2.2 vs -1.2 before it). The model seems to have its own repetition-suppression mechanism that loses this fight.

I used the wrong norm convention above, so I'll redo the decomposition with (1 + weight). I'll also look at three things that bear on how an edit could break the loop:
- the decision points inside each loop line, where the copy is least certain and where escapes must happen,
- whether the copy margin and the induction-head attention grow with each repetition (self-reinforcement),
- which individual layer-23 MLP neurons carry the anti-repetition push.

If the margin at decision points grows with the iteration count together with the induction heads' attention, weakening that reinforcement is the right target.

```python
# Hypothesis: copy margin at loop decision points grows with repetition count together with late induction-head attention; MLP23 holds specific anti-repetition neurons.
import re
ge = 1 + gW
res = {(i, k): decomp23(i, pos_sets[i][k], ge) for i in range(2) for k in ['loop', 'pre_copy', 'pre_nov']}
def m(i, k, f): return f(res[(i, k)]).mean()
print('corrected norm (1+w): mean gap [loop, pre_copy, pre_nov] per text')
for name, f in [('final', lambda x: x[4]), ('lens@23', lambda x: x[3]), ('mix15', lambda x: x[1][15]), ('mix19', lambda x: x[1][19]),
                ('mix20', lambda x: x[1][20]), ('mix21', lambda x: x[1][21]), ('mlp19', lambda x: x[2][19]), ('mlp22', lambda x: x[2][22]),
                ('mlp23~', lambda x: x[5]), ('L23-mlp23', lambda x: x[4] - x[3] - x[5])]:
    print(f'{name:>10s}', [round(m(i, k, f), 2) for i in range(2) for k in ['loop', 'pre_copy', 'pre_nov']])

def safe(tid):
    s = tok.decode([int(tid)])
    return s if re.fullmatch(r"[A-Za-z0-9 $\\^+=.,:;\-\n*()/{}\[\]']{0,14}", s) else f'#{int(tid)}'

attn_heads = [(15, 7), (19, 5), (23, 5), (23, 4), (23, 7), (11, 1)]
AH = {L: np.load(D + f'attn_{L}.npy', mmap_mode='r') for L in [11, 15, 19, 23]}
for i in range(2):
    P = pos_sets[i]['loop']; kmax, sstar, allm = MI[i]
    per = int(np.median([t - sstar[t] for t in P]))
    gaps = res[(i, 'loop')][4]
    offs = (P - P[0]) % per
    mg = np.array([gaps[offs == o].mean() for o in range(per)])
    o_min = int(mg.argmin())
    print(f'\ntext{i+1}: period {per} tokens; mean gap by offset:', mg.round(1).tolist())
    tt = P[offs == o_min][0]
    print(f'  decision offset {o_min}: next actual {safe(ids[i, tt+1])!r}; top5', [(safe(x), round(float(y), 2)) for x, y in zip(tki[i, tt, :5], tkl[i, tt, :5])])
    dpos = P[offs == o_min]
    k_iter = np.array([len(allm[t]) for t in dpos])  # number of earlier matches ~ iteration count
    print('  decision-point gap vs #earlier matches:', [(int(k), round(float(g), 1)) for k, g in list(zip(k_iter, gaps[offs == o_min]))[::max(1, len(dpos)//8)]])
    for (L, h) in attn_heads:
        Ai = np.asarray(AH[L][i, h])
        mass = [Ai[t, allm[t] + 1].sum() for t in dpos]
        print(f'  head {L}.{h} ind-mass at decision pts, first/last thirds: {np.mean(mass[:len(mass)//3]):.2f} -> {np.mean(mass[-len(mass)//3:]):.2f}')
```