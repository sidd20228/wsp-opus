E1 barely slows the growth of the " Wait" logit at line starts: the slope falls from +4.9 to +4.1 logits per 10 earlier "Wait"s. The 3 removed heads are therefore not the main drivers of the doubt spiral.

The growth sits in the block of layers 15–18:
- slope +3.1 originally, +2.7 under E1;
- inside the loop this block adds about 4.7 extra logits to " Wait" (3.9 → 8.6), more than any other block.

The per-layer decomposition points to the MLPs there, especially MLP16 (+1.4 per 10 "Wait"s) and MLP15 (+0.4), with layer 15's attention mixer contributing +0.75. That fits a pathway: attention in layer 15 collects earlier "Wait"s, and the MLP right after turns that into a stronger " Wait" prediction.

To check whether a small set of neurons carries this, I'll rank MLP neurons in layers 14–23 by how much more they push " Wait" at loop line starts than at pre-loop line starts in text 2. I'll also check how much they push in text 1 and whether their activation rises with the number of earlier "Wait"s. If a few neurons account for most of the 4.7-logit loop boost, they are a concrete, testable target.

```python
# Hypothesis: a few MLP neurons (esp. L16/L15) carry the loop-specific boost of ' Wait' at line starts; rank neurons by loop-vs-preloop contribution.
i = 1
h23 = np.asarray(H[23][i, LS], np.float64); rms = np.sqrt((h23 ** 2).mean(1) + 1e-6)
dvW = ((1 + gW)[None] * uW[None]) / rms[:, None]       # [nLS,1024]
pre = LS < onset[1]
cand = []
for l in range(14, 24):
    act = np.asarray(Mact[l][i, LS], np.float64)        # [nLS,3584]
    proj = dvW @ Wd[l]                                  # [nLS,3584] per-neuron unit contribution
    c = act * proj
    dl = c[~pre].mean(0) - c[pre].mean(0)
    tot_dl = dl.sum()
    sl = np.array([np.polyfit(nW, c[:, n], 1)[0] * 10 for n in np.argsort(-dl)[:10]])
    top = np.argsort(-dl)[:10]
    print(f'L{l}: total loop-pre contribution diff {tot_dl:+.2f}; top10 sum {dl[top].sum():+.2f}; top3 {[(int(n), round(float(dl[n]),2)) for n in top[:3]]}')
    for n in top[:10]:
        cand.append((dl[n], l, int(n), c[pre, n].mean(), c[~pre, n].mean(), act[pre, n].mean(), act[~pre, n].mean()))
cand.sort(key=lambda x: -x[0])
print('\nTop 15 neurons overall: (diff, layer, idx, contrib pre, contrib loop, act pre, act loop)')
for x in cand[:15]: print(tuple(round(v, 2) if isinstance(v, float) else v for v in x))
```