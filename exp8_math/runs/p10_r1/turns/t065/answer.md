Within text 2, the centred " Wait" logit at line starts rises with the number of earlier "Wait"s (correlation 0.60). The "high count" group came out empty because the text only reaches 21 earlier "Wait"s, so I read the slopes instead.

The growth is distributed (slopes per 10 earlier "Wait"s):

| Component | Slope | Main contributors |
|---|---|---|
| MLPs | +2.4 | MLP23 +1.7, MLP16 +1.4, MLP21 +0.6 |
| Late full-attention mixers | +1.4 | layer 15 +0.75, layer 19 +0.5 |
| DeltaNet mixers | +0.75 | |

The DeltaNet mixers and MLPs also supply most of the base level of the " Wait" logit. So projecting "Wait" out of the attention outputs alone would cut only a fraction of the feedback.

The key question is whether the MLP part of the growth is downstream of the copy heads (MLP16 sits right after layer 15, whose heads attend more and more to earlier "Wait"s) or independent of them. Pass 3 (E1) recorded the residual stream at layers 15, 19, 23 and 24, so I can compare original and E1 per block at text 2's line starts:
- layers 0–14,
- layers 15–18 (contains MLP16),
- layers 19–22,
- layer 23.

If E1 flattened the slope in the 15–18 block too, the MLP growth was copy-driven. Then a broader copy-damping edit would address the spiral. If the slope survives E1, the spiral has an independent driver.

```python
# Does E1 flatten the growth of the ' Wait' logit at T2 line starts, and in which block (0-14, 15-18, 19-22, 23)?
H5 = {l: np.load(D5 + f'hidden_{l}.npy', mmap_mode='r') for l in [15, 19, 23, 24]}
def block_contribs(Hd, i, P, u):
    h23 = np.asarray(Hd[23][i, P], np.float64); rms = np.sqrt((h23 ** 2).mean(1) + 1e-6)
    dvv = ((1 + gW)[None] * u[None]) / rms[:, None]
    b = {}
    b['0-14'] = (np.asarray(Hd[15][i, P], np.float64) * dvv).sum(1)
    b['15-18'] = ((np.asarray(Hd[19][i, P], np.float64) - np.asarray(Hd[15][i, P], np.float64)) * dvv).sum(1)
    b['19-22'] = ((h23 - np.asarray(Hd[19][i, P], np.float64)) * dvv).sum(1)
    b['23'] = (np.asarray(Hd[24][i, P], np.float64) * u[None]).sum(1) - (h23 * dvv).sum(1)
    b['final'] = (np.asarray(Hd[24][i, P], np.float64) * u[None]).sum(1)
    return b
for nm, Hd in [('orig', H), ('E1', H5)]:
    b = block_contribs(Hd, 1, LS, uW)
    print(nm, ' | '.join(f'{k}: mean {v.mean():5.2f} slope/10 {np.polyfit(nW, v, 1)[0]*10:+.2f}' for k, v in b.items()))
# same on loop-iteration line starts only vs pre-loop line starts
pre = LS < onset[1]
for nm, Hd in [('orig', H), ('E1', H5)]:
    b = block_contribs(Hd, 1, LS, uW)
    print(nm, 'pre-loop line starts: ' + ' | '.join(f'{k} {v[pre].mean():5.2f}' for k, v in b.items()), '|| loop: ' + ' | '.join(f'{k} {v[~pre].mean():5.2f}' for k, v in b.items()))
lpW0 = np.array([lp[1, t] if ids[1, t+1] in WAIT else np.nan for t in LS]); lpW1 = np.array([lp5[1, t] if ids[1, t+1] in WAIT else np.nan for t in LS])
print('logprob of actual Wait at Wait line starts: orig %.2f E1 %.2f (n=%d)' % (np.nanmean(lpW0), np.nanmean(lpW1), (~np.isnan(lpW0)).sum()))
```