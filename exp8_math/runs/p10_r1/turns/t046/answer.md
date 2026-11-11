The decomposition did not add up exactly (correlation 0.99, but means of 3.9 or 5.0 against an observed 4.6). The final-norm weights average 3.3, which is not the zero-centred form I tested.

The likely reason is a standard Hugging Face convention: the last returned hidden state, layer 24, is taken after the final norm, not before it. If so:
- the logits are simply that state times the unembedding,
- the raw final residual (and therefore layer 23's attention output) cannot be recovered exactly.

I'll check this directly. If the last state is post-norm, its normalised scale should be about 1 for the right weight convention, and its dot product with the unembedding difference should reproduce the observed gaps.

Then I'll decompose the logit-lens gap measured at the input of the last layer exactly into the embedding plus the mixer and MLP outputs of layers 0–22. Layer 23's contribution is the final gap minus that. This shows which layers build the copy preference specifically in the loop, compared with copyable and novel positions before it.

```python
# Hypothesis: hidden_24 is post-final-norm (HF convention); then decompose the logit-lens gap at the last layer's input into per-layer mixer/MLP parts, loop vs pre-loop.
for nm, ge in [('w', gW), ('1+w', 1 + gW)]:
    u = np.asarray(H[24][0, 500:520], np.float64) / ge
    print(nm, 'mean((H24/g)^2) =', np.round((u ** 2).mean(1)[:5], 3))
P = pos_sets[0]['loop'][:50]
a = ids[0, P + 1]; r = np.array([next(x for x in tki[0, t] if x != ids[0, t + 1]) for t in P])
g_fin = (np.asarray(H[24][0, P], np.float64) * (E[a] - E[r])).sum(1)
obs = np.array([lp[0, t] - tkl[0, t][list(tki[0, t]).index(rr)] for t, rr in zip(P, r)])
print('post-norm check: corr %.4f mean final gap %.3f obs %.3f' % (np.corrcoef(g_fin, obs)[0, 1], g_fin.mean(), obs.mean()))

FA = {3, 7, 11, 15, 19, 23}
def decomp23(i, P, geff):
    a = ids[i, P + 1]; r = np.array([next(x for x in tki[i, t] if x != ids[i, t + 1]) for t in P])
    dU = E[a] - E[r]
    h23 = np.asarray(H[23][i, P], np.float64); rms = np.sqrt((h23 ** 2).mean(1) + 1e-6)
    dv = geff[None] * dU / rms[:, None]
    emb = (np.asarray(H[0][i, P], np.float64) * dv).sum(1)
    mix, mlp = np.zeros((23, len(P))), np.zeros((23, len(P)))
    for l in range(23):
        delta = np.asarray(H[l + 1][i, P], np.float64) - np.asarray(H[l][i, P], np.float64)
        mo = np.asarray(Mact[l][i, P], np.float64) @ Wd[l].T
        mix[l] = ((delta - mo) * dv).sum(1); mlp[l] = (mo * dv).sum(1)
    G23 = (h23 * dv).sum(1)
    Gfin = (np.asarray(H[24][i, P], np.float64) * dU).sum(1)
    mo23 = np.asarray(Mact[23][i, P], np.float64) @ Wd[23].T
    mlp23 = (mo23 * dv).sum(1)
    return emb, mix, mlp, G23, Gfin, mlp23

ge = gW  # convention chosen from the check above (printed); recomputed below if needed
res = {}
for i in range(2):
    for key in ['loop', 'pre_copy', 'pre_nov']:
        res[(i, key)] = decomp23(i, pos_sets[i][key], ge)
print('\nmean logit gap (actual - runner-up) decomposition; columns: text1 loop, pre_copy, pre_nov | text2 loop, pre_copy, pre_nov')
def row(name, f):
    vals = [f(res[(i, k)]).mean() for i in range(2) for k in ['loop', 'pre_copy', 'pre_nov']]
    print(f'{name:>10s} ' + ' '.join(f'{v:7.2f}' for v in vals[:3]) + ' |' + ' '.join(f'{v:7.2f}' for v in vals[3:]))
row('final', lambda x: x[4]); row('lens@23', lambda x: x[3]); row('emb', lambda x: x[0])
for l in range(23):
    row(f'mix{l}{"*" if l in FA else ""}', lambda x, l=l: x[1][l])
for l in range(23):
    row(f'mlp{l}', lambda x, l=l: x[2][l])
row('mlp23~', lambda x: x[5]); row('L23 eff', lambda x: x[4] - x[3])
```