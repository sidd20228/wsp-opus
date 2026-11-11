At the line-start points where text 2 chooses " Wait", attention to earlier " Wait" tokens grows with each repetition in almost every late head, not only in the three I removed:
- Layer 19: h0 0.22→0.65, h6 0.31→0.68, h7 0.15→0.59.
- Layer 23: h0 0.27→0.75, h7 0.33→0.59, h1 0.25→0.51.
- Also L15h5, L15h7 and L11h1.

After E1 these attention patterns barely change. So line-start copying of "Wait" is spread over many heads. That explains why removing three heads didn't stop the "Wait" spiral, and why removing more whole heads would cost too much general copying.

This suggests a token-level positive feedback loop. Each line opening that copies an earlier line opening makes "Wait" more frequent in context, which makes the next copied opening more likely to be "Wait". That fits the gradual climb of the "Wait" rate in generation.

A more targeted repair would cut only the copy path for "Wait" in these heads rather than whole heads. Before building it, I'll check it with an exact decomposition of the " Wait" logit (relative to the vocabulary mean) at every line start of text 2. That tells me which components' contributions grow with the number of earlier "Wait"s in context. If the late attention mixers' direct push toward " Wait" grows with that count while MLPs and DeltaNet layers stay flat, removing the " Wait" output direction from those heads is justified.

```python
# Hypothesis: the direct push toward ' Wait' at line starts from late attention mixers grows with the number of earlier 'Wait's (copy feedback); MLP/DeltaNet parts stay flat.
WAIT = [13428, 13784]
def line_start_positions(i, lo, hi):
    out = []
    for n in range(lo, hi - 1):
        if '\n' in tok.decode([int(ids[i, n])]):
            f = n + 1
            while f < hi and re.fullmatch(r'[\s*\-#>0-9.]*', tok.decode([int(ids[i, f])])) is not None: f += 1
            if f < hi and f - 1 not in out: out.append(f - 1)
    return np.array(out)
i = 1
LS = line_start_positions(i, think_pos[i] + 2, T - 1)
nW = np.array([np.isin(ids[i, :t + 1], WAIT).sum() for t in LS])
isW = np.isin(ids[i, LS + 1], WAIT)
uW = E[13428].astype(np.float64) - E.mean(0)
h23 = np.asarray(H[23][i, LS], np.float64); rms = np.sqrt((h23 ** 2).mean(1) + 1e-6)
dv = ((1 + gW)[None] * uW[None]) / rms[:, None]
comp = {}
comp['emb'] = (np.asarray(H[0][i, LS], np.float64) * dv).sum(1)
for l in range(23):
    delta = np.asarray(H[l + 1][i, LS], np.float64) - np.asarray(H[l][i, LS], np.float64)
    mo = np.asarray(Mact[l][i, LS], np.float64) @ Wd[l].T
    comp[f'mix{l}'] = ((delta - mo) * dv).sum(1); comp[f'mlp{l}'] = (mo * dv).sum(1)
fin = (np.asarray(H[24][i, LS], np.float64) * ((1) * uW[None])).sum(1)  # final centered Wait logit
lens23 = (h23 * dv).sum(1)
mo23 = np.asarray(Mact[23][i, LS], np.float64) @ Wd[23].T
comp['mlp23'] = (mo23 * dv).sum(1); comp['mix23~'] = fin - lens23 - comp['mlp23']
print(f'T2 line starts n={len(LS)}, next token Wait in {isW.sum()}; prior-Wait count range {nW.min()}-{nW.max()}')
print('final centered Wait logit: corr with prior-Wait count %.2f; mean low-count(<=8) %.2f high-count(>=25) %.2f' % (np.corrcoef(fin, nW)[0,1], fin[nW<=8].mean(), fin[nW>=25].mean()))
rows = []
for k, v in comp.items():
    sl = np.polyfit(nW, v, 1)[0]
    rows.append((k, v.mean(), v[nW <= 8].mean(), v[nW >= 25].mean(), sl * 10))
rows.sort(key=lambda r: -abs(r[4]))
print('component: mean | low-count | high-count | slope per 10 prior Waits (top 14 by |slope|)')
for r in rows[:14]: print(f'{r[0]:>8s} {r[1]:6.2f} {r[2]:6.2f} {r[3]:6.2f} {r[4]:+6.2f}')
grp = lambda keys: sum(comp[k] for k in keys)
FAm = ['mix15', 'mix19', 'mix23~']; DNm = [f'mix{l}' for l in range(23) if l not in FA]; MLm = [f'mlp{l}' for l in range(24)]
for nm, keys in [('late FA mixers', FAm), ('early FA mixers', ['mix3', 'mix7', 'mix11']), ('DeltaNet mixers', DNm), ('all MLPs', MLm)]:
    v = grp(keys); print(f'{nm:>16s}: low {v[nW<=8].mean():6.2f} high {v[nW>=25].mean():6.2f} slope/10 {np.polyfit(nW, v, 1)[0]*10:+.2f}')
```