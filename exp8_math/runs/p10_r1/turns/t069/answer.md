The input side is broader than "Wait" alone. Neuron 998's gate and up weights respond most strongly to the embeddings of hesitation markers: "Actually"/" actually", " hmm"/"Hmm", " hey". "Wait" tokens are aligned in the same direction but less strongly (rank 200–1,300 of 248k). So it is a hesitation-reading, "wait"-writing neuron.

In pass 1 it is more active (more negative) in cut-off answers than in finished ones (-0.36 vs -0.21), and its mean activation tracks the "Wait" rate weakly (r = -0.17). That is consistent with the earlier findings, but weak and confounded at the per-answer level.

Turning this neuron down wholesale would act like a general "doubt less" knob, which I want to avoid. The repetition-specific part is its growth. So I need to know which upstream components produce that growth.

I'll decompose the neuron's up pre-activation at text-2 line starts into exact contributions from every earlier component: the embedding, each mixer and MLP up to layer 16's own mixer. The MLP16 input is the recorded layer-17 state minus MLP16's recorded output, so this is the same linear attribution I used for the logits. I'll compare pre-loop with loop line starts and fit slopes against the number of earlier "Wait"s.

What would identify a feedback path: the growth comes mainly from the attention mixers that attend to earlier "Wait"s (layers 11 and 15) or from a specific DeltaNet layer.

```python
# Which upstream components drive n998's growth? exact linear attribution of its up/gate pre-activation at T2 line starts (input = H17 - mlp16_out).
i = 1
mo16 = np.asarray(Mact[16][i, LS], np.float64) @ Wd[16].T
x16 = np.asarray(H[17][i, LS], np.float64) - mo16
rms16 = np.sqrt((x16 ** 2).mean(1) + 1e-6)
ru_dir = (wln * Wu)[None] / rms16[:, None]; rg_dir = (wln * Wg)[None] / rms16[:, None]
compsU, compsG = {}, {}
compsU['emb'] = (np.asarray(H[0][i, LS], np.float64) * ru_dir).sum(1); compsG['emb'] = (np.asarray(H[0][i, LS], np.float64) * rg_dir).sum(1)
for l in range(16):
    delta = np.asarray(H[l + 1][i, LS], np.float64) - np.asarray(H[l][i, LS], np.float64)
    mo = np.asarray(Mact[l][i, LS], np.float64) @ Wd[l].T
    for nm, v in [(f'mix{l}', delta - mo), (f'mlp{l}', mo)]:
        compsU[nm] = (v * ru_dir).sum(1); compsG[nm] = (v * rg_dir).sum(1)
mix16 = x16 - np.asarray(H[16][i, LS], np.float64)
compsU['mix16'] = (mix16 * ru_dir).sum(1); compsG['mix16'] = (mix16 * rg_dir).sum(1)
totU = sum(compsU.values()); totG = sum(compsG.values())
print('check: corr(u, act) %.2f, corr(g, act) %.2f' % (np.corrcoef(totU, np.asarray(Mact[16][i, LS, 998]))[0, 1], np.corrcoef(totG, np.asarray(Mact[16][i, LS, 998]))[0, 1]))
print(f'u: pre-loop {totU[pre].mean():.2f} loop {totU[~pre].mean():.2f} | g: pre-loop {totG[pre].mean():.2f} loop {totG[~pre].mean():.2f}')
rowsU = sorted([(k, v[~pre].mean() - v[pre].mean(), np.polyfit(nW, v, 1)[0] * 10, v.mean()) for k, v in compsU.items()], key=lambda r: r[1])
print('up pre-act: components most decreasing in loop (loop-pre diff, slope/10 prior Waits, mean):')
for r in rowsU[:8]: print(f'   {r[0]:>6s} {r[1]:+.2f} {r[2]:+.2f} {r[3]:+.2f}')
print('   ... most increasing:', [(r[0], round(r[1], 2)) for r in rowsU[-3:]])
rowsG = sorted([(k, v[~pre].mean() - v[pre].mean(), np.polyfit(nW, v, 1)[0] * 10) for k, v in compsG.items()], key=lambda r: -abs(r[1]))
print('gate pre-act: largest loop-pre changes:', [(r[0], round(r[1], 2), round(r[2], 2)) for r in rowsG[:6]])
```