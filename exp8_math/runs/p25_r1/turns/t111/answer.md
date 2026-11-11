Junction positions are almost linearly separable at every full-attention input. A plain difference-of-means direction gives held-out AUC 0.994–0.998, and junction projections sit near 1 while other positions sit near 0. The two classes still overlap in the tails: the 99th percentile of other positions is 0.64–0.84 on that scale, and the 10th percentile of junctions 0.63–0.74. This matters because q_proj has no bias, so any gate offset I add is linear in this score and two-sided. With a large offset it would close the induction heads at junctions, but it would also open or close them noticeably at ordinary positions whose score deviates from zero. That would reintroduce collateral damage to within-line copying. Before deciding whether the gate edit is viable, I will check whether a sharper direction (shrinkage LDA, which accounts for within-class spread) concentrates the ordinary positions tightly around zero across the whole distribution, including the negative tail, and still separates junctions. If ordinary positions stay within about ±0.15 of zero while junctions sit near 1, an offset of β ≈ 6–8 closes the gates at junctions while shifting ordinary gates by less than about 1. If the spread stays near ±0.3, the edit is too blunt and I will drop it.

```python
# Can a sharper (shrinkage-LDA) direction make the junction score one-sided enough for a bias-free gate offset? Full score distributions, held-out texts.
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
for L in [15, 23]:
    gin = (1 + load_original(f'model.language_model.layers.{L}.input_layernorm.weight')).numpy()
    X, Y, T = [], [], []
    for k, ((d_, t, a, n), m) in enumerate(zip(sets, masks)):
        h = np.load(f'/work/{d_}/hidden_{L}.npy', mmap_mode='r')[t, :n].astype(np.float32)
        xh = h / np.sqrt((h**2).mean(1, keepdims=True)) * gin
        X.append(xh[60:]); Y.append(m[60:]); T.append(np.full(n-60, k))
    X = np.concatenate(X); Y = np.concatenate(Y); T = np.concatenate(T)
    tr = T < 7; te = ~tr
    for name in ['diffmean', 'lda']:
        if name == 'diffmean':
            w = X[tr & Y].mean(0) - X[tr & ~Y].mean(0)
        else:
            lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X[tr], Y[tr]); w = lda.coef_[0]
        mo = X[tr & ~Y].mean(0); w = w - (w @ mo) / (mo @ mo) * mo      # orthogonal to 'other' mean -> no bias needed on average
        pj = X @ w; sc = pj[tr & Y].mean(); s_ = pj / sc
        print(f"L{L} {name:8s} test AUC {roc_auc_score(Y[te], s_[te]):.4f} | OTHER(test) pct .1/1/10/50/90/99/99.9: {np.percentile(s_[te & ~Y], [.1,1,10,50,90,99,99.9]).round(2)} | JUNCTION(test) pct 1/10/50/90: {np.percentile(s_[te & Y], [1,10,50,90]).round(2)}")
```