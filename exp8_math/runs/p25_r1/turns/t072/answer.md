The scan shows that doubt openers are voiced by many token-specific output neurons, not one pair. There are 214 neurons in layers 10-23 whose output column is extremely specific (|z| > 8) for one of these words. Examples: L17 n1915 ("Wait" +13), L17 n886 ("Hold" +14), L17 n1933 ("Maybe" +14), L16 n3439 and L17 n3547 (" But" +14 to +16), L16 n1207 ("Is" +9.5). There are also commit-word writers (L17 n838 "Final" +19.7, L16 n508 "Done" +10). Editing one "Wait" brake would therefore only shift the doubt into other openers.

The shared upstream cause is the doubt state, which is linearly readable from layer 12 on and is built diffusely over layers 10-16 (n998's input drivers were spread over those layers). The appropriate intervention is a directional one. I will estimate the doubt direction within texts: the average over texts of mean(high-doubt) - mean(low-doubt) line-start states, which removes text identity. I will check its held-out separation, its agreement with the independent averaged-regression direction, and its output effect on doubt versus commit openers. Good held-out separation together with positive agreement would justify a generation test of projecting this direction out of the residual writes.

```python
# Within-text doubt direction r at mid layers: held-out AUC, agreement with averaged-regression dir, logit-lens on doubt vs commit openers.
from sklearn.metrics import roc_auc_score
lab = np.where(P_.pdoubt > 0.5, 1, np.where(P_.pdoubt < 0.05, 0, -1)); keep = lab >= 0
def within_dir(l, texts_used):
    vs = []
    for t in texts_used:
        m_ = (P_.t.values == t) & keep
        hi_ = m_ & (lab == 1); lo_ = m_ & (lab == 0)
        if hi_.sum() >= 3 and lo_.sum() >= 3:
            X_ = np.stack([Hs[l][tt, b] for tt, b in zip(P_.t[hi_], P_.pos[hi_])]).mean(0) - np.stack([Hs[l][tt, b] for tt, b in zip(P_.t[lo_], P_.pos[lo_])]).mean(0)
            vs.append(X_)
    v = np.mean(vs, 0); return v / np.linalg.norm(v), len(vs)
for l in [10, 12, 14, 16, 18]:
    aucs = []
    for t in range(7):
        r_, k_ = within_dir(l, [x for x in range(7) if x != t]); m_ = (P_.t.values == t) & keep
        if len(set(lab[m_])) == 2:
            proj = np.stack([Hs[l][tt, b] for tt, b in zip(P_.t[m_], P_.pos[m_])]) @ r_; aucs.append(roc_auc_score(lab[m_], proj))
    r_all, k_ = within_dir(l, range(7))
    ll = WU[[13428, 13784] + [targets[w] for w in [' But', ' Is', ' Could', ' Maybe', ' Hmm', ' Let']]] @ (scale * r_all); lc = WU[[targets[w] for w in [' Okay', ' So', ' Final', ' Done', ' I']] + [248069]] @ (scale * r_all)
    print(f'L{l}: texts used {k_}, LOTO AUC {np.round(aucs, 2).tolist()} mean {np.mean(aucs):.2f} | cos avg-regr wait {float(r_all @ dw[l]):.2f} loop {float(r_all @ dl[l]):.2f} | lens doubt {ll.mean():+.2f} commit {lc.mean():+.2f}')
```