The linear estimate says that strengthening the brake works, but only moderately. With β = 2, the probability of a doubt opener at high-doubt line starts falls from 0.80 to 0.57 (all line starts: 0.26 → 0.18). With β = 3 it falls to 0.50 (0.16). Low-doubt lines are untouched (0.008 → 0.004), and P(`</think>`) at the five closing junctions is unchanged (0.72/0.44/0.15/0.07/0.04). The effect is partial because n573 suppresses mainly "Wait" and "Hmm"; other doubt openers ("Is", "Could", "Maybe", "Let") take some of the freed probability. Sampling at temperature 0.6 will amplify these log-odds shifts by about 1.7x.

Before a generation test I want to check collateral effects: where else n573 is strongly active, and whether boosting it would lower the probability of the tokens the model actually wrote outside doubt openers. A sizable predicted log-likelihood loss on the healthy texts (T3-T5) would argue for a smaller β.

```python
# Collateral check for n573 x beta: where is n573 strongly active, and predicted change in log p(actual next token) on non-'Wait' tokens.
acts = []; 
for t in range(7):
    n = lens6[t]
    for b in range(60, n - 1):
        acts.append((t, b, float(A22[t, b, 573]), int(ids6[t, b + 1])))
AC = pd.DataFrame(acts, columns=['t', 'b', 'a573', 'nxt'])
print('n573 act quantiles', AC.a573.quantile([.01, .05, .5, .95, .99]).round(2).tolist(), '| share |a|>2: %.3f' % (AC.a573.abs() > 2).mean())
strong = AC[AC.a573 < -2]; print('next tokens at strong positions:', Counter(tok.decode([x]).strip() for x in strong.nxt).most_common(8))
def lp_change(t, b, beta):
    lp = tkl[t, b].astype(np.float64); ids_k = tki[t, b]; nxt = int(ids6[t, b + 1])
    d_all = (beta - 1) * float(A22[t, b, 573]) / rms_final(t, b)
    d = d_all * ue573[ids_k]; p_other = max(1e-6, 1 - np.exp(lp).sum())
    lg = np.concatenate([lp + d, [np.log(p_other)]]); Z = np.logaddexp.reduce(lg)
    new = (float(tlp[t, b]) + d_all * ue573[nxt]) - Z
    return new - float(tlp[t, b])
for beta in [2.0, 3.0]:
    sel = AC[(AC.t >= 3) & (AC.t <= 5) & (~AC.nxt.isin([13428, 13784]))]
    ch = np.array([lp_change(t, b, beta) for t, b in zip(sel.t, sel.b)])
    print(f'beta {beta}: healthy texts non-Wait tokens: mean dlogp {ch.mean():+.4f}, 1% worst {np.quantile(ch, .01):+.3f}, n {len(ch)}')
```