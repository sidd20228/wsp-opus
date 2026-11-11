The direction was confounded, as I feared. The low-doubt class is dominated by math lines (36 of 131 are followed by "$", plus digits, "No", "<think>"), while the high-doubt class is prose bullet lines. A "doubt direction" from that contrast partly encodes "prose line vs math line", and ablating it could disrupt formatting. Also, the high-doubt points that do not continue with "Wait" continue with other doubt openers ("Is", "Could", "What"). So suppressing "Wait" alone (the n573 brake) would likely just change the opener.

To get a clean doubt contrast, I will restrict to prose line starts: positions where most of the predicted mass (>70%) is on alphabetic word openers. Among these, I compare lines where doubt words take most of the word mass with lines where commit/progress words dominate. If enough such positions exist, I will re-estimate the within-text direction there and check that it still separates held-out texts. If there are too few, I will need more teacher-forced prose junctions before any direction edit.

```python
# Cleaner contrast: prose-only line starts (word openers dominate); doubt share among word openers; counts and within-text direction quality.
def word_mass(t, b):
    pw = pdub = 0.0
    for j in range(20):
        s_ = tok.decode([int(tki[t, b, j])]).strip(); p = float(np.exp(tkl[t, b, j]))
        if re.fullmatch(r"[A-Za-z][A-Za-z']*", s_): pw += p; pdub += p * (int(tki[t, b, j]) in doubt_ids)
    return pw, pdub
wm = [word_mass(t, b) for t, b in zip(P_.t, P_.pos)]; P_['pw'] = [a for a, _ in wm]; P_['dshare'] = [d / max(a, 1e-9) for a, d in wm]
prose = P_[P_.pw > 0.7]
print('prose points', len(prose), '| dshare quantiles', prose.dshare.quantile([.1, .25, .5, .75, .9]).round(2).tolist())
print('per text (n, hi>0.6, lo<0.2):', {t: (int((prose.t == t).sum()), int(((prose.t == t) & (prose.dshare > .6)).sum()), int(((prose.t == t) & (prose.dshare < .2)).sum())) for t in range(7)})
print('lo examples next words:', Counter(prose[prose.dshare < .2].word).most_common(10))
```