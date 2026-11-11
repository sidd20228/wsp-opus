Near-perfect held-out separation (AUC 0.97-1.00) already at layer 10 is suspicious: a genuine "doubt" decision should not be that clean so early. The high and low classes may differ in surface form, for example prose bullet lines (which can open with "Wait") against math lines starting with "$" (which never do). A direction trained on that contrast would encode line type rather than doubt, and ablating it could damage formatting.

I will check the composition of the two classes: the current token at the decision position and the word that actually follows. If they differ systematically in line type, I will rebuild the contrast within comparable prose lines only: same position type, low doubt meaning a commit/progress opener follows.

```python
# Confound check: what token sits at the decision position and what word follows, for high- vs low-doubt points?
P_['cur'] = [tok.decode([int(ids6[t, b])]) for t, b in zip(P_.t, P_.pos)]
for nm, sel in [('hi', P_[P_.pdoubt > 0.5]), ('lo', P_[P_.pdoubt < 0.05]), ('mid', P_[(P_.pdoubt >= 0.05) & (P_.pdoubt <= 0.5)])]:
    print(nm, len(sel), '| cur tok:', Counter(repr(c) for c in sel.cur).most_common(5), '| next word:', Counter(sel.word).most_common(8))
```