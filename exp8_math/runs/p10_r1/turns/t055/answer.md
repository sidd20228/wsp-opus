The ablation causally confirms that L15h7, L19h5 and L23h5 drive the loop copying. They are copiers, not suppressors.

In the loop:
- the mean copy margin halves (8.7→4.1 and 8.1→4.7);
- in text 2, the self-reinforcing growth of the line-start margin is largely removed: by iteration 9 it reaches 2.2 instead of 4.2, so the loop stays escapable much longer;
- in text 1, the decision margin drops by about 1 logit (3.5→2.4) but stays just above the escape threshold.

Other heads compensate only slightly: L23h4's induction attention rose from 0.36 to 0.45.

The cost on ordinary reasoning is not negligible. Before the loop, the log-probability of the model's own tokens falls by 0.04–0.10 nats on average (median about 0). The top choice flips at 5–11% of positions.

Whether that cost is harmless depends on where the flips happen. Flips at positions that were already uncertain (margin under 2, sampled randomly anyway) matter little. Flips at confident positions, especially copied digits, would mean real damage such as mis-copied numbers. I'll split the flips by original confidence and by token type. The deciding question is whether to test full ablation or a partial attenuation in generation.

```python
# Check cost: are pre-loop top1 flips under ablation concentrated at already-uncertain positions (gap<2) or at confident ones, esp. digits?
digit_ids = set(t for t in range(248320) if False)
def is_digit_tok(tid):
    s = tok.decode([int(tid)]); return s.strip().isdigit()
for i in range(2):
    P = np.arange(think_pos[i] + 2, onset[i])
    g0 = np.array([gap_at(lp, tkl, tki, i, t) for t in P]); g1_ = np.array([gap_at(lp5, tkl5, tki5, i, t) for t in P])
    top0 = tki[i, P, 0]; top1_ = tki5[i, P, 0]; flip = top0 != top1_
    actual = ids[i, P + 1]
    isdig = np.array([is_digit_tok(a) for a in actual])
    print(f'text{i+1} pre-loop n={len(P)}')
    for lo, hi in [(-99, 0), (0, 1), (1, 2), (2, 4), (4, 99)]:
        sel = (g0 >= lo) & (g0 < hi)
        print(f'   orig gap [{lo},{hi}) n={sel.sum():4d} flip {flip[sel].mean():.3f}  mean dlp {(lp5[i,P]-lp[i,P])[sel].mean():+.3f}')
    print(f'   digits n={isdig.sum()}: flip {flip[isdig].mean():.3f}, mean dlp {(lp5[i,P]-lp[i,P])[isdig].mean():+.3f}, frac orig gap>2 {(g0[isdig]>2).mean():.2f} -> edited {(g1_[isdig]>2).mean():.2f}')
    conf_flip = np.where(flip & (g0 > 2))[0]
    ex = [(safe(ids[i, P[j] + 1]), safe(top1_[j]), round(float(g0[j]), 1)) for j in conf_flip[:12]]
    print('   confident flips (actual, edited top1, orig gap):', ex)
```