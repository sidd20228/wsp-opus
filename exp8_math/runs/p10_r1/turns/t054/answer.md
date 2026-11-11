The edited pass ran. The overall mean token log-probability fell from -0.297 to -0.377 on text 1 and from -0.379 to -0.471 on text 2, so removing the three heads changes the model noticeably. What matters is where that change falls. I'll compare original and edited at the same positions:
- the loop region (overall, and at the decision points, measuring the margin to the best alternative),
- ordinary pre-loop positions, both copyable and novel: mean log-probability change and how often the top choice changes,
- the attention of the remaining heads in layers 19 and 23, to check whether backup heads took over the copying.

The hypothesis is confirmed if the decision-point margins drop below about 2 logits with comparatively small costs before the loop. It is undermined if the cost is spread evenly, or if the loop margins hold up because other heads compensate.

```python
# Test: ablating 15.7/19.5/23.5 lowers loop decision gaps below ~2 while ordinary positions change little; check compensation by other heads.
D5 = '/work/fwd_5/'
ids5 = np.load(D5 + 'input_ids.npy'); print('ids same:', (ids5 == ids).all())
lp5 = np.load(D5 + 'token_logprobs.npy'); tkl5 = np.load(D5 + 'topk_logprobs.npy'); tki5 = np.load(D5 + 'topk_ids.npy')
def gap_at(lpa, tklx, tkix, i, t):
    a = ids[i, t + 1]
    alt = [(x, y) for x, y in zip(tkix[i, t], tklx[i, t]) if x != a]
    return lpa[i, t] - alt[0][1]
def top1(tkix, i, t): return tkix[i, t, 0]
for i in range(2):
    print(f'--- text{i+1}')
    for key in ['loop', 'pre_copy', 'pre_nov']:
        P = pos_sets[i][key]
        d = lp5[i, P] - lp[i, P]
        g0 = np.array([gap_at(lp, tkl, tki, i, t) for t in P]); g1_ = np.array([gap_at(lp5, tkl5, tki5, i, t) for t in P])
        flip = np.mean([top1(tki, i, t) != top1(tki5, i, t) for t in P])
        det0 = (g0 > 2).mean(); det1 = (g1_ > 2).mean()
        print(f'{key:8s} n={len(P):4d} mean lp {lp[i,P].mean():.3f}->{lp5[i,P].mean():.3f} (d={d.mean():+.3f}, median {np.median(d):+.3f}) | mean gap {g0.mean():.2f}->{g1_.mean():.2f} | top1 flips {flip:.3f} | frac gap>2: {det0:.2f}->{det1:.2f}')
    dp = dp_positions(i)
    g0 = np.array([gap_at(lp, tkl, tki, i, t) for t in dp]); g1_ = np.array([gap_at(lp5, tkl5, tki5, i, t) for t in dp])
    print('decision-pt gaps orig  :', g0.round(1).tolist())
    print('decision-pt gaps edited:', g1_.round(1).tolist())
# compensation: induction mass of remaining heads in 19, 23 (orig vs edited), loop positions
for L in [19, 23]:
    A0 = np.load(D + f'attn_{L}.npy', mmap_mode='r'); A1 = np.load(D5 + f'attn_{L}.npy', mmap_mode='r')
    out = []
    for h in range(8):
        v = []
        for i in range(2):
            P = pos_sets[i]['loop']; allm = MI[i][2]
            a0 = np.asarray(A0[i, h]); a1 = np.asarray(A1[i, h])
            v.append((np.mean([a0[t, allm[t] + 1].sum() for t in P]), np.mean([a1[t, allm[t] + 1].sum() for t in P])))
        out.append(f'{L}.{h}: ' + ' '.join(f'{x:.2f}->{y:.2f}' for x, y in v))
    print(' | '.join(out))
```