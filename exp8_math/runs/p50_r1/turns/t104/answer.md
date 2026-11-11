The damping is spread over many neurons rather than concentrated. The 20 strongest damping neurons supply -1.46 of the -1.57 nats on ordinary text and still supply -1.08 in late loop cycles. So the MLP's switch to net zero in loops comes partly from those neurons weakening and partly from other neurons pushing the copy up. No single "switched-off" neuron stands out as a repair target. Both final-layer changes in loops (attention copying more, MLP damping less) could, however, share a common cause: the repetition direction in the residual, which already pushes the H4/H5 gates open. Before considering an edit that makes layer 23 insensitive to that direction, I must rule out a confound. Late loop cycles are also hundreds of tokens later in the text, so the "repetition" direction might partly encode position or length. I will project the gen_2 per-sample mean residuals (same 8,192-token averaging window) onto the direction and compare early-looping samples with samples that loop only after 10k tokens, whose first 8k tokens are ordinary reasoning. This compares samples of equal window length that differ in loop content. I will also check whether the projection drifts with position inside pre-loop text.

```python
# Hypothesis: r_hat is a loop signature, not a position/length feature: early-loop >> late-loop samples at equal 8192-token windows; no pre-loop drift.
rh = {l: (REP[l][0]/np.linalg.norm(REP[l][0]) + REP[l][1]/np.linalg.norm(REP[l][1])) for l in [16, 20, 22, 23]}
rh = {l: v / np.linalg.norm(v) for l, v in rh.items()}
L2k = L2.set_index(['id', 'si'])
grp = []
for i, s in enumerate(idx['samples']):
    r = L2k.loc[(s['item'], s['sample'])]
    g_ = 'early' if (r['looped'] and r['cut'] and r['onset'] < 3000) else ('late' if (r['looped'] and r['cut'] and r['onset'] > 10000) else ('fin' if not r['cut'] else 'mid'))
    grp.append(g_)
grp = np.array(grp); print({g: int((grp == g).sum()) for g in ['early', 'mid', 'late', 'fin']})
for l in [16, 20, 22, 23]:
    pr = np.asarray(HM[:, l], np.float32) @ rh[l]
    d_ = (pr[grp == 'early'].mean() - pr[grp == 'late'].mean()) / pr[grp != 'mid'].std()
    print(f"hidden_{l}: proj mean early {pr[grp=='early'].mean():.2f} mid {pr[grp=='mid'].mean():.2f} late {pr[grp=='late'].mean():.2f} fin {pr[grp=='fin'].mean():.2f} | d'(early-late) {d_:.2f}")
for t in range(2):
    X_ = np.asarray(np.load('/work/fwd_5/hidden_22.npy', mmap_mode='r')[t, 100:on5[t]], np.float32) @ rh[22]
    print(f'text {t} pre-loop drift: corr(proj, position) = {np.corrcoef(X_, np.arange(len(X_)))[0,1]:.2f}, proj first/last 200: {X_[:200].mean():.2f} {X_[-200:].mean():.2f} | late-loop mean {np.asarray(np.load("/work/fwd_5/hidden_22.npy", mmap_mode="r")[t, on5[t]+3*per5[t]:lens5[t]], np.float32).dot(rh[22]).mean():.2f}')
```