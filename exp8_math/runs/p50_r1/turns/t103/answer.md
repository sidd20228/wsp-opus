This is a clean split with two mechanisms that change together in loops. On ordinary text, the layer-23 MLP trims the margin of the predicted token by about 1.6 nats (2.0 at a loop's first cycle). In late loop cycles that trimming disappears entirely (+0.01). At the same time, layer-23 attention's copy push rises from about 1.0 to 4.9 nats. So the network has a built-in final-layer brake on over-confident predictions, and the loop state switches it off. Losing it is worth about 2 nats of the loop's extra confidence, and the attention amplification about 4. A failing native brake is exactly the kind of component worth repairing, as with n573. I will find which layer-23 MLP neurons supply this damping on normal text and how their activity changes in loops. If a small set carries most of it and they go silent in late cycles, I can then examine what silences them, for example whether their input weights read the repetition direction negatively.

```python
# Hypothesis: a few MLP23 neurons carry the normal-text copy damping (-1.6 nats) and go silent in loops.
per = np.zeros((3, 3584)); actm = np.zeros((3, 3584))
for t in range(2):
    T_ = lens5[t] - 1; X_ = np.asarray(H23[t, :T_], np.float32); s23 = np.sqrt((X_**2).mean(1) + 1e-6)
    D_ = Dd5[t] * wf1 / s23[:, None]; Aa = np.asarray(A23m[t, :T_], np.float32); C_ = Aa * (D_ @ Wd23_)
    m0, m3 = cyc_masks(t); pre = np.zeros(T_, bool); pre[100:on5[t]-1] = True
    for j, m in enumerate([pre, m0, m3]): per[j] += C_[m].mean(0) / 2; actm[j] += Aa[m].mean(0) / 2
o = np.argsort(per[0])
print('sum of 20 most damping neurons (pre):', round(per[0, o[:20]].sum(), 2), 'of total', round(per[0].sum(), 2), '| same neurons late:', round(per[2, o[:20]].sum(), 2))
print(pd.DataFrame({'neuron': o[:12], 'contrib_pre': per[0, o[:12]], 'contrib_c0': per[1, o[:12]], 'contrib_late': per[2, o[:12]], 'act_pre': actm[0, o[:12]], 'act_late': actm[2, o[:12]]}).round(3).to_string())
```