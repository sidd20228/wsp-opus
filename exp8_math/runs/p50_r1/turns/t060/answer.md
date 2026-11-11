The total logit difference came out negative (−2.0 inside the loop, −7.4 before it), even though P(Wait) ≈ 0.97 inside the loop. So the target token is wrong. My attribution direction always used " Wait" (13428), but many of these positions, especially in text B where lines begin with "Wait" at the line start, actually emit "Wait" without a space (13784). At such positions " Wait" itself is unlikely, which produced the meaningless negative totals.

I'll redo the attribution with the correct direction at each position: the embedding of the token actually emitted, minus the probability-weighted competitors. I'll rebuild the per-layer and per-neuron tables with the scale correction. As a sanity check, the total should now be clearly positive, about +3 nats or more inside the loop, and the layer contributions should add up to it.

```python
# Redo DLA with the ACTUAL next token (' Wait' 13428 or 'Wait' 13784) as target; per-layer (nats) and per-neuron contributions.
def udir2(i, t):
    nxt = int(ids5[i, t + 1])
    alts = [(int(tk_ids[i, t, j]), np.exp(tk_lp[i, t, j])) for j in range(20) if int(tk_ids[i, t, j]) not in WAIT]
    w_ = np.array([p for _, p in alts]); w_ /= w_.sum()
    ua = (Ew[[a for a, _ in alts]].numpy() * w_[:, None]).sum(0)
    return Ew[nxt].numpy() - ua
D_ = np.stack([udir2(i, t) for i, t, _ in POS])               # (P,1024) raw token-difference direction
U2 = gfin[None] * D_
total2 = (D_ * h24s).sum(1)                                   # exact logit diff (logits = E h24)
cnt = {13428: 0, 13784: 0}
for i, t, _ in POS: cnt[int(ids5[i, t + 1])] += 1
print('target counts', cnt, '| total logit diff loop %.2f pre %.2f' % (total2[inl].mean(), total2[~inl].mean()))
mix2 = np.zeros((24, P_)); mlp2 = np.zeros((24, P_)); NEU2 = {}
for l in range(24):
    Wd = load(f'model.language_model.layers.{l}.mlp.down_proj.weight').numpy()
    A_ = np.load(f'/work/fwd_5/mlp_act_{l}.npy', mmap_mode='r')
    acts = np.stack([np.asarray(A_[i, t], dtype=np.float32) for i, t, _ in POS])
    contrib = acts * ((U2 @ Wd) / s_est[:, None])
    NEU2[l] = contrib; mlp2[l] = contrib.sum(1)
    if l < 23:
        d_ = np.stack([Hs[l + 1][i, t] - Hs[l][i, t] for i, t, _ in POS])
        mix2[l] = (d_ * U2).sum(1) / s_est - mlp2[l]
emb2 = (np.stack([Hs[0][i, t] for i, t, _ in POS]) * U2).sum(1) / s_est
mix2[23] = total2 - (emb2 + mix2[:23].sum(0) + mlp2.sum(0))
print('embedding %.2f / %.2f' % (emb2[inl].mean(), emb2[~inl].mean()))
print('layer | mixer(loop) mlp(loop) | mixer(pre) mlp(pre)   [only |x|>0.3]')
for l in range(24):
    v = [mix2[l, inl].mean(), mlp2[l, inl].mean(), mix2[l, ~inl].mean(), mlp2[l, ~inl].mean()]
    if max(abs(x) for x in v) > 0.3: print('%2d | %6.2f %6.2f | %6.2f %6.2f' % (l, *v))
```