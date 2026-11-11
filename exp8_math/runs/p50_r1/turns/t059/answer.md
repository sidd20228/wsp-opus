Confirmed: `hidden_24` is the post-norm state, since E·h24 reproduces the recorded log-probabilities. The residual stream itself is small, with an RMS of about 0.29 at layer 23. My earlier attributions were therefore divided by about 4 instead of about 0.3, so they are understated roughly 13-fold. More importantly, layer 23's own mixer output was not measured.

I'll redo the attribution properly. The normalised final direction is y = h24/g = x_final/s. I'll estimate the scale s per position by least squares from the known parts (the state after layer 22 plus layer 23's MLP output, which I compute from its recorded activations). Layer 23's mixer contribution then follows as the remainder that closes the exact total logit difference, which is (E_Wait − Ē_alt)·h24.

This yields contributions in nats per layer and per neuron, inside and before the loop. I expect a few large positive drivers: the layer-19 and layer-23 induction attention mainly inside the loop, and the layer-16 MLP everywhere, against negative MLPs at layers 22–23. Neurons that drive "Wait" consistently, both before and inside the loop, would be candidates for the doubt drive.

```python
# Proper DLA (nats) for Wait-vs-competitors: estimate final-norm scale s per position; L23 mixer = remainder; then top neurons.
P_ = len(POS)
Wd23 = load('model.language_model.layers.23.mlp.down_proj.weight').numpy()
A23_ = np.load('/work/fwd_5/mlp_act_23.npy', mmap_mode='r')
acts23 = np.stack([np.asarray(A23_[i, t], dtype=np.float32) for i, t, _ in POS])
m23 = acts23 @ Wd23.T                                        # (P,1024) layer-23 MLP output
h23s = np.stack([Hs[23][i, t] for i, t, _ in POS]); h24s = np.stack([Hs[24][i, t] for i, t, _ in POS])
Y = h24s / gfin
s_est = ((Y * (h23s + m23)).sum(1) / (Y * Y).sum(1))
Uraw = U                                                      # already g*(E_wait - ua)
total = ((Ew[13428].numpy()[None] - (U / gfin - Ew[13428].numpy()[None]) * 0) * 0).sum(1)  # placeholder
ua_ = Ew[13428].numpy()[None] - U / gfin                      # recover ua
total = ((Ew[13428].numpy()[None] - ua_) * h24s).sum(1)       # exact logit difference
scale = rms / s_est                                            # earlier numbers were divided by rms(h24); fix to s
emb = (np.stack([Hs[0][i, t] for i, t, _ in POS]) * U).sum(1) / s_est
mix = layer_mix * scale[None]; mlp = layer_mlp * scale[None]
mix[23] = total - (emb + mix[:23].sum(0) + mlp.sum(0))         # remainder = layer-23 mixer
print('s_est median %.3f; total Wait-vs-alt logit diff: loop %.2f pre %.2f; embedding %.2f/%.2f' % (np.median(s_est), total[inl].mean(), total[~inl].mean(), emb[inl].mean(), emb[~inl].mean()))
print('layer | mixer(loop) mlp(loop) | mixer(pre) mlp(pre)   [only |x|>0.3 shown]')
for l in range(24):
    v = [mix[l, inl].mean(), mlp[l, inl].mean(), mix[l, ~inl].mean(), mlp[l, ~inl].mean()]
    if max(abs(x) for x in v) > 0.3: print('%2d | %6.2f %6.2f | %6.2f %6.2f' % (l, *v))
```