The surviving restatement loops have no dominant opener comparable to "Wait" in the original model, where "Wait, if/I/is/let's" opened more than half the loop units. Their line openers are spread over "Let's check/try" (71 units), "So" (57), "If ..." (49), "This" (35), "Is it", "Maybe" and others. Many of these are doubt rephrased ("If the...", "Is it...", "Maybe the...", "Let's check..."). So the brake suppressed the doubt words, but the underlying re-verification state partly re-expresses itself through other openers. A second token-level brake would chase a moving target. The more general lever is the loop-specific copy amplification I just found in the final layer (+5 nats only in late cycles). Its usefulness depends on which part of layer 23 does it. If its MLP normally trims copy confidence (pre-loop 5.53 → 4.97) and that trimming disappears in loops while attention adds the rest, the repair would target different weights than if attention alone were responsible. Under the approximation that layer 23 roughly preserves the residual norm, I will estimate the MLP's direct share of the layer-23 margin change, with the attention share as the remainder.

```python
# Hypothesis: in the last layer, MLP23 trims copy confidence on normal text while attention23 supplies the loop-only +5 nats amplification.
Wd23_ = load_original('model.language_model.layers.23.mlp.down_proj.weight').numpy(); A23m = np.load('/work/fwd_5/mlp_act_23.npy', mmap_mode='r')
H24a = np.load('/work/fwd_5/hidden_24.npy', mmap_mode='r'); res = np.zeros((3, 3))
for t in range(2):
    T_ = lens5[t] - 1; X_ = np.asarray(H23[t, :T_], np.float32); s23 = np.sqrt((X_**2).mean(1) + 1e-6)
    D_ = Dd5[t] * wf1; lens23 = (X_ * D_).sum(1) / s23; lens24 = (np.asarray(H24a[t, :T_], np.float32) * Dd5[t]).sum(1)
    mlp_c = ((np.asarray(A23m[t, :T_], np.float32) @ Wd23_.T) * D_).sum(1) / s23
    m0, m3 = cyc_masks(t); pre = np.zeros(T_, bool); pre[100:on5[t]-1] = True
    for j, m in enumerate([pre, m0, m3]): res[j] += np.array([(lens24 - lens23)[m].mean(), mlp_c[m].mean(), (lens24 - lens23 - mlp_c)[m].mean()]) / 2
print(pd.DataFrame(res.round(2), index=['pre', 'cycle0', 'late'], columns=['L23 total', 'MLP23 (approx)', 'attn23+norm (remainder)']).to_string())
```