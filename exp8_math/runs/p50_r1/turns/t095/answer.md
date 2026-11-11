The direct effect of MLP neurons on the copy is negligible. Summed over all 3,584 neurons, no layer from 8 to 23 contributes more than about 0.06 logits to the copy margin, either at the loop's first cycle or in late cycles, and the largest single-neuron change is 0.009. The confident copying (log-odds around 6 by cycle 3) must therefore come from the token-mixing blocks: full attention and the Gated DeltaNet layers. That fits the classic induction-copy picture of loops, and it means the mid-layer repetition neurons could act only indirectly. To locate the copy signal, I will decompose each layer's write to the residual stream, recorded in pass 2, into its mixer part (layer write minus the MLP's own output) and its MLP part. I will then project both onto the same copy-margin direction for cycle 0, late cycles and ordinary pre-loop text. The mixer layers whose copy contribution jumps from pre-loop to late-loop are where loop-sustaining copying happens.

```python
# Hypothesis: the copy margin in loops is carried by a few token-mixer blocks (full-attn and/or GatedDeltaNet), growing from cycle0 to late cycles.
out = []
Hprev = np.load('/work/fwd_5/hidden_0.npy', mmap_mode='r')
for l in range(24):
    Hn = np.load(f'/work/fwd_5/hidden_{l+1}.npy', mmap_mode='r')
    Wd_l = load_original(f'model.language_model.layers.{l}.mlp.down_proj.weight').numpy(); Aact = np.load(f'/work/fwd_5/mlp_act_{l}.npy', mmap_mode='r')
    acc_ = np.zeros((2, 3))
    for t in range(2):
        T_ = lens5[t] - 1; dl = np.asarray(Hn[t, :T_], np.float32) - np.asarray(Hprev[t, :T_], np.float32)
        mlp_o = np.asarray(Aact[t, :T_], np.float32) @ Wd_l.T; mix_o = dl - mlp_o
        cm = (mix_o * dirs5[t]).sum(1); cl = (mlp_o * dirs5[t]).sum(1)
        m0, m3 = cyc_masks(t); pre = np.zeros(T_, bool); pre[100:on5[t] - 1] = True
        acc_[0] += [cm[pre].mean()/2, cm[m0].mean()/2, cm[m3].mean()/2]; acc_[1] += [cl[pre].mean()/2, cl[m0].mean()/2, cl[m3].mean()/2]
    out.append((l, 'A' if l in (3, 7, 11, 15, 19, 23) else 'D', *np.round(acc_[0], 2), *np.round(acc_[1], 2)))
    Hprev = Hn
OD = pd.DataFrame(out, columns=['layer', 'type', 'mix_pre', 'mix_c0', 'mix_late', 'mlp_pre', 'mlp_c0', 'mlp_late'])
print(OD.to_string()); print('sum', OD[['mix_pre', 'mix_c0', 'mix_late', 'mlp_pre', 'mlp_c0', 'mlp_late']].sum().round(2).to_dict())
```